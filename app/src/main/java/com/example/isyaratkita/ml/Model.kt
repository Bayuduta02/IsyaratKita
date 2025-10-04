package com.example.isyaratkita.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Model private constructor(
    private val interpreter: Interpreter,
    private val gpuDelegate: GpuDelegate?,
    val labels: List<String>
) {

    companion object {
        private const val TAG = "Model"
        private const val MODEL_NAME = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val CONF_THRESHOLD = 0.3f

        fun newInstance(context: Context): Model {
            Log.d(TAG, "Loading model: $MODEL_NAME")
            val modelBuffer = try {
                FileUtil.loadMappedFile(context, MODEL_NAME)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model file: $MODEL_NAME", e)
                throw IllegalStateException("Model file '$MODEL_NAME' not found or corrupted. Please add a valid TFLite model to assets folder.", e)
            }

            Log.d(TAG, "Model buffer loaded: ${modelBuffer.capacity()} bytes")
            if (modelBuffer.capacity() < 100) {
                throw IllegalStateException("Model file is too small (${modelBuffer.capacity()} bytes). It appears to be a placeholder or corrupted.")
            }

            val labels = try {
                FileUtil.loadLabels(context, LABELS_FILE)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load labels file: $LABELS_FILE", e)
                throw IllegalStateException("Labels file '$LABELS_FILE' not found.", e)
            }
            Log.d(TAG, "Loaded ${labels.size} labels: ${labels.take(5).joinToString(", ")}...")

            var gpuDelegate: GpuDelegate? = null
            val compatList = CompatibilityList()

            val interpreter = try {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        try {
                            val delegateOptions = compatList.bestOptionsForThisDevice
                            gpuDelegate = GpuDelegate(delegateOptions)
                            addDelegate(gpuDelegate)
                            Log.d(TAG, "GPU delegate enabled.")
                        } catch (delegateError: Exception) {
                            Log.w(TAG, "Unable to enable GPU delegate: ${delegateError.message}")
                            gpuDelegate?.close()
                            gpuDelegate = null
                        }
                    }
                }
                Interpreter(modelBuffer, options)
            } catch (interpreterError: Exception) {
                Log.w(TAG, "Falling back to CPU interpreter: ${interpreterError.message}", interpreterError)
                gpuDelegate?.close()
                gpuDelegate = null

                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(false)
                    setUseXNNPACK(true)
                }
                Interpreter(modelBuffer, cpuOptions)
            }

            Log.d(TAG, "Interpreter created successfully")
            val model = Model(interpreter, gpuDelegate, labels)
            Log.d(TAG, "Model initialized: Input=${model.inputWidth}x${model.inputHeight}, Output shape=${model.outputShape.contentToString()}")
            return model
        }
    }

    data class DetectionResult(
        val classIndex: Int,
        val score: Float,
        val boundingBox: FloatArray,
        val label: String
    )

    private val inputTensor = interpreter.getInputTensor(0)
    private val inputDataType: DataType = inputTensor.dataType()
    private val inputQuantParams = inputTensor.quantizationParams()
    private val inputShape: IntArray = inputTensor.shape()

    val inputWidth: Int
    val inputHeight: Int
    private val inputChannels: Int

    private val inputByteBuffer: ByteBuffer
    private val inputIntBuffer: IntArray

    private val outputTensor = interpreter.getOutputTensor(0)
    private val outputShape: IntArray = outputTensor.shape()
    private val outputByteBuffer: ByteBuffer = ByteBuffer.allocateDirect(
        outputTensor.numElements() * outputTensor.dataType().byteSize()
    ).order(ByteOrder.nativeOrder())

    private val scratchOutput = FloatArray(outputTensor.numElements())

    init {
        Log.d(TAG, "Input tensor shape: ${inputShape.contentToString()}")
        require(inputShape.size == 4) { "Unsupported input tensor shape: ${inputShape.contentToString()}" }

        val (heightIndex, widthIndex, channelsIndex) = if (inputShape[3] == 3) {
            Triple(1, 2, 3)
        } else {
            Triple(2, 3, 1)
        }
        inputHeight = inputShape[heightIndex]
        inputWidth = inputShape[widthIndex]
        inputChannels = inputShape[channelsIndex]

        require(inputChannels == 3) { "Model expects 3 input channels but found $inputChannels" }

        val inputBytes = inputTensor.numElements() * inputDataType.byteSize()
        inputByteBuffer = ByteBuffer.allocateDirect(inputBytes).order(ByteOrder.nativeOrder())
        inputIntBuffer = IntArray(inputWidth * inputHeight)
    }

    fun process(bitmap: Bitmap): Pair<List<DetectionResult>, Long> {
        if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
            throw IllegalArgumentException("Bitmap size must match model input size $inputWidth x $inputHeight")
        }
        preprocessImage(bitmap)
        outputByteBuffer.rewind()
        val startTime = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputByteBuffer, outputByteBuffer)
        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000

        outputByteBuffer.rewind()
        val detections = postprocessDetections()

        return Pair(detections, inferenceTime)
    }

    private fun preprocessImage(bitmap: Bitmap) {
        inputByteBuffer.rewind()
        bitmap.getPixels(inputIntBuffer, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        when (inputDataType) {
            DataType.FLOAT32 -> {
                val floatBuffer = inputByteBuffer.asFloatBuffer()
                floatBuffer.rewind()
                for (pixel in inputIntBuffer) {
                    floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
                    floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
                    floatBuffer.put((pixel and 0xFF) / 255f)
                }
            }
            DataType.UINT8 -> {
                for (pixel in inputIntBuffer) {
                    inputByteBuffer.put(((pixel shr 16) and 0xFF).toByte())
                    inputByteBuffer.put(((pixel shr 8) and 0xFF).toByte())
                    inputByteBuffer.put((pixel and 0xFF).toByte())
                }
            }
            DataType.INT8 -> {
                val scale = inputQuantParams.scale.toFloat()
                val zeroPoint = inputQuantParams.zeroPoint

                for (pixel in inputIntBuffer) {
                    quantizeChannel((pixel shr 16) and 0xFF, scale, zeroPoint)
                    quantizeChannel((pixel shr 8) and 0xFF, scale, zeroPoint)
                    quantizeChannel(pixel and 0xFF, scale, zeroPoint)
                }
            }
            else -> throw IllegalArgumentException("Unsupported input data type: $inputDataType")
        }

        inputByteBuffer.rewind()
    }

    private fun quantizeChannel(value: Int, scale: Float, zeroPoint: Int) {
        val normalized = value / 255f
        val quantized = (normalized / scale + zeroPoint).roundToInt().coerceIn(-128, 127)
        inputByteBuffer.put(quantized.toByte())
    }

    private fun postprocessDetections(): List<DetectionResult> {
        val floatBuffer = outputByteBuffer.asFloatBuffer()
        floatBuffer.rewind()
        floatBuffer.get(scratchOutput)

        val results = ArrayList<DetectionResult>()
        if (outputShape.size == 3 && outputShape[2] == 6) {
            val maxDet = outputShape[1]
            val stride = 6

            for (index in 0 until maxDet) {
                val base = index * stride
                val left = max(0f, scratchOutput[base] * inputWidth)
                val top = max(0f, scratchOutput[base + 1] * inputHeight)
                val right = min(inputWidth.toFloat(), scratchOutput[base + 2] * inputWidth)
                val bottom = min(inputHeight.toFloat(), scratchOutput[base + 3] * inputHeight)
                val score = scratchOutput[base + 4]
                val classIndex = scratchOutput[base + 5].toInt()

                if (score < CONF_THRESHOLD || classIndex !in labels.indices) continue

                results.add(
                    DetectionResult(
                        classIndex = classIndex,
                        score = score,
                        boundingBox = floatArrayOf(left, top, right, bottom),
                        label = labels[classIndex]
                    )
                )
            }
        } else {
            val boxes = outputShape[1]
            val features = outputShape[2]
            val classLimit = min(features, labels.size + 4)

            for (boxIndex in 0 until boxes) {
                var bestScore = 0f
                var bestClassIndex = -1
                val baseIndex = boxIndex * features

                for (classOffset in 4 until classLimit) {
                    val score = scratchOutput[baseIndex + classOffset]
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIndex = classOffset - 4
                    }
                }

                if (bestScore < CONF_THRESHOLD || bestClassIndex !in labels.indices) continue

                val xCenter = scratchOutput[baseIndex] * inputWidth
                val yCenter = scratchOutput[baseIndex + 1] * inputHeight
                val width = scratchOutput[baseIndex + 2] * inputWidth
                val height = scratchOutput[baseIndex + 3] * inputHeight

                val left = max(0f, xCenter - width / 2f)
                val top = max(0f, yCenter - height / 2f)
                val right = min(inputWidth.toFloat(), xCenter + width / 2f)
                val bottom = min(inputHeight.toFloat(), yCenter + height / 2f)

                results.add(
                    DetectionResult(
                        classIndex = bestClassIndex,
                        score = bestScore,
                        boundingBox = floatArrayOf(left, top, right, bottom),
                        label = labels[bestClassIndex]
                    )
                )
            }
        }
        return results.sortedByDescending { it.score }
    }

    fun close() {
        gpuDelegate?.close()
        interpreter.close()
    }
}
