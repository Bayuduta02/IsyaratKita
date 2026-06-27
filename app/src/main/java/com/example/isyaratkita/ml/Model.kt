package com.example.isyaratkita.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Model private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {

    companion object {
        private const val TAG = "Model"
        private const val MODEL_NAME = "best_int8.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f

        fun newInstance(context: Context): Model {
            val assetFileDescriptor = context.assets.openFd(MODEL_NAME)
            val inputStream = assetFileDescriptor.createInputStream()
            val modelBytes = inputStream.readBytes()
            inputStream.close()
            assetFileDescriptor.close()

            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            if (modelBuffer.capacity() < 1000) {
                throw RuntimeException(
                    "File $MODEL_NAME terlalu kecil (${modelBuffer.capacity()} bytes). " +
                            "Pastikan file model sudah benar di assets/$MODEL_NAME"
                )
            }

            Log.i(TAG, "✓ Model loaded: ${modelBuffer.capacity() / 1024}KB")

            val labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
            Log.i(TAG, "✓ Labels loaded: ${labels.size} classes")

            val options = Interpreter.Options().apply {
                numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
                useXNNPACK = true
            }

            return try {
                val interpreter = Interpreter(modelBuffer, options)
                Log.i(TAG, "✓ Model initialized with CPU (XNNPACK enabled)")
                Model(interpreter, labels)
            } catch (e: Exception) {
                val errorMessage = when {
                    e.message?.contains("Could not find") == true ->
                        "Model file tidak ditemukan. Pastikan $MODEL_NAME ada di folder assets."
                    e.message?.contains("Error loading model") == true ->
                        "Error memuat model. File mungkin rusak atau tidak kompatibel."
                    else -> "Error initializing TFLite Model: ${e.message}"
                }
                Log.e(TAG, errorMessage, e)
                throw RuntimeException(errorMessage)
            }
        }
    }

    private val inputShape = interpreter.getInputTensor(0).shape()
    private val inputDataType = interpreter.getInputTensor(0).dataType()
    private val outputShape = interpreter.getOutputTensor(0).shape()

    private val inputHeight: Int
    private val inputWidth: Int
    private val channels: Int
    private val isChannelLast: Boolean

    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val reusableBitmap: Bitmap
    private val canvas: Canvas
    private val srcRect = Rect()
    private val dstRect = Rect()

    init {
        if (inputShape.size != 4) {
            throw IllegalStateException("Expected 4D input tensor, got: ${inputShape.toList()}")
        }

        when {
            inputShape[3] == 3 -> {
                inputHeight = inputShape[1]
                inputWidth = inputShape[2]
                channels = inputShape[3]
                isChannelLast = true
            }
            inputShape[1] == 3 -> {
                inputHeight = inputShape[2]
                inputWidth = inputShape[3]
                channels = inputShape[1]
                isChannelLast = false
            }
            else -> throw IllegalStateException("Unsupported input shape: ${inputShape.toList()}")
        }

        inputBuffer = ByteBuffer.allocateDirect(interpreter.getInputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

        outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(4) { acc, dim -> acc * dim })
            .order(ByteOrder.nativeOrder())

        reusableBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        canvas = Canvas(reusableBitmap)
        dstRect.set(0, 0, inputWidth, inputHeight)

        Log.i(TAG, "Model ready: ${inputWidth}x${inputHeight}, channels=$channels, " +
                "format=${if (isChannelLast) "NHWC" else "NCHW"}, dataType=$inputDataType")
    }

    data class DetectionResult(
        val classIndex: Int,
        val score: Float,
        val boundingBox: FloatArray,
        val label: String
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DetectionResult
            return classIndex == other.classIndex &&
                    score == other.score &&
                    boundingBox.contentEquals(other.boundingBox) &&
                    label == other.label
        }

        override fun hashCode(): Int {
            var result = classIndex
            result = 31 * result + score.hashCode()
            result = 31 * result + boundingBox.contentHashCode()
            result = 31 * result + label.hashCode()
            return result
        }
    }

    fun process(bitmap: Bitmap): Pair<List<DetectionResult>, Long> {
        val startTime = SystemClock.elapsedRealtimeNanos()

        val prepStart = SystemClock.elapsedRealtimeNanos()
        preprocessImage(bitmap)
        val prepTime = (SystemClock.elapsedRealtimeNanos() - prepStart) / 1_000_000

        val infStart = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        val infTime = (SystemClock.elapsedRealtimeNanos() - infStart) / 1_000_000

        val postStart = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        val detections = postprocessDetections(outputBuffer)
        val postTime = (SystemClock.elapsedRealtimeNanos() - postStart) / 1_000_000

        val totalTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000

        Log.d(TAG, "Prep:${prepTime}ms | Inf:${infTime}ms | Post:${postTime}ms | Total:${totalTime}ms | Det:${detections.size}")

        return Pair(detections, totalTime)
    }

    private fun preprocessImage(bitmap: Bitmap) {
        inputBuffer.rewind()

        if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
            srcRect.set(0, 0, bitmap.width, bitmap.height)
            canvas.drawBitmap(bitmap, srcRect, dstRect, null)
        } else {
            canvas.drawBitmap(bitmap, 0f, 0f, null)
        }

        val pixelCount = inputWidth * inputHeight
        val pixels = IntArray(pixelCount)
        reusableBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        when (inputDataType) {
            DataType.UINT8 -> {
                if (isChannelLast) {
                    for (pixel in pixels) {
                        inputBuffer.put(((pixel shr 16) and 0xFF).toByte())
                        inputBuffer.put(((pixel shr 8) and 0xFF).toByte())
                        inputBuffer.put((pixel and 0xFF).toByte())
                    }
                } else {
                    for (channel in 0 until 3) {
                        val shift = 16 - (channel * 8)
                        for (pixel in pixels) {
                            inputBuffer.put(((pixel shr shift) and 0xFF).toByte())
                        }
                    }
                }
            }

            DataType.FLOAT32 -> {
                val floatBuffer = inputBuffer.asFloatBuffer()
                val scale = 1f / 255f

                if (isChannelLast) {
                    for (pixel in pixels) {
                        floatBuffer.put(((pixel shr 16) and 0xFF) * scale)
                        floatBuffer.put(((pixel shr 8) and 0xFF) * scale)
                        floatBuffer.put((pixel and 0xFF) * scale)
                    }
                } else {
                    for (channel in 0 until 3) {
                        val shift = 16 - (channel * 8)
                        for (pixel in pixels) {
                            floatBuffer.put(((pixel shr shift) and 0xFF) * scale)
                        }
                    }
                }
            }

            else -> throw IllegalStateException("Unsupported data type: $inputDataType")
        }

        inputBuffer.rewind()
    }

    private fun postprocessDetections(outputBuffer: ByteBuffer): List<DetectionResult> {
        val numBoxes = outputShape[2]
        val numClasses = outputShape[1] - 4
        val floatBuffer = outputBuffer.asFloatBuffer()

        Log.i(TAG, "Output shape: [${outputShape[0]}, ${outputShape[1]}, ${outputShape[2]}] -> $numClasses classes, $numBoxes boxes")

        if (numClasses != labels.size) {
            Log.w(TAG, "Warning: model has $numClasses classes but labels.txt has ${labels.size} entries!")
        }

        val candidates = mutableListOf<DetectionResult>()
        val inputHeightF = inputHeight.toFloat()
        val inputWidthF = inputWidth.toFloat()

        for (boxIdx in 0 until numBoxes) {
            val x = floatBuffer.get(boxIdx)
            val y = floatBuffer.get(numBoxes + boxIdx)
            val w = floatBuffer.get(2 * numBoxes + boxIdx)
            val h = floatBuffer.get(3 * numBoxes + boxIdx)

            var maxScore = 0f
            var classIndex = -1

            for (classIdx in 0 until numClasses) {
                val score = floatBuffer.get((4 + classIdx) * numBoxes + boxIdx)
                if (score > maxScore) {
                    maxScore = score
                    classIndex = classIdx
                }
            }

            if (maxScore >= CONF_THRESHOLD && classIndex in labels.indices) {
                val xCenter = x * inputWidthF
                val yCenter = y * inputHeightF
                val width = w * inputWidthF
                val height = h * inputHeightF

                val x1 = xCenter - width / 2
                val y1 = yCenter - height / 2
                val x2 = xCenter + width / 2
                val y2 = yCenter + height / 2

                if (candidates.isEmpty()) {
                    Log.d(TAG, "First detection: x=$x, y=$y, w=$w, h=$h -> " +
                            "x1=$x1, y1=$y1, x2=$x2, y2=$y2, score=$maxScore, class=${labels[classIndex]}")
                }

                candidates.add(
                    DetectionResult(
                        classIndex,
                        maxScore,
                        floatArrayOf(x1, y1, x2, y2),
                        labels[classIndex]
                    )
                )
            }
        }

        Log.d(TAG, "Candidates found: ${candidates.size} (before NMS)")

        return if (candidates.size > 1) applyNMS(candidates) else candidates
    }

    private fun applyNMS(detections: List<DetectionResult>): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.score }
        val keep = mutableListOf<DetectionResult>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            keep.add(sorted[i])

            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (sorted[i].classIndex == sorted[j].classIndex) {
                    val iou = calculateIoU(sorted[i].boundingBox, sorted[j].boundingBox)
                    if (iou > IOU_THRESHOLD) {
                        suppressed[j] = true
                    }
                }
            }
        }

        return keep
    }

    private fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
        val x1 = maxOf(box1[0], box2[0])
        val y1 = maxOf(box1[1], box2[1])
        val x2 = minOf(box1[2], box2[2])
        val y2 = minOf(box1[3], box2[3])

        if (x2 <= x1 || y2 <= y1) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        val union = area1 + area2 - intersection

        return if (union > 0f) intersection / union else 0f
    }

    fun close() {
        interpreter.close()
        reusableBitmap.recycle()
    }
}