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

class Model private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {

    companion object {
        private const val TAG = "Model"
        private const val MODEL_NAME = "model.tflite" // Pastikan nama file TFLite Anda sesuai
        private const val LABELS_FILE = "labels.txt"

        // Parameter ini sesuai dengan metadata.yaml Anda
        private const val CONF_THRESHOLD = 0.30f // Anda bisa sesuaikan threshold ini jika perlu

        fun newInstance(context: Context): Model {
            val options = Interpreter.Options().apply {
                setNumThreads(4)

                // Coba gunakan NNAPI terlebih dahulu (Android Neural Networks API)
                try {
                    val nnApiDelegate = org.tensorflow.lite.nnapi.NnApiDelegate()
                    addDelegate(nnApiDelegate)
                    Log.d(TAG, "NNAPI Delegate is enabled.")
                } catch (e: Exception) {
                    Log.d(TAG, "NNAPI Delegate tidak tersedia: ${e.message}")

                    // Fallback ke GPU jika NNAPI tidak tersedia
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        try {
                            val gpuDelegate = GpuDelegate()
                            addDelegate(gpuDelegate)
                            Log.d(TAG, "GPU Delegate is enabled.")
                        } catch (e: Exception) {
                            Log.d(TAG, "Error saat menggunakan GPU Delegate: ${e.message}")
                            Log.d(TAG, "Fallback ke CPU.")
                        }
                    } else {
                        Log.d(TAG, "GPU Delegate is not supported, using CPU.")
                    }
                }
            }

            try {
                val modelBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)
                val interpreter = Interpreter(modelBuffer, options)
                val labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
                return Model(interpreter, labels)
            } catch (e: Exception) {
                val errorMessage = when {
                    e.message?.contains("Could not find") == true ->
                        "Model file tidak ditemukan. Pastikan file $MODEL_NAME ada di folder assets."
                    e.message?.contains("Error loading model") == true ->
                        "Error saat memuat model. Model mungkin rusak atau tidak kompatibel."
                    e.message?.contains("labels") == true || e.message?.contains("LABELS") == true ->
                        "File label tidak ditemukan. Pastikan file $LABELS_FILE ada di folder assets."
                    else -> "Error initializing TFLite Model: ${e.message}"
                }
                Log.e(TAG, errorMessage, e)
                throw RuntimeException(errorMessage)
            }
        }
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
            if (classIndex != other.classIndex) return false
            if (score != other.score) return false
            if (!boundingBox.contentEquals(other.boundingBox)) return false
            if (label != other.label) return false
            return true
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
        val inputBuffer = preprocessImage(bitmap)

        // Output model tetap float, jadi bagian ini tidak berubah
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(4) { acc, dim -> acc * dim })
            .order(ByteOrder.nativeOrder())

        val startTime = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuffer, outputBuffer)
        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000

        outputBuffer.rewind()
        val detections = postprocessDetections(outputBuffer, outputShape)

        return Pair(detections, inferenceTime)
    }

    /**
     * --- PERBAIKAN TOTAL UNTUK MODEL INT8 ---
     * Fungsi ini diubah untuk model int8 (terkuantisasi).
     * Tidak ada lagi normalisasi float (pembagian dengan 255.0f).
     * Data piksel (0-255) langsung dimasukkan sebagai Byte.
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val inputDataType = inputTensor.dataType()

        if (inputShape.size != 4) {
            throw IllegalStateException("Expected input tensor to have 4 dimensions but was: ${inputShape.toList()}")
        }

        val height: Int
        val width: Int
        val channels: Int
        val isChannelLast: Boolean

        when {
            inputShape[3] == 3 -> {
                height = inputShape[1]
                width = inputShape[2]
                channels = inputShape[3]
                isChannelLast = true
            }

            inputShape[1] == 3 -> {
                height = inputShape[2]
                width = inputShape[3]
                channels = inputShape[1]
                isChannelLast = false
            }

            else -> throw IllegalStateException("Unsupported input tensor shape: ${inputShape.toList()}")
        }

        val inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
            .order(ByteOrder.nativeOrder())

        val scaledBitmap = if (bitmap.width != width || bitmap.height != height) {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        } else {
            bitmap
        }

        val pixelCount = width * height
        val pixels = IntArray(pixelCount)
        scaledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        if (scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }

        when (inputDataType) {
            DataType.UINT8 -> {
                if (channels != 3) {
                    throw IllegalStateException("Unsupported channel count for UINT8 input: $channels")
                }

                if (isChannelLast) {
                    for (pixel in pixels) {
                        inputBuffer.put(((pixel shr 16) and 0xFF).toByte()) // R
                        inputBuffer.put(((pixel shr 8) and 0xFF).toByte())  // G
                        inputBuffer.put((pixel and 0xFF).toByte())          // B
                    }
                } else {
                    for (channel in 0 until channels) {
                        for (i in 0 until pixelCount) {
                            val pixel = pixels[i]
                            val component = when (channel) {
                                0 -> (pixel shr 16) and 0xFF
                                1 -> (pixel shr 8) and 0xFF
                                2 -> pixel and 0xFF
                                else -> 0
                            }
                            inputBuffer.put(component.toByte())
                        }
                    }
                }
            }

            DataType.FLOAT32 -> {
                val floatBuffer = inputBuffer.asFloatBuffer()
                if (channels != 3) {
                    throw IllegalStateException("Unsupported channel count for FLOAT32 input: $channels")
                }

                if (isChannelLast) {
                    for (pixel in pixels) {
                        floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
                        floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
                        floatBuffer.put((pixel and 0xFF) / 255f)
                    }
                } else {
                    for (channel in 0 until channels) {
                        for (i in 0 until pixelCount) {
                            val pixel = pixels[i]
                            val component = when (channel) {
                                0 -> (pixel shr 16) and 0xFF
                                1 -> (pixel shr 8) and 0xFF
                                2 -> pixel and 0xFF
                                else -> 0
                            }
                            floatBuffer.put(component / 255f)
                        }
                    }
                }
                floatBuffer.rewind()
            }

            else -> {
                // Tidak mungkin terjadi karena kasus lain ditangani di when sebelumnya
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    /**
     * Fungsi ini memproses output dari model.
     * Karena output model tetap dalam bentuk float, fungsi ini tidak banyak berubah.
     */
    private fun postprocessDetections(
        outputBuffer: ByteBuffer,
        outputShape: IntArray
    ): List<DetectionResult> {
        val inputShape = interpreter.getInputTensor(0).shape()
        if (inputShape.size != 4) {
            throw IllegalStateException("Expected input tensor to have 4 dimensions but was: ${inputShape.toList()}")
        }

        val inputHeight: Float
        val inputWidth: Float

        if (inputShape[3] == 3) {
            inputHeight = inputShape[1].toFloat()
            inputWidth = inputShape[2].toFloat()
        } else if (inputShape[1] == 3) {
            inputHeight = inputShape[2].toFloat()
            inputWidth = inputShape[3].toFloat()
        } else {
            throw IllegalStateException("Unsupported input tensor shape: ${inputShape.toList()}")
        }

        val numBoxes = outputShape[2]
        val floatBuffer = outputBuffer.asFloatBuffer()

        val detections = mutableListOf<DetectionResult>()

        val transposedOutput = Array(numBoxes) { FloatArray(outputShape[1]) }
        for (i in 0 until outputShape[1]) {
            for (j in 0 until numBoxes) {
                transposedOutput[j][i] = floatBuffer.get(i * numBoxes + j)
            }
        }

        for (i in 0 until numBoxes) {
            val detection = transposedOutput[i]

            var maxScore = 0f
            var classIndex = -1
            // Skor kelas dimulai dari indeks ke-4
            for (j in 4 until detection.size) {
                if (detection[j] > maxScore) {
                    maxScore = detection[j]
                    classIndex = j - 4
                }
            }

            if (maxScore >= CONF_THRESHOLD) {
                val box = floatArrayOf(detection[0], detection[1], detection[2], detection[3])

                // Konversi dari [center_x, center_y, width, height] ke [x1, y1, x2, y2]
                // dalam koordinat piksel (relatif ke INPUT_SIZE)
                val xCenter = box[0] * inputWidth
                val yCenter = box[1] * inputHeight
                val w = box[2] * inputWidth
                val h = box[3] * inputHeight

                val x1 = xCenter - w / 2
                val y1 = yCenter - h / 2
                val x2 = xCenter + w / 2
                val y2 = yCenter + h / 2

                // Pastikan classIndex valid sebelum mengakses array labels
                if (classIndex in labels.indices) {
                    detections.add(
                        DetectionResult(
                            classIndex,
                            maxScore,
                            floatArrayOf(x1, y1, x2, y2),
                            labels[classIndex]
                        )
                    )
                }
            }
        }
        return detections
    }

    fun close() {
        interpreter.close()
    }
}
