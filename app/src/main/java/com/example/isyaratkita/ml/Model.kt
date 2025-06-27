package com.example.isyaratkita.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
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
        private const val INPUT_SIZE = 640
        private const val NUM_CHANNELS = 3
        private const val CONF_THRESHOLD = 0.50f // Anda bisa sesuaikan threshold ini jika perlu

        fun newInstance(context: Context): Model {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    addDelegate(GpuDelegate())
                    Log.d(TAG, "GPU Delegate is enabled.")
                } else {
                    Log.d(TAG, "GPU Delegate is not supported, using CPU.")
                }
            }

            try {
                val modelBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)
                val interpreter = Interpreter(modelBuffer, options)
                val labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
                return Model(interpreter, labels)
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TFLite Model: ${e.message}")
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
        // Bitmap yang masuk diasumsikan sudah berukuran INPUT_SIZE x INPUT_SIZE
        // Alokasi buffer untuk Byte (1 byte per channel), bukan Float (4 bytes).
        val inputBuffer = ByteBuffer.allocateDirect(1 * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Ekstrak channel warna dan masukan sebagai Byte
            inputBuffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            inputBuffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            inputBuffer.put((pixel and 0xFF).toByte())          // B
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
                val xCenter = box[0] * INPUT_SIZE
                val yCenter = box[1] * INPUT_SIZE
                val w = box[2] * INPUT_SIZE
                val h = box[3] * INPUT_SIZE

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
