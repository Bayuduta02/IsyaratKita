package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class ObjectDetector(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null
    private val inputSize = 224
    private val maxDetections = 5
    private val scoreThreshold = 0.5f

    // Map of class indices to labels - sesuai dengan training Anda
    private val labelMap = mapOf(
        0 to "a", 1 to "apa", 2 to "b", 3 to "c", 4 to "d", 5 to "e",
        6 to "f", 7 to "g", 8 to "h", 9 to "halo", 10 to "i", 11 to "j",
        12 to "k", 13 to "kabar", 14 to "l", 15 to "m", 16 to "n",
        17 to "nama", 18 to "o", 19 to "p", 20 to "perkenalkan",
        21 to "q", 22 to "r", 23 to "s", 24 to "saya", 25 to "t",
        26 to "u", 27 to "v", 28 to "w", 29 to "x", 30 to "y", 31 to "z"
    )

    init {
        try {
            // Load model dari assets
            val model = FileUtil.loadMappedFile(context, modelPath)

            // Configure interpreter options
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Gunakan 4 threads untuk performa lebih baik
                setUseNNAPI(true) // Gunakan NNAPI jika tersedia
            }

            interpreter = Interpreter(model, options)

            // Print input/output tensor info untuk debugging
            val inputTensor = interpreter?.getInputTensor(0)
            val outputTensor1 = interpreter?.getOutputTensor(0) // bbox output
            val outputTensor2 = interpreter?.getOutputTensor(1) // class output

            println("Input tensor shape: ${inputTensor?.shape()?.contentToString()}")
            println("Output tensor 1 (bbox) shape: ${outputTensor1?.shape()?.contentToString()}")
            println("Output tensor 2 (class) shape: ${outputTensor2?.shape()?.contentToString()}")

        } catch (e: Exception) {
            e.printStackTrace()
            println("Error loading model: ${e.message}")
        }
    }

    data class DetectionResult(
        val boundingBox: RectF,
        val label: String,
        val score: Float
    )

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize bitmap ke 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Buat ByteBuffer untuk input model
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap ke ByteBuffer dengan normalisasi [0, 1]
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Extract RGB values dan normalize ke [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    private fun postprocessDetections(
        bboxOutput: Array<FloatArray>,
        classOutput: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()

        // Get predictions
        val bbox = bboxOutput[0] // [xmin, ymin, xmax, ymax] dalam format normalized
        val classProbs = classOutput[0] // probabilitas untuk setiap kelas

        // Find class dengan probability tertinggi
        var maxProbIndex = 0
        var maxProb = classProbs[0]

        for (i in 1 until classProbs.size) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i]
                maxProbIndex = i
            }
        }

        // Jika confidence di atas threshold, buat detection result
        if (maxProb >= scoreThreshold) {
            // Convert normalized coordinates ke pixel coordinates
            val xmin = bbox[0] * originalWidth
            val ymin = bbox[1] * originalHeight
            val xmax = bbox[2] * originalWidth
            val ymax = bbox[3] * originalHeight

            // Pastikan coordinates dalam bounds
            val clampedXmin = max(0f, min(xmin, originalWidth.toFloat()))
            val clampedYmin = max(0f, min(ymin, originalHeight.toFloat()))
            val clampedXmax = max(clampedXmin, min(xmax, originalWidth.toFloat()))
            val clampedYmax = max(clampedYmin, min(ymax, originalHeight.toFloat()))

            val boundingBox = RectF(clampedXmin, clampedYmin, clampedXmax, clampedYmax)
            val label = labelMap[maxProbIndex] ?: "unknown"

            results.add(DetectionResult(boundingBox, label, maxProb))
        }

        return results
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        if (interpreter == null) {
            println("Interpreter is null")
            return emptyList()
        }

        try {
            // Preprocess image
            val inputBuffer = preprocessImage(bitmap)

            // Prepare output arrays
            val bboxOutput = Array(1) { FloatArray(4) } // [batch, 4] untuk bbox
            val classOutput = Array(1) { FloatArray(labelMap.size) } // [batch, num_classes]

            // Run inference
            val inputs = arrayOf(inputBuffer)
            val outputs = mapOf(
                0 to bboxOutput,  // bbox output
                1 to classOutput  // class output
            )

            interpreter?.runForMultipleInputsOutputs(inputs, outputs)

            // Postprocess results
            return postprocessDetections(bboxOutput, classOutput, bitmap.width, bitmap.height)

        } catch (e: Exception) {
            e.printStackTrace()
            println("Error during detection: ${e.message}")
            return emptyList()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}