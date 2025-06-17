package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class ObjectDetector(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null

    // YOLOv11m configuration
    private val inputSize = 640  // YOLOv11 default input size
    private val maxDetections = 300
    private val scoreThreshold = 0.25f
    private val iouThreshold = 0.45f

    // YOLOv11 output format: [batch, detections, (x, y, w, h, conf, class_probs...)]
    private val outputBoxes = 8400  // 8400 detections for 640x640 input
    private val numClasses = 32     // Sesuai dengan jumlah kelas sign language Anda
    private val outputSize = 4 + 1 + numClasses  // bbox(4) + confidence(1) + classes(32)

    // Map of class indices to labels - sesuai dengan training Anda
    private val labelMap = mapOf(
        0 to "a", 1 to "b", 2 to "c", 3 to "d", 4 to "e", 5 to "f",
        6 to "g", 7 to "h", 8 to "i", 9 to "j", 10 to "k", 11 to "l",
        12 to "m", 13 to "n", 14 to "o", 15 to "p", 16 to "q",
        17 to "r", 18 to "s", 19 to "t", 20 to "u",
        21 to "v", 22 to "w", 23 to "x", 24 to "y", 25 to "z"
    )

    init {
        try {
            // Load model dari assets
            val model = FileUtil.loadMappedFile(context, modelPath)

            // Configure interpreter options untuk LiteRT
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Gunakan 4 threads untuk performa lebih baik
                // Gunakan GPU delegate jika tersedia
                try {
                    // Uncomment jika ingin menggunakan GPU delegate
                    // addDelegate(GpuDelegate())
                } catch (e: Exception) {
                    println("GPU delegate not available, using CPU")
                }
            }

            interpreter = Interpreter(model, options)

            // Print input/output tensor info untuk debugging
            val inputTensor = interpreter?.getInputTensor(0)
            val outputTensor = interpreter?.getOutputTensor(0)

            println("YOLOv11 Input tensor shape: ${inputTensor?.shape()?.contentToString()}")
            println("YOLOv11 Output tensor shape: ${outputTensor?.shape()?.contentToString()}")

            // Verify tensor shapes
            val inputShape = inputTensor?.shape()
            if (inputShape != null) {
                println("Expected input: [1, 3, 640, 640], Got: ${inputShape.contentToString()}")
            }

        } catch (e: Exception) {
            e.printStackTrace()
            println("Error loading YOLOv11 model: ${e.message}")
        }
    }

    data class DetectionResult(
        val boundingBox: RectF,
        val label: String,
        val score: Float
    )

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize bitmap ke 640x640 (YOLOv11 input size)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // YOLOv11 expects input format: [1, 3, 640, 640] (NCHW format)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 1 * 3 * inputSize * inputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to NCHW format dan normalize ke [0, 1]
        // Order: semua R values, kemudian semua G values, kemudian semua B values

        // Red channel
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            byteBuffer.putFloat(r)
        }

        // Green channel
        for (pixel in pixels) {
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            byteBuffer.putFloat(g)
        }

        // Blue channel
        for (pixel in pixels) {
            val b = (pixel and 0xFF) / 255.0f
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    private fun postprocessDetections(
        output: Array<Array<FloatArray>>,
        originalWidth: Int,
        originalHeight: Int
    ): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()

        // YOLOv11 output format: [1, 8400, 37] dimana 37 = 4(bbox) + 1(conf) + 32(classes)
        val predictions = output[0] // Shape: [8400, 37]

        for (i in predictions.indices) {
            val prediction = predictions[i]

            // Extract bbox (center_x, center_y, width, height) - normalized [0,1]
            val centerX = prediction[0]
            val centerY = prediction[1]
            val width = prediction[2]
            val height = prediction[3]

            // Extract objectness confidence
            val objectness = prediction[4]

            // Skip jika objectness rendah
            if (objectness < scoreThreshold) continue

            // Find class dengan confidence tertinggi
            var maxClassProb = 0f
            var maxClassIndex = 0

            for (j in 5 until prediction.size) {
                if (prediction[j] > maxClassProb) {
                    maxClassProb = prediction[j]
                    maxClassIndex = j - 5  // Offset karena class mulai dari index 5
                }
            }

            // Calculate final confidence score
            val finalScore = objectness * maxClassProb

            // Skip jika final score rendah
            if (finalScore < scoreThreshold) continue

            // Convert dari center format ke corner format
            val xMin = (centerX - width / 2) * originalWidth
            val yMin = (centerY - height / 2) * originalHeight
            val xMax = (centerX + width / 2) * originalWidth
            val yMax = (centerY + height / 2) * originalHeight

            // Clamp coordinates
            val clampedXMin = max(0f, min(xMin, originalWidth.toFloat()))
            val clampedYMin = max(0f, min(yMin, originalHeight.toFloat()))
            val clampedXMax = max(clampedXMin, min(xMax, originalWidth.toFloat()))
            val clampedYMax = max(clampedYMin, min(yMax, originalHeight.toFloat()))

            val boundingBox = RectF(clampedXMin, clampedYMin, clampedXMax, clampedYMax)
            val label = labelMap[maxClassIndex] ?: "unknown"

            detections.add(DetectionResult(boundingBox, label, finalScore))
        }

        // Apply Non-Maximum Suppression (NMS)
        return applyNMS(detections)
    }

    private fun applyNMS(detections: List<DetectionResult>): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence score (descending)
        val sortedDetections = detections.sortedByDescending { it.score }
        val finalDetections = mutableListOf<DetectionResult>()
        val suppressed = BooleanArray(sortedDetections.size)

        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue

            finalDetections.add(sortedDetections[i])

            // Suppress overlapping detections
            for (j in i + 1 until sortedDetections.size) {
                if (suppressed[j]) continue

                val iou = calculateIoU(sortedDetections[i].boundingBox, sortedDetections[j].boundingBox)
                if (iou > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        return finalDetections.take(maxDetections)
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)

        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }

        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        if (interpreter == null) {
            println("Interpreter is null")
            return emptyList()
        }

        try {
            // Preprocess image
            val inputBuffer = preprocessImage(bitmap)

            // Prepare output array untuk YOLOv11
            // Output shape: [1, 8400, 37] where 37 = 4(bbox) + 1(conf) + 32(classes)
            val output = Array(1) { Array(outputBoxes) { FloatArray(outputSize) } }

            // Run inference
            interpreter?.run(inputBuffer, output)

            // Postprocess results
            return postprocessDetections(output, bitmap.width, bitmap.height)

        } catch (e: Exception) {
            e.printStackTrace()
            println("Error during YOLOv11 detection: ${e.message}")
            return emptyList()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}