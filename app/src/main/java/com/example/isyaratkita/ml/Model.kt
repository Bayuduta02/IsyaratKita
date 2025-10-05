package com.example.isyaratkita.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
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
        private const val MODEL_NAME = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val CONF_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.45f

        fun newInstance(context: Context): Model {
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)

            // Validasi ukuran model
            if (modelBuffer.capacity() < 1000) {
                throw RuntimeException("File model.tflite terlalu kecil (${modelBuffer.capacity()} bytes). " +
                        "Model YOLO yang valid biasanya 5-20 MB. Pastikan Anda sudah meng-upload model yang benar ke assets/model.tflite")
            }

            val labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()

            // Untuk model int8 quantized 320x320, gunakan XNNPACK
            val baseOptions = Interpreter.Options().apply {
                setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                setUseXNNPACK(true)
            }

            // Try GPU first (biasanya lebih lambat untuk int8, tapi coba dulu)
            try {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val gpuOptions = Interpreter.Options(baseOptions)
                    val gpuDelegate = GpuDelegate()
                    gpuOptions.addDelegate(gpuDelegate)
                    Log.d(TAG, "Attempting to use GPU Delegate.")

                    val interpreter = Interpreter(modelBuffer, gpuOptions)
                    Log.i(TAG, "✓ Model initialized with GPU delegate.")
                    return Model(interpreter, labels)
                } else {
                    Log.d(TAG, "GPU Delegate not supported on this device.")
                }
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate failed: ${e.message}. Using CPU with XNNPACK.")
            }

            // Fallback to CPU with XNNPACK (optimal untuk int8)
            try {
                Log.d(TAG, "Using CPU with XNNPACK for inference.")
                val interpreter = Interpreter(modelBuffer, baseOptions)
                Log.i(TAG, "✓ Model initialized with CPU (XNNPACK enabled).")
                return Model(interpreter, labels)
            } catch (e: Exception) {
                val errorMessage = when {
                    e.message?.contains("Could not find") == true ->
                        "Model file tidak ditemukan. Pastikan $MODEL_NAME ada di folder assets."
                    e.message?.contains("Error loading model") == true ->
                        "Error memuat model. File mungkin rusak atau tidak kompatibel."
                    e.message?.contains("labels") == true ->
                        "File label tidak ditemukan. Pastikan $LABELS_FILE ada di folder assets."
                    else -> "Error initializing TFLite Model: ${e.message}"
                }
                Log.e(TAG, errorMessage, e)
                throw RuntimeException(errorMessage)
            }
        }
    }

    // Cache input/output tensor info
    private val inputShape = interpreter.getInputTensor(0).shape()
    private val inputDataType = interpreter.getInputTensor(0).dataType()
    private val outputShape = interpreter.getOutputTensor(0).shape()

    private val inputHeight: Int
    private val inputWidth: Int
    private val channels: Int
    private val isChannelLast: Boolean

    // Pre-allocated buffers (reuse untuk setiap inference)
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

        // Parse input shape
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

        // Pre-allocate buffers (sekali saja)
        inputBuffer = ByteBuffer.allocateDirect(interpreter.getInputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

        outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(4) { acc, dim -> acc * dim })
            .order(ByteOrder.nativeOrder())

        // Reusable bitmap dan canvas untuk scaling cepat
        reusableBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        canvas = Canvas(reusableBitmap)
        dstRect.set(0, 0, inputWidth, inputHeight)

        Log.i(TAG, "Model ready: ${inputWidth}x${inputHeight}, channels=$channels, " +
                "format=${if(isChannelLast) "NHWC" else "NCHW"}, dataType=$inputDataType")
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

        // 1. Preprocessing
        val prepStart = SystemClock.elapsedRealtimeNanos()
        preprocessImage(bitmap)
        val prepTime = (SystemClock.elapsedRealtimeNanos() - prepStart) / 1_000_000

        // 2. Inference
        val infStart = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        val infTime = (SystemClock.elapsedRealtimeNanos() - infStart) / 1_000_000

        // 3. Postprocessing
        val postStart = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        val detections = postprocessDetections(outputBuffer)
        val postTime = (SystemClock.elapsedRealtimeNanos() - postStart) / 1_000_000

        val totalTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000

        Log.d(TAG, "⏱ Prep:${prepTime}ms | Inf:${infTime}ms | Post:${postTime}ms | Total:${totalTime}ms | Det:${detections.size}")

        return Pair(detections, totalTime)
    }

    /**
     * Preprocessing optimized dengan Canvas (lebih cepat dari createScaledBitmap)
     */
    private fun preprocessImage(bitmap: Bitmap) {
        inputBuffer.rewind()

        // Fast scaling dengan Canvas
        if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
            srcRect.set(0, 0, bitmap.width, bitmap.height)
            canvas.drawBitmap(bitmap, srcRect, dstRect, null)
        } else {
            canvas.drawBitmap(bitmap, 0f, 0f, null)
        }

        // Copy pixels
        val pixelCount = inputWidth * inputHeight
        val pixels = IntArray(pixelCount)
        reusableBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        when (inputDataType) {
            DataType.UINT8 -> {
                // int8 quantized model - langsung masukkan byte [0-255]
                if (isChannelLast) {
                    // NHWC: [R,G,B, R,G,B, ...]
                    for (pixel in pixels) {
                        inputBuffer.put(((pixel shr 16) and 0xFF).toByte()) // R
                        inputBuffer.put(((pixel shr 8) and 0xFF).toByte())  // G
                        inputBuffer.put((pixel and 0xFF).toByte())          // B
                    }
                } else {
                    // NCHW: [R,R,R... G,G,G... B,B,B...]
                    for (channel in 0 until 3) {
                        val shift = 16 - (channel * 8)
                        for (pixel in pixels) {
                            inputBuffer.put(((pixel shr shift) and 0xFF).toByte())
                        }
                    }
                }
            }

            DataType.FLOAT32 -> {
                // Float model - normalize ke [0, 1]
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

    /**
     * Postprocessing dengan NMS (karena metadata nms: false)
     * Output format YOLOv8: [1, 84, 8400] => [batch, (4 bbox + 80 classes), anchors]
     * Untuk 26 classes: [1, 30, 8400] => [batch, (4 bbox + 26 classes), anchors]
     * Untuk 320x320: anchors sekitar 2100 (80x80/4 + 40x40/4 + 20x20/4)
     */
    private fun postprocessDetections(outputBuffer: ByteBuffer): List<DetectionResult> {
        val numBoxes = outputShape[2]
        val numClasses = outputShape[1] - 4 // 26 classes untuk A-Z
        val floatBuffer = outputBuffer.asFloatBuffer()

        Log.d(TAG, "Output shape: [${outputShape[0]}, ${outputShape[1]}, ${outputShape[2]}] -> numClasses=$numClasses, numBoxes=$numBoxes")

        val candidates = mutableListOf<DetectionResult>()

        val inputHeightF = inputHeight.toFloat()
        val inputWidthF = inputWidth.toFloat()

        // Parse detections
        for (boxIdx in 0 until numBoxes) {
            // Baca bbox coordinates [x_center, y_center, width, height] (normalized 0-1)
            val x = floatBuffer.get(boxIdx)
            val y = floatBuffer.get(numBoxes + boxIdx)
            val w = floatBuffer.get(2 * numBoxes + boxIdx)
            val h = floatBuffer.get(3 * numBoxes + boxIdx)

            // Cari class dengan score tertinggi
            var maxScore = 0f
            var classIndex = -1

            for (classIdx in 0 until numClasses) {
                val score = floatBuffer.get((4 + classIdx) * numBoxes + boxIdx)
                if (score > maxScore) {
                    maxScore = score
                    classIndex = classIdx
                }
            }

            // Filter by confidence
            if (maxScore >= CONF_THRESHOLD && classIndex in labels.indices) {
                // Convert to pixel coordinates [x1, y1, x2, y2]
                // Koordinat dari model kemungkinan sudah normalized [0-1] atau dalam pixel
                val xCenter = x * inputWidthF
                val yCenter = y * inputHeightF
                val width = w * inputWidthF
                val height = h * inputHeightF

                val x1 = xCenter - width / 2
                val y1 = yCenter - height / 2
                val x2 = xCenter + width / 2
                val y2 = yCenter + height / 2

                if (candidates.isEmpty()) {
                    Log.d(TAG, "First detection: x=$x, y=$y, w=$w, h=$h -> x1=$x1, y1=$y1, x2=$x2, y2=$y2, score=$maxScore, class=${labels[classIndex]}")
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

        // Apply NMS
        val finalResults = if (candidates.size > 1) {
            applyNMS(candidates)
        } else {
            candidates
        }

        if (finalResults.isEmpty() && candidates.isEmpty()) {
            Log.d(TAG, "⚠️ No detections above confidence threshold ($CONF_THRESHOLD)")
        }

        return finalResults
    }

    /**
     * Non-Maximum Suppression - menghapus deteksi duplikat
     */
    private fun applyNMS(detections: List<DetectionResult>): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()

        // Sort by score (descending)
        val sorted = detections.sortedByDescending { it.score }
        val keep = mutableListOf<DetectionResult>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue

            keep.add(sorted[i])

            // Suppress overlapping boxes dari class yang sama
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue

                // Hanya NMS untuk class yang sama
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

    /**
     * Calculate Intersection over Union
     */
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