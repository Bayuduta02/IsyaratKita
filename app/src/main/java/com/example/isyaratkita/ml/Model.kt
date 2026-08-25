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

/**
 * Model wrapper untuk YOLO end-to-end TFLite (best_int8.tflite).
 *
 * metadata.yaml:
 *   end2end: true, nms: false, imgsz: [640, 640], quantize: 8
 *
 * Output shape: [1, 300, 6]
 *   format: [x1, y1, x2, y2, score, classIndex]
 *   koordinat: NORMALIZED [0, 1]
 */
class Model private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {

    companion object {
        private const val TAG = "Model"
        private const val MODEL_NAME = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val CONF_THRESHOLD = 0.10f

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
                    "File $MODEL_NAME terlalu kecil (${modelBuffer.capacity()} bytes)."
                )
            }

            Log.i(TAG, "✓ Model loaded: ${modelBuffer.capacity() / 1024}KB")

            val labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
            Log.i(TAG, "✓ Labels loaded: ${labels.size} classes → ${labels.joinToString()}")

            val options = Interpreter.Options().apply {
                numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
                useXNNPACK = true
            }

            return try {
                val interpreter = Interpreter(modelBuffer, options)
                Log.i(TAG, "✓ Interpreter initialized (XNNPACK enabled)")
                Model(interpreter, labels)
            } catch (e: Exception) {
                val msg = "Error initializing TFLite Model: ${e.message}"
                Log.e(TAG, msg, e)
                throw RuntimeException(msg)
            }
        }
    }

    private val inputShape    = interpreter.getInputTensor(0).shape()
    private val inputDType    = interpreter.getInputTensor(0).dataType()
    private val outputCount   = interpreter.getOutputTensorCount()
    private val outputDType   = interpreter.getOutputTensor(0).dataType()

    private val inputHeight: Int
    private val inputWidth: Int
    private val isChannelLast: Boolean

    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private var outputBuffers: Array<Any>? = null

    // Reusable bitmap untuk resize
    private val reusableBitmap: Bitmap
    private val canvas = Canvas()
    private val srcRect = Rect()
    private val dstRect = Rect()

    // Pixel buffer — reuse untuk menghindari alokasi setiap frame
    private lateinit var pixels: IntArray

    init {
        require(inputShape.size == 4) {
            "Expected 4D input tensor, got: ${inputShape.toList()}"
        }

        when {
            inputShape[3] == 3 -> {
                inputHeight = inputShape[1]; inputWidth = inputShape[2]; isChannelLast = true
            }
            inputShape[1] == 3 -> {
                inputHeight = inputShape[2]; inputWidth = inputShape[3]; isChannelLast = false
            }
            else -> throw IllegalStateException("Unsupported input shape: ${inputShape.toList()}")
        }

        inputBuffer = ByteBuffer.allocateDirect(interpreter.getInputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

        if (outputCount > 1) {
            val boxes   = Array(1) { Array(300) { FloatArray(4) } }
            val classes = Array(1) { FloatArray(300) }
            val scores  = Array(1) { FloatArray(300) }
            val count   = FloatArray(1)
            outputBuffers = arrayOf(boxes, classes, scores, count)
            outputBuffer  = ByteBuffer.allocateDirect(0)
        } else {
            outputBuffer = ByteBuffer.allocateDirect(interpreter.getOutputTensor(0).numBytes())
                .order(ByteOrder.nativeOrder())
        }

        reusableBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        canvas.setBitmap(reusableBitmap)
        dstRect.set(0, 0, inputWidth, inputHeight)
        pixels = IntArray(inputWidth * inputHeight)

        Log.i(TAG, "Model ready → input: ${inputWidth}x${inputHeight} " +
                "${if (isChannelLast) "NHWC" else "NCHW"} | inputType=$inputDType | outputType=$outputDType")
        for (i in 0 until outputCount) {
            Log.i(TAG, "  Output[$i] shape: ${interpreter.getOutputTensor(i).shape().contentToString()}")
        }
    }

    data class DetectionResult(
        val classIndex: Int,
        val score: Float,
        /** [x1, y1, x2, y2] dalam pixel ruang inputWidth×inputHeight (640×640) */
        val boundingBox: FloatArray,
        val label: String
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DetectionResult
            return classIndex == other.classIndex && score == other.score &&
                    boundingBox.contentEquals(other.boundingBox) && label == other.label
        }
        override fun hashCode(): Int {
            var r = classIndex
            r = 31 * r + score.hashCode()
            r = 31 * r + boundingBox.contentHashCode()
            r = 31 * r + label.hashCode()
            return r
        }
    }

    fun process(bitmap: Bitmap): Pair<List<DetectionResult>, Long> {
        val t0 = SystemClock.elapsedRealtimeNanos()
        preprocessImage(bitmap)

        val infStart = SystemClock.elapsedRealtimeNanos()
        val detections = if (outputCount > 1) {
            val outputs = mutableMapOf<Int, Any>()
            outputBuffers!!.forEachIndexed { i, buf -> outputs[i] = buf }
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            postprocessMultiOutput()
        } else {
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            postprocessSingleOutput(outputBuffer)
        }
        val infMs   = (SystemClock.elapsedRealtimeNanos() - infStart) / 1_000_000
        val totalMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000

        if (detections.isEmpty()) {
            Log.d(TAG, "Inference: ${infMs}ms | Total: ${totalMs}ms | Detections: 0 | maxScore=${String.format("%.3f", peekMaxScore())}")
        } else {
            Log.d(TAG, "Inference: ${infMs}ms | Total: ${totalMs}ms | Detections: ${detections.size}")
        }
        return Pair(detections, totalMs)
    }

    fun close() {
        interpreter.close()
        reusableBitmap.recycle()
    }

    // ── Pre-processing ────────────────────────────────────────────────────────

    private fun preprocessImage(bitmap: Bitmap) {
        inputBuffer.rewind()

        // Resize ke inputWidth x inputHeight menggunakan canvas reusable
        srcRect.set(0, 0, bitmap.width, bitmap.height)
        canvas.drawBitmap(bitmap, srcRect, dstRect, null)

        // Ambil pixel sekali — reuse array
        reusableBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val qParams   = interpreter.getInputTensor(0).quantizationParams()
        val scale     = qParams.scale
        val zeroPoint = qParams.zeroPoint

        when (inputDType) {
            DataType.FLOAT32 -> {
                // PERBAIKAN: Normalisasi [0, 255] → [0.0, 1.0]
                val fb    = inputBuffer.asFloatBuffer()
                val inv255 = 1f / 255f
                for (px in pixels) {
                    fb.put(((px shr 16) and 0xFF) * inv255)  // R
                    fb.put(((px shr 8)  and 0xFF) * inv255)  // G
                    fb.put((px          and 0xFF) * inv255)  // B
                }
            }
            DataType.UINT8 -> {
                for (px in pixels) {
                    inputBuffer.put(((px shr 16) and 0xFF).toByte())
                    inputBuffer.put(((px shr 8)  and 0xFF).toByte())
                    inputBuffer.put((px           and 0xFF).toByte())
                }
            }
            DataType.INT8 -> {
                for (px in pixels) {
                    val r = (px shr 16) and 0xFF
                    val g = (px shr 8)  and 0xFF
                    val b =  px         and 0xFF
                    if (scale != 0f) {
                        inputBuffer.put(((r / 255f / scale) + zeroPoint).toInt().toByte())
                        inputBuffer.put(((g / 255f / scale) + zeroPoint).toInt().toByte())
                        inputBuffer.put(((b / 255f / scale) + zeroPoint).toInt().toByte())
                    } else {
                        inputBuffer.put((r - 128).toByte())
                        inputBuffer.put((g - 128).toByte())
                        inputBuffer.put((b - 128).toByte())
                    }
                }
            }
            else -> throw IllegalStateException("Unsupported input type: $inputDType")
        }

        inputBuffer.rewind()
    }

    // ── Post-processing ───────────────────────────────────────────────────────

    private fun postprocessMultiOutput(): List<DetectionResult> {
        val boxes   = (outputBuffers!![0] as Array<*>)[0] as Array<FloatArray>
        val classes = (outputBuffers!![1] as Array<*>)[0] as FloatArray
        val scores  = (outputBuffers!![2] as Array<*>)[0] as FloatArray
        val count   = (outputBuffers!![3] as FloatArray)[0].toInt()

        val results = mutableListOf<DetectionResult>()
        for (i in 0 until count) {
            if (scores[i] < CONF_THRESHOLD) continue
            val classIdx = classes[i].toInt()
            if (classIdx !in labels.indices) continue
            val x1 = boxes[i][1] * inputWidth
            val y1 = boxes[i][0] * inputHeight
            val x2 = boxes[i][3] * inputWidth
            val y2 = boxes[i][2] * inputHeight
            results.add(DetectionResult(classIdx, scores[i], floatArrayOf(x1, y1, x2, y2), labels[classIdx]))
        }
        return results.sortedByDescending { it.score }
    }

    private fun peekMaxScore(): Float {
        outputBuffer.rewind()
        val tensor = interpreter.getOutputTensor(0)
        val shape = tensor.shape()
        if (shape.size != 3 || shape[2] != 6) return 0f

        val floatData: FloatArray = if (outputDType == DataType.INT8 || outputDType == DataType.UINT8) {
            val qp = tensor.quantizationParams()
            val raw = ByteArray(outputBuffer.remaining())
            outputBuffer.get(raw)
            FloatArray(raw.size) { i ->
                val v = if (outputDType == DataType.UINT8) (raw[i].toInt() and 0xFF) else raw[i].toInt()
                (v - qp.zeroPoint) * qp.scale
            }
        } else {
            FloatArray(outputBuffer.remaining() / 4).also { outputBuffer.asFloatBuffer().get(it) }
        }

        var maxScore = 0f
        val numBoxes = shape[1]
        for (i in 0 until numBoxes) {
            maxScore = maxOf(maxScore, floatData[i * 6 + 4])
        }
        outputBuffer.rewind()
        return maxScore
    }

    private fun postprocessSingleOutput(buffer: ByteBuffer): List<DetectionResult> {
        val tensor = interpreter.getOutputTensor(0)
        val shape  = tensor.shape()

        val floatData: FloatArray = if (outputDType == DataType.INT8 || outputDType == DataType.UINT8) {
            val qp    = tensor.quantizationParams()
            val qScale = qp.scale
            val qZero  = qp.zeroPoint
            val raw   = ByteArray(buffer.remaining())
            buffer.get(raw)
            FloatArray(raw.size) { i ->
                val v = if (outputDType == DataType.UINT8) (raw[i].toInt() and 0xFF) else raw[i].toInt()
                (v - qZero) * qScale
            }
        } else {
            FloatArray(buffer.remaining() / 4).also { buffer.asFloatBuffer().get(it) }
        }

        return when {
            shape.size == 3 && shape[2] == 6  -> parseEnd2End(floatData, shape[1])
            shape.size == 3 && shape[1] > shape[2] -> parseTransposed(floatData, shape)
            else -> parseStandard(floatData, shape)
        }
    }

    /**
     * End-to-end output: [1, numBoxes, 6]
     * Format: [x1, y1, x2, y2, score, classIdx]
     * Koordinat: NORMALIZED [0, 1] → konversi ke pixel
     */
    private fun parseEnd2End(data: FloatArray, numBoxes: Int): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()

        for (i in 0 until numBoxes) {
            val offset = i * 6
            val score  = data[offset + 4]
            if (score > 0.05f) {
                Log.d(TAG, "RAW[$i]: x1=${data[offset+0]} y1=${data[offset+1]} " +
                        "x2=${data[offset+2]} y2=${data[offset+3]} " +
                        "score=$score class=${data[offset+5].toInt()}")
            }

            if (score < CONF_THRESHOLD) continue

            val classIdx = kotlin.math.round(data[offset + 5]).toInt()
            if (classIdx !in labels.indices) continue

            // Koordinat normalized 0-1 → pixel 640x640
            val x1 = (data[offset + 0] * inputWidth).coerceIn(0f, inputWidth.toFloat())
            val y1 = (data[offset + 1] * inputHeight).coerceIn(0f, inputHeight.toFloat())
            val x2 = (data[offset + 2] * inputWidth).coerceIn(0f, inputWidth.toFloat())
            val y2 = (data[offset + 3] * inputHeight).coerceIn(0f, inputHeight.toFloat())

            if (x2 <= x1 || y2 <= y1) continue

            results.add(DetectionResult(classIdx, score, floatArrayOf(x1, y1, x2, y2), labels[classIdx]))
        }
        return results.sortedByDescending { it.score }
    }

    private fun parseStandard(data: FloatArray, shape: IntArray): List<DetectionResult> {
        val numBoxes   = shape[2]
        val numClasses = shape[1] - 4
        val candidates = mutableListOf<DetectionResult>()

        for (b in 0 until numBoxes) {
            var maxScore = 0f; var classIdx = -1
            for (c in 0 until numClasses) {
                val s = data[(4 + c) * numBoxes + b]
                if (s > maxScore) { maxScore = s; classIdx = c }
            }
            if (maxScore < CONF_THRESHOLD || classIdx !in labels.indices) continue
            val cx = data[b]; val cy = data[numBoxes + b]
            val w  = data[2 * numBoxes + b]; val h = data[3 * numBoxes + b]
            candidates.add(DetectionResult(classIdx, maxScore, floatArrayOf(
                (cx - w / 2) * inputWidth, (cy - h / 2) * inputHeight,
                (cx + w / 2) * inputWidth, (cy + h / 2) * inputHeight
            ), labels[classIdx]))
        }
        return applyNMS(candidates)
    }

    private fun parseTransposed(data: FloatArray, shape: IntArray): List<DetectionResult> {
        val numBoxes   = shape[1]
        val numElems   = shape[2]
        val numClasses = numElems - 4
        val candidates = mutableListOf<DetectionResult>()

        for (i in 0 until numBoxes) {
            val off = i * numElems
            var maxScore = 0f; var classIdx = -1
            for (c in 0 until numClasses) {
                val s = data[off + 4 + c]
                if (s > maxScore) { maxScore = s; classIdx = c }
            }
            if (maxScore < CONF_THRESHOLD || classIdx !in labels.indices) continue
            val cx = data[off]; val cy = data[off + 1]
            val w  = data[off + 2]; val h = data[off + 3]
            candidates.add(DetectionResult(classIdx, maxScore, floatArrayOf(
                (cx - w / 2) * inputWidth, (cy - h / 2) * inputHeight,
                (cx + w / 2) * inputWidth, (cy + h / 2) * inputHeight
            ), labels[classIdx]))
        }
        return applyNMS(candidates)
    }

    private fun applyNMS(detections: List<DetectionResult>, iouThresh: Float = 0.45f): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }
        val suppressed = BooleanArray(sorted.size)
        val keep = mutableListOf<DetectionResult>()
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            keep.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (!suppressed[j] && sorted[i].classIndex == sorted[j].classIndex) {
                    if (iou(sorted[i].boundingBox, sorted[j].boundingBox) > iouThresh)
                        suppressed[j] = true
                }
            }
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix1 = maxOf(a[0], b[0]); val iy1 = maxOf(a[1], b[1])
        val ix2 = minOf(a[2], b[2]); val iy2 = minOf(a[3], b[3])
        if (ix2 <= ix1 || iy2 <= iy1) return 0f
        val inter = (ix2 - ix1) * (iy2 - iy1)
        val union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return if (union > 0f) inter / union else 0f
    }
}