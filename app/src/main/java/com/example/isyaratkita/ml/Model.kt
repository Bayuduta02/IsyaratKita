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

/**
 * Model wrapper untuk model.tflite
 * Parameter diambil dari metadata.yaml yang tersedia
 */
class Model private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {
    
    companion object {
        private const val TAG = "Model"
        private const val MODEL_NAME = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        
        // Parameter dari metadata.yaml
        private const val INPUT_SIZE = 640
        private const val NUM_CHANNELS = 3
        private const val NUM_CLASSES = 26
        private const val CONF_THRESHOLD = 0.4f
        
        // Stride disimpan untuk dokumentasi dan referensi meskipun tidak digunakan secara langsung
        // Nilai ini penting dalam konteks YOLO untuk memahami downsampling faktor
        @Suppress("unused")
        private const val STRIDE = 32
        
        /**
         * Membaca label dari file labels.txt di assets
         */
        private fun loadLabelsFromAsset(context: Context): List<String> {
            return try {
                context.assets.open(LABELS_FILE).bufferedReader().useLines { lines ->
                    lines.filter { it.isNotBlank() }.toList()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading labels from assets: ${e.message}")
                // Fallback ke label default jika terjadi error
                listOf("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                       "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
            }
        }
        
        /**
         * Factory method untuk membuat instance Model
         */
        fun newInstance(context: Context): Model {
            // Buat options untuk interpreter
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                
                // Coba gunakan GPU jika tersedia
                try {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU Delegate digunakan")
                    } else {
                        Log.d(TAG, "GPU Delegate tidak didukung, menggunakan CPU")
                    }
                } catch (e: Exception) {
                    // Fallback ke CPU jika GPU tidak tersedia
                    Log.e(TAG, "Error saat setup GPU: ${e.message}")
                }
            }
            
            try {
                // Load model dari assets
                val modelBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)
                val interpreter = Interpreter(modelBuffer, options)
                
                // Load labels dari assets
                val labels = loadLabelsFromAsset(context)
                Log.d(TAG, "Loaded ${labels.size} labels from assets")
                
                Log.d(TAG, "Model berhasil dimuat dari assets")
                return Model(interpreter, labels)
            } catch (e: Exception) {
                Log.e(TAG, "Error saat memuat model: ${e.message}")
                throw RuntimeException("Error saat memuat model: ${e.message}")
            }
        }
    }
    
    /**
     * Kelas untuk menyimpan hasil deteksi
     */
    data class DetectionResult(
        val classIndex: Int,
        val score: Float,
        val boundingBox: FloatArray,  // [x1, y1, x2, y2] dalam koordinat normalized (0-1)
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
    
    /**
     * Memproses gambar dan mengembalikan hasil deteksi
     */
    fun process(bitmap: Bitmap): Pair<List<DetectionResult>, Long> {
        // Preprocess image
        val inputBuffer = preprocessImage(bitmap)
        
        // Prepare output buffer (sesuai dengan YOLOv8 format output)
        // Output format: [1, 84, 8400] untuk model 640x640 dengan 26 kelas
        // 84 = 4 (bbox) + 80 (classes), tetapi kita hanya menggunakan 4 + 26 = 30
        val outputShape = interpreter.getOutputTensor(0).shape()
        Log.d(TAG, "Output shape: ${outputShape.contentToString()}")
        
        val outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(4) { acc, dim -> acc * dim })
            .order(ByteOrder.nativeOrder())
        
        // Run inference with timing
        val startTime = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuffer, outputBuffer)
        val inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000 // Convert to ms
        Log.d(TAG, "Inference time: $inferenceTime ms")
        
        // Process results
        outputBuffer.rewind()
        val detections = postprocessDetections(outputBuffer, outputShape)
        
        return Pair(detections, inferenceTime)
    }
    
    /**
     * Preprocess image untuk input ke model
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize bitmap ke model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        // Allocate buffer untuk Float32 (4 bytes per float)
        val inputBuffer = ByteBuffer.allocateDirect(4 * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE)
            .order(ByteOrder.nativeOrder())
        
        // Extract RGB values dan normalize ke 0-1
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        inputBuffer.rewind()
        
        // Process RGB channels - YOLOv8 mengharapkan [R,G,B] dengan normalisasi 0-1
        for (pixel in pixels) {
            // Extract dan normalize RGB values (0-255 -> 0-1)
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        
        inputBuffer.rewind()
        return inputBuffer
    }
    
    /**
     * Postprocess output dari model
     */
    private fun postprocessDetections(
        outputBuffer: ByteBuffer,
        outputShape: IntArray
    ): List<DetectionResult> {
        // YOLOv8 output format for detection: [1, 84, 8400]
        // where 84 = 4 (bbox) + 80 (classes), dan 8400 adalah jumlah anchor points
        // Untuk kita, kita hanya perlu 4 + 26 = 30
        
        val numBoxes = outputShape[2] // Typically 8400 for YOLOv8 with 640x640 input
        Log.d(TAG, "Number of boxes: $numBoxes")
        
        // In YOLOv8, first 4 rows are box coordinates, next 80 rows are class probabilities
        // We need to transpose this
        
        // Get box coordinates
        val boxes = Array(numBoxes) { FloatArray(4) }
        for (i in 0 until 4) {
            for (j in 0 until numBoxes) {
                boxes[j][i] = outputBuffer.float
            }
        }
        
        // Get class probabilities (only the NUM_CLASSES we need)
        val scores = Array(numBoxes) { FloatArray(NUM_CLASSES) }
        for (i in 0 until NUM_CLASSES) {
            for (j in 0 until numBoxes) {
                scores[j][i] = outputBuffer.float
            }
        }
        
        // Skip remaining classes we don't need (if there are more)
        val classesToSkip = outputShape[1] - 4 - NUM_CLASSES
        if (classesToSkip > 0) {
            for (i in 0 until classesToSkip) {
                for (j in 0 until numBoxes) {
                    outputBuffer.float // Just read and discard
                }
            }
        }
        
        val detections = mutableListOf<DetectionResult>()
        
        // Process each detection
        for (i in 0 until numBoxes) {
            // Find class with highest confidence
            var maxConfidence = 0f
            var classIndex = -1
            
            for (j in 0 until NUM_CLASSES) {
                if (scores[i][j] > maxConfidence) {
                    maxConfidence = scores[i][j]
                    classIndex = j
                }
            }
            
            // Skip if confidence is below threshold
            if (maxConfidence < CONF_THRESHOLD) continue
            
            // YOLOv8 outputs bounding boxes in format [x_center, y_center, width, height]
            val x = boxes[i][0]
            val y = boxes[i][1]
            val w = boxes[i][2]
            val h = boxes[i][3]
            
            // Convert from center format to corner format (normalized coordinates 0-1)
            val x1 = x - w / 2
            val y1 = y - h / 2
            val x2 = x + w / 2
            val y2 = y + h / 2
            
            // Get label for class index
            val label = if (classIndex >= 0 && classIndex < labels.size) {
                labels[classIndex]
            } else {
                "unknown"
            }
            
            detections.add(
                DetectionResult(
                    classIndex,
                    maxConfidence,
                    floatArrayOf(x1, y1, x2, y2),
                    label
                )
            )
        }
        
        Log.d(TAG, "Detected ${detections.size} objects")
        return detections
    }
    
    /**
     * Close interpreter dan release resources
     */
    fun close() {
        interpreter.close()
    }
} 