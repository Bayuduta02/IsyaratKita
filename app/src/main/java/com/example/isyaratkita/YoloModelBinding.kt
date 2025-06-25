package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 model binding class for sign language detection
 * Based on metadata:
 * - Model version: YOLOv8 v8.3.159
 * - Task: detect
 * - Input size: 640x640
 * - Classes: 26 (a-z)
 * - Channels: 3 (RGB)
 * - Stride: 32
 * - NMS: false (implemented in app)
 * - Int8: false (using float32)
 */
class YoloModelBinding(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // YOLOv8 configuration from metadata.yaml
    private val inputSize = 640  // From metadata: imgsz: [640, 640]
    private val confThreshold = 0.4f // Confidence threshold sesuai permintaan
    private val iouThreshold = 0.45f
    private val maxDetections = 10
    private val numChannels = 3  // From metadata: channels: 3
    private val stride = 32      // From metadata: stride: 32
    
    // Performance tracking
    private var inferenceTime = 0L
    
    // Class labels from labels.txt file or metadata
    private val labels: List<String> by lazy {
        loadLabelsFromAssets()
    }
    
    init {
        try {
            setupInterpreter()
        } catch (e: Exception) {
            Log.e("YoloModelBinding", "Error initializing model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun loadLabelsFromAssets(): List<String> {
        return try {
            // Baca file labels.txt dari assets
            context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                lines.filter { it.isNotBlank() }.toList()
            }
        } catch (e: Exception) {
            Log.e("YoloModelBinding", "Error loading labels: ${e.message}")
            e.printStackTrace()
            
            // Fallback ke label dari metadata.yaml jika terjadi error
            listOf("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
                   "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
        }
    }
    
    private fun setupInterpreter() {
        // Create options with GPU acceleration for better performance
        val options = Interpreter.Options().apply {
            setNumThreads(4) // Use 4 threads for CPU fallback
            
            // Try to use GPU acceleration
            try {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    // Buat GpuDelegate dengan opsi default
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.d("YoloModelBinding", "GPU delegate added successfully")
                }
            } catch (e: Exception) {
                Log.w("YoloModelBinding", "GPU acceleration not available: ${e.message}")
            }
        }
        
        // Load model from assets
        try {
            val model = FileUtil.loadMappedFile(context, "model.tflite")
            interpreter = Interpreter(model, options)
            
            // Log input/output tensor info
            val inputTensor = interpreter?.getInputTensor(0)
            val outputTensor = interpreter?.getOutputTensor(0)
            
            Log.d("YoloModelBinding", "Input tensor shape: ${inputTensor?.shape()?.contentToString()}")
            Log.d("YoloModelBinding", "Output tensor shape: ${outputTensor?.shape()?.contentToString()}")
        } catch (e: Exception) {
            Log.e("YoloModelBinding", "Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float
    )
    
    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        if (interpreter == null) {
            return Pair(emptyList(), 0L)
        }
        
        try {
            // Preprocess: resize and normalize the image
            val inputBuffer = preprocessImage(bitmap)
            
            // Prepare output buffer based on model output shape
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            val outputBuffer = ByteBuffer.allocateDirect(outputShape.fold(4) { acc, dim -> acc * dim })
                .order(ByteOrder.nativeOrder())
            
            // Run inference with timing
            val startTime = SystemClock.elapsedRealtimeNanos()
            interpreter!!.run(inputBuffer, outputBuffer)
            inferenceTime = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000 // Convert to ms
            
            // Process results
            outputBuffer.rewind()
            val detections = postprocessDetections(outputBuffer, outputShape, bitmap.width, bitmap.height)
            
            return Pair(detections, inferenceTime)
        } catch (e: Exception) {
            Log.e("YoloModelBinding", "Detection error: ${e.message}")
            e.printStackTrace()
            return Pair(emptyList(), 0L)
        }
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize bitmap to model input size from metadata (640x640)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // Allocate buffer for input (4 bytes per float * numChannels * width * height)
        val inputBuffer = ByteBuffer.allocateDirect(4 * numChannels * inputSize * inputSize)
            .order(ByteOrder.nativeOrder())
        
        // Extract RGB values and normalize to 0-1
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        // Optimize by processing all channels in one pass
        inputBuffer.rewind()
        
        for (pixel in pixels) {
            // Extract and normalize RGB values (0-255 -> 0-1)
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        
        inputBuffer.rewind()
        return inputBuffer
    }
    
    private fun postprocessDetections(
        outputBuffer: ByteBuffer,
        outputShape: IntArray,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        // YOLOv8 output format for detection based on metadata:
        // For our model with 26 classes (a-z): [1, 30, 8400] where 30 = 4 (bbox) + 26 (classes)
        // and 8400 is the number of anchor points
        
        // Jumlah kelas dalam model dari metadata (a-z)
        val numClasses = labels.size
        val numBoxes = outputShape[2] // Typically 8400 for YOLOv8 with 640x640 input
        val rowSize = 4 + numClasses // 4 box coordinates + class probabilities
        
        Log.d("YoloModelBinding", "Output shape: ${outputShape.contentToString()}")
        
        // For YOLOv8, we need to transpose the output
        val output = Array(numBoxes) { FloatArray(rowSize) }
        
        // Reset buffer position
        outputBuffer.rewind()
        
        try {
            // Read data from buffer
            for (i in 0 until rowSize) {
                for (j in 0 until numBoxes) {
                    if (outputBuffer.hasRemaining()) {
                        output[j][i] = outputBuffer.float
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("YoloModelBinding", "Error parsing output buffer: ${e.message}")
            e.printStackTrace()
            return emptyList()
        }
        
        val detections = mutableListOf<Detection>()
        
        // Process each detection
        for (i in 0 until numBoxes) {
            // First 4 values are bounding box coordinates (x, y, w, h)
            val x = output[i][0]
            val y = output[i][1]
            val w = output[i][2]
            val h = output[i][3]
            
            // Find class with highest confidence
            var maxConfidence = 0f
            var classIndex = -1
            
            for (j in 4 until 4 + numClasses) {
                if (output[i][j] > maxConfidence) {
                    maxConfidence = output[i][j]
                    classIndex = j - 4
                }
            }
            
            // Skip if confidence is below threshold
            if (maxConfidence < confThreshold || classIndex < 0) continue
            
            // Convert normalized coordinates to actual pixel values
            val xMin = (x - w / 2) * originalWidth
            val yMin = (y - h / 2) * originalHeight
            val xMax = (x + w / 2) * originalWidth
            val yMax = (y + h / 2) * originalHeight
            
            // Ensure coordinates are within image bounds
            val left = max(0f, min(xMin, originalWidth.toFloat()))
            val top = max(0f, min(yMin, originalHeight.toFloat()))
            val right = max(left, min(xMax, originalWidth.toFloat()))
            val bottom = max(top, min(yMax, originalHeight.toFloat()))
            
            // Get class label
            val label = if (classIndex >= 0 && classIndex < labels.size) labels[classIndex] else "unknown"
            
            detections.add(
                Detection(
                    RectF(left, top, right, bottom),
                    label,
                    maxConfidence
                )
            )
        }
        
        // Apply Non-Maximum Suppression
        return applyNMS(detections)
    }
    
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        
        // Sort by confidence (highest first)
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()
        val isSelected = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (isSelected[i]) continue
            
            // Add current detection to selected list
            selectedDetections.add(sortedDetections[i])
            
            // Mark overlapping detections for removal
            for (j in i + 1 until sortedDetections.size) {
                if (isSelected[j]) continue
                
                val iou = calculateIoU(sortedDetections[i].boundingBox, sortedDetections[j].boundingBox)
                if (iou > iouThreshold) {
                    isSelected[j] = true
                }
            }
            
            // Limit number of detections
            if (selectedDetections.size >= maxDetections) break
        }
        
        return selectedDetections
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
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
} 