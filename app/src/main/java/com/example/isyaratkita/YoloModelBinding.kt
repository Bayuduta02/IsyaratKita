package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.isyaratkita.ml.Model
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 model binding class for sign language detection
 * Menggunakan parameter dari metadata.yaml:
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
    private var model: Model? = null
    private val tag = "YoloModelBinding"
    
    // YOLOv8 configuration dari metadata.yaml
    private val iouThreshold = 0.45f
    private val maxDetections = 10
    
    // Performance tracking
    private var inferenceTime = 0L
    
    init {
        try {
            setupModel()
        } catch (e: Exception) {
            Log.e(tag, "Error initializing model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun setupModel() {
        try {
            model = Model.newInstance(context)
            Log.d(tag, "Model initialized successfully")
        } catch (e: Exception) {
            Log.e(tag, "Error initializing model: ${e.message}")
            throw RuntimeException("Failed to initialize model: ${e.message}")
        }
    }
    
    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float
    )
    
    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        if (model == null) {
            Log.e(tag, "Model is null, cannot perform detection")
            return Pair(emptyList(), 0L)
        }
        
        try {
            // Process bitmap dengan model
            val (detectionResults, inferenceTime) = model!!.process(bitmap)
            this.inferenceTime = inferenceTime
            
            // Convert model output ke Detection objects
            val detections = detectionResults.map { result ->
                // Convert normalized coordinates to actual pixel values
                val left = result.boundingBox[0] * bitmap.width
                val top = result.boundingBox[1] * bitmap.height
                val right = result.boundingBox[2] * bitmap.width
                val bottom = result.boundingBox[3] * bitmap.height
                
                // Ensure coordinates are within image bounds
                val clampedLeft = max(0f, min(left, bitmap.width.toFloat()))
                val clampedTop = max(0f, min(top, bitmap.height.toFloat()))
                val clampedRight = max(clampedLeft, min(right, bitmap.width.toFloat()))
                val clampedBottom = max(clampedTop, min(bottom, bitmap.height.toFloat()))
                
                Detection(
                    RectF(clampedLeft, clampedTop, clampedRight, clampedBottom),
                    result.label,
                    result.score
                )
            }
            
            // Apply Non-Maximum Suppression
            val nmsDetections = applyNMS(detections)
            
            Log.d(tag, "Detected ${nmsDetections.size} objects after NMS, inference time: $inferenceTime ms")
            return Pair(nmsDetections, inferenceTime)
        } catch (e: Exception) {
            Log.e(tag, "Detection error: ${e.message}")
            e.printStackTrace()
            return Pair(emptyList(), 0L)
        }
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
        model?.close()
        model = null
        Log.d(tag, "Model resources released")
    }
} 