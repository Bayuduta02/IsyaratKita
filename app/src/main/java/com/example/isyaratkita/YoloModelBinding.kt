package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.isyaratkita.ml.Model

class YoloModelBinding(private val context: Context) {

    private val tag = "YoloModelBinding"
    private var model: Model? = null
    private var initError: String? = null

    init {
        try {
            Log.d(tag, "Attempting to load TFLite model...")
            model = Model.newInstance(context)
            Log.d(tag, "TFLite model loaded successfully")
            Log.d(tag, "Model input size: ${inputWidth}x${inputHeight}")
            Log.d(tag, "Number of labels: ${labels.size}")
        } catch (e: Exception) {
            val errorMsg = when {
                e.message?.contains("model.tflite") == true ->
                    "Model file not found or corrupted. Please add a valid model.tflite file to assets folder."
                e.message?.contains("labels.txt") == true ->
                    "Labels file not found or corrupted."
                else -> "Failed to initialize model: ${e.message}"
            }
            initError = errorMsg
            Log.e(tag, errorMsg, e)
            e.printStackTrace()
        }
    }
    // --- PERBAIKAN ---
    // Konstanta iouThreshold dan maxDetections tidak lagi diperlukan
    // karena NMS sudah ada di dalam model.
    val inputWidth: Int
        get() = model?.inputWidth ?: 640

    val inputHeight: Int
        get() = model?.inputHeight ?: 640

    val labels: List<String>
        get() = model?.labels ?: emptyList()

    val isModelLoaded: Boolean
        get() = model != null

    val errorMessage: String?
        get() = initError

    data class Detection(val boundingBox: RectF, val label: String, val confidence: Float)

    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        if (model == null) {
            Log.w(tag, "Cannot detect: model not loaded")
            return Pair(emptyList(), 0L)
        }

        try {
            val (detectionResults, inferenceTime) = model!!.process(bitmap)

            val detections = detectionResults.map { result ->
                Detection(
                    RectF(
                        result.boundingBox[0],
                        result.boundingBox[1],
                        result.boundingBox[2],
                        result.boundingBox[3]
                    ),
                    result.label,
                    result.score
                )
            }
            return Pair(detections, inferenceTime)

        } catch (e: Exception) {
            Log.e(tag, "Error during detection", e)
            return Pair(emptyList(), 0L)
        }
    }

    fun close() {
        model?.close()
        model = null
    }
}