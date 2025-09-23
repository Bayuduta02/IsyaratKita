package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.isyaratkita.ml.Model

class YoloModelBinding(private val context: Context) {

    private val tag = "YoloModelBinding"
    private var model: Model? = try {
        Model.newInstance(context)
    } catch (e: Exception) {
        Log.e(tag, "Failed to load TFLite model", e)
        null
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

    data class Detection(val boundingBox: RectF, val label: String, val confidence: Float)

    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        if (model == null) return Pair(emptyList(), 0L)

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
            return Pair(emptyList(), 0L)
        }
    }

    fun close() {
        model?.close()
        model = null
    }
}