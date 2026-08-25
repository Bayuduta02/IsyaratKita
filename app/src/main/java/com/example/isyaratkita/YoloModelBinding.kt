package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.isyaratkita.ml.Model
import kotlin.math.max
import kotlin.math.min

class YoloModelBinding(context: Context) {
    private var model: Model? = null
    private val tag = "YoloModelBinding"
    val initError: String?

    val isReady: Boolean get() = model != null

    init {
        initError = try {
            model = Model.newInstance(context)
            Log.d(tag, "Model initialized successfully")
            null
        } catch (e: Exception) {
            Log.e(tag, "Error initializing model: ${e.message}", e)
            e.message
        }
    }

    data class Detection(val boundingBox: RectF, val label: String, val confidence: Float)

    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        if (model == null) return Pair(emptyList(), 0L)

        try {
            val startTime = System.currentTimeMillis()

            // Model.process() sekarang mengembalikan hasil yang sudah di-filter oleh NMS internal model.
            val (detectionResults, inferenceTime) = model!!.process(bitmap)

            // Kita hanya perlu mengurutkan hasilnya berdasarkan confidence
            val sortedDetections = detectionResults.sortedByDescending { it.score }
            Log.d(tag, "Detections from model (NMS included): ${sortedDetections.size}")

            val detections = sortedDetections.map { result ->
                // Mengambil koordinat dari hasil deteksi
                val left = result.boundingBox[0]
                val top = result.boundingBox[1]
                val right = result.boundingBox[2]
                val bottom = result.boundingBox[3]

                // Memastikan koordinat tidak keluar dari batas gambar
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

            val totalProcessingTime = System.currentTimeMillis() - startTime
            Log.d(tag, "Final processing time: ${totalProcessingTime}ms")

            // --- PERBAIKAN ---
            // Kembalikan 'detections' secara langsung tanpa memanggil NMS manual.
            return Pair(detections, inferenceTime)

        } catch (e: Exception) {
            Log.e(tag, "Detection error: ${e.message}")
            return Pair(emptyList(), 0L)
        }
    }

    fun close() {
        model?.close()
        model = null
    }
}