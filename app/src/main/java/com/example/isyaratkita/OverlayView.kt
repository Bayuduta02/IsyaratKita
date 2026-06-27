package com.example.isyaratkita

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<YoloModelBinding.Detection> = emptyList()
    private var viewWidth = 0
    private var viewHeight = 0
    @Suppress("unused")
    private val tag = "OverlayView"

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f // Sedikit lebih tebal
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f // Sedikit lebih besar
        textAlign = Paint.Align.LEFT
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
        alpha = 180 // Sedikit lebih transparan
    }

    fun setResults(results: List<YoloModelBinding.Detection>) {
        detections = results
        invalidate() // Meminta view untuk digambar ulang
    }

    // Fungsi setPreviewSize tidak lagi begitu relevan karena kita menggunakan ukuran view
    // tapi tidak apa-apa untuk tetap ada.
    fun setPreviewSize(width: Int, height: Int) {}


    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        viewWidth = w
        viewHeight = h
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detections.isEmpty() || viewWidth == 0 || viewHeight == 0) {
            return
        }

        // --- PERBAIKAN ---
        // Logika penskalaan yang benar.
        // Hasil deteksi (detections) memiliki koordinat dalam sistem 640x640 (ukuran input model).
        // Kita perlu mengubahnya ke sistem koordinat View ini.
        val scaleX = viewWidth.toFloat() / 640f
        val scaleY = viewHeight.toFloat() / 640f

        for (detection in detections) {
            // Mengalikan koordinat asli dengan faktor skala
            val scaledBox = RectF(
                detection.boundingBox.left * scaleX,
                detection.boundingBox.top * scaleY,
                detection.boundingBox.right * scaleX,
                detection.boundingBox.bottom * scaleY
            )

            // Menggambar bounding box
            val boxColor = when {
                detection.confidence > 0.8f -> Color.GREEN
                detection.confidence > 0.6f -> Color.YELLOW
                else -> Color.RED
            }
            boxPaint.color = boxColor
            canvas.drawRoundRect(scaledBox, 16f, 16f, boxPaint) // Menggunakan drawRoundRect agar sudut lebih halus

            // Menggambar label dan confidence
            val label = "${detection.label.uppercase()} ${String.format("%.0f", detection.confidence * 100)}%"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)

            val textX = scaledBox.left + 10f
            val textY = scaledBox.top + textBounds.height() + 10f

            // Menggambar background untuk teks agar mudah dibaca
            val backgroundRect = RectF(
                scaledBox.left,
                scaledBox.top,
                scaledBox.left + textBounds.width() + 20f,
                scaledBox.top + textBounds.height() + 20f
            )
            textBackgroundPaint.color = Color.argb(180, 0, 0, 0)
            canvas.drawRoundRect(backgroundRect, 16f, 16f, textBackgroundPaint)

            // Menggambar teks
            canvas.drawText(label, textX, textY, textPaint)
        }
    }
}