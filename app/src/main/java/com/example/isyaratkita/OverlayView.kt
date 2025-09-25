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
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<YoloModelBinding.Detection> = emptyList()
    private var viewWidth = 0
    private var viewHeight = 0
    private var previewWidth = 0
    private var previewHeight = 0
    private var isFrontFacing = false
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
        invalidate()
    }

    fun setSourceInfo(width: Int, height: Int, frontFacing: Boolean) {
        if (width <= 0 || height <= 0) return
        val sourceChanged = previewWidth != width || previewHeight != height
        val facingChanged = isFrontFacing != frontFacing

        if (sourceChanged || facingChanged) {
            previewWidth = width
            previewHeight = height
            isFrontFacing = frontFacing
            invalidate()
        }
    }


    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        viewWidth = w
        viewHeight = h
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detections.isEmpty() || viewWidth == 0 || viewHeight == 0 || previewWidth == 0 || previewHeight == 0) {
            return
        }

        val scale = max(
            viewWidth.toFloat() / previewWidth.toFloat(),
            viewHeight.toFloat() / previewHeight.toFloat()
        )
        val scaledWidth = previewWidth * scale
        val scaledHeight = previewHeight * scale
        val dx = (viewWidth - scaledWidth) / 2f
        val dy = (viewHeight - scaledHeight) / 2f

        for (detection in detections) {
            var left = detection.boundingBox.left * scale + dx
            val top = detection.boundingBox.top * scale + dy
            var right = detection.boundingBox.right * scale + dx
            val bottom = detection.boundingBox.bottom * scale + dy


            if (isFrontFacing) {
                val mirroredLeft = viewWidth - right
                val mirroredRight = viewWidth - left
                left = mirroredLeft
                right = mirroredRight
            }

            if (left > right) {
                val temp = left
                left = right
                right = temp
            }

            val clippedRect = RectF(
                left.coerceIn(0f, viewWidth.toFloat()),
                top.coerceIn(0f, viewHeight.toFloat()),
                right.coerceIn(0f, viewWidth.toFloat()),
                bottom.coerceIn(0f, viewHeight.toFloat())
            )

            if (clippedRect.width() <= 0f || clippedRect.height() <= 0f) {
                continue
            }
            val boxColor = when {
                detection.confidence > 0.8f -> Color.GREEN
                detection.confidence > 0.6f -> Color.YELLOW
                else -> Color.RED
            }
            boxPaint.color = boxColor
            canvas.drawRoundRect(clippedRect, 16f, 16f, boxPaint)

            // Menggambar label dan confidence
            val label = "${detection.label.uppercase()} ${String.format("%.0f", detection.confidence * 100)}%"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)

            val backgroundRect = RectF(
                clippedRect.left,
                clippedRect.top,
                clippedRect.left + textBounds.width() + 20f,
                clippedRect.top + textBounds.height() + 20f,
            )
            textBackgroundPaint.color = Color.argb(180, 0, 0, 0)
            canvas.drawRoundRect(backgroundRect, 16f, 16f, textBackgroundPaint)

            // Menggambar teks
            val textX = clippedRect.left + 10f
            val textY = clippedRect.top + textBounds.height() + 10f
            canvas.drawText(label, textX, textY, textPaint)
        }
    }
}