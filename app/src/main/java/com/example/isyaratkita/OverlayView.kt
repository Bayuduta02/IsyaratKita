package com.example.isyaratkita

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var boxesAndLabels: List<Pair<RectF, String>> = emptyList()
    private var viewWidth = 0
    private var viewHeight = 0
    private var previewWidth = 0
    private var previewHeight = 0

    // Paint objects untuk drawing
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        textAlign = Paint.Align.LEFT
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
        alpha = 180
    }

    private val shadowPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    fun setPreviewSize(width: Int, height: Int) {
        previewWidth = width
        previewHeight = height
    }

    fun setBoxesAndLabels(boxes: List<Pair<RectF, String>>) {
        boxesAndLabels = boxes
        invalidate() // Trigger redraw
    }

    fun clearBoxes() {
        boxesAndLabels = emptyList()
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        viewWidth = w
        viewHeight = h
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (boxesAndLabels.isEmpty() || viewWidth == 0 || viewHeight == 0) {
            return
        }

        // Calculate scale factors untuk mapping dari preview ke view coordinates
        val scaleX = viewWidth.toFloat() / previewWidth.toFloat()
        val scaleY = viewHeight.toFloat() / previewHeight.toFloat()

        for ((box, text) in boxesAndLabels) {
            // Scale bounding box ke view coordinates
            val scaledBox = RectF(
                box.left * scaleX,
                box.top * scaleY,
                box.right * scaleX,
                box.bottom * scaleY
            )

            // Draw shadow untuk bounding box
            canvas.drawRect(scaledBox, shadowPaint)

            // Draw main bounding box
            canvas.drawRect(scaledBox, boxPaint)

            // Draw text label
            drawTextWithBackground(canvas, text, scaledBox)
        }
    }

    private fun drawTextWithBackground(canvas: Canvas, text: String, box: RectF) {
        // Measure text
        val textBounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, textBounds)

        val textWidth = textBounds.width()
        val textHeight = textBounds.height()
        val padding = 12f

        // Calculate text position
        var textX = box.left + padding
        var textY = box.top - padding

        // Adjust if text goes outside view bounds
        if (textY < textHeight + padding) {
            textY = box.top + textHeight + padding
        }
        if (textX + textWidth + padding > viewWidth) {
            textX = viewWidth - textWidth - padding
        }
        if (textX < padding) {
            textX = padding
        }

        // Draw text background
        val backgroundRect = RectF(
            textX - padding,
            textY - textHeight - padding,
            textX + textWidth + padding,
            textY + padding
        )

        canvas.drawRoundRect(backgroundRect, 8f, 8f, textBackgroundPaint)

        // Draw text
        canvas.drawText(text, textX, textY, textPaint)
    }
}