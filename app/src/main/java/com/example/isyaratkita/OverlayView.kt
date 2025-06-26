package com.example.isyaratkita

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
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
    private var screenWidth = 0
    private var screenHeight = 0
    private val TAG = "OverlayView"

    // Paint objects untuk drawing
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f // Lebih tebal untuk visibilitas yang lebih baik
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
        strokeWidth = 10f // Lebih tebal untuk shadow yang lebih jelas
        isAntiAlias = true
        maskFilter = BlurMaskFilter(8f, BlurMaskFilter.Blur.NORMAL) // Tambahkan blur untuk efek shadow
    }

    fun setPreviewSize(width: Int, height: Int) {
        previewWidth = width
        previewHeight = height
        Log.d(TAG, "Preview size set to: $width x $height")
    }
    
    fun setScreenSize(width: Int, height: Int) {
        screenWidth = width
        screenHeight = height
        Log.d(TAG, "Screen size set to: $width x $height")
    }

    fun setBoxesAndLabels(boxes: List<Pair<RectF, String>>) {
        boxesAndLabels = boxes
        Log.d(TAG, "Set ${boxes.size} boxes")
        for ((box, label) in boxes) {
            Log.d(TAG, "Box: $box, Label: $label")
        }
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
        Log.d(TAG, "View size changed to: $w x $h")
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (boxesAndLabels.isEmpty() || viewWidth == 0 || viewHeight == 0) {
            Log.d(TAG, "Skipping draw: boxes=${boxesAndLabels.size}, viewSize=${viewWidth}x${viewHeight}")
            return
        }

        // Calculate scale factors untuk mapping dari preview ke view coordinates
        // Jika screenWidth/Height tersedia, gunakan itu untuk scaling yang lebih akurat
        val scaleX: Float
        val scaleY: Float
        
        if (screenWidth > 0 && screenHeight > 0) {
            // Scaling berdasarkan ukuran layar, bukan ukuran preview
            scaleX = viewWidth.toFloat() / screenWidth.toFloat()
            scaleY = viewHeight.toFloat() / screenHeight.toFloat()
            Log.d(TAG, "Using screen-based scaling: scaleX=$scaleX, scaleY=$scaleY")
        } else if (previewWidth > 0 && previewHeight > 0) {
            // Fallback ke scaling berdasarkan ukuran preview
            scaleX = viewWidth.toFloat() / previewWidth.toFloat()
            scaleY = viewHeight.toFloat() / previewHeight.toFloat()
            Log.d(TAG, "Using preview-based scaling: scaleX=$scaleX, scaleY=$scaleY")
        } else {
            // Jika tidak ada ukuran referensi, gunakan 1:1
            scaleX = 1.0f
            scaleY = 1.0f
            Log.d(TAG, "No reference size available, using 1:1 scaling")
        }

        for ((box, text) in boxesAndLabels) {
            // Scale bounding box ke view coordinates
            val scaledBox = RectF(
                box.left * scaleX,
                box.top * scaleY,
                box.right * scaleX,
                box.bottom * scaleY
            )
            
            // Pastikan box tidak keluar dari view bounds
            val clampedBox = RectF(
                Math.max(0f, scaledBox.left),
                Math.max(0f, scaledBox.top),
                Math.min(viewWidth.toFloat(), scaledBox.right),
                Math.min(viewHeight.toFloat(), scaledBox.bottom)
            )
            
            Log.d(TAG, "Drawing box: original=$box, scaled=$scaledBox, clamped=$clampedBox")

            // Draw shadow untuk bounding box (untuk visibilitas yang lebih baik)
            canvas.drawRect(clampedBox, shadowPaint)

            // Draw main bounding box dengan warna hijau
            canvas.drawRect(clampedBox, boxPaint)

            // Draw text label
            drawTextWithBackground(canvas, text, clampedBox)
        }
    }

    private fun drawTextWithBackground(canvas: Canvas, text: String, box: RectF) {
        // Measure text
        val textBounds = Rect()
        textPaint.getTextBounds(text, 0, text.length, textBounds)

        val textWidth = textBounds.width()
        val textHeight = textBounds.height()
        val padding = 12f

        // Calculate text position - posisikan di atas kotak jika memungkinkan
        var textX = box.left + padding
        var textY = box.top - padding
        
        // Jika kotak terlalu dekat dengan bagian atas, letakkan teks di dalam kotak
        if (textY < textHeight + padding) {
            textY = box.top + textHeight + padding * 2
        }

        // Batasi posisi teks agar tetap di dalam view
        if (textX + textWidth + padding > viewWidth) {
            textX = viewWidth - textWidth - padding
        }
        if (textX < padding) {
            textX = padding
        }

        // Draw text background dengan sudut rounded
        val backgroundRect = RectF(
            textX - padding,
            textY - textHeight - padding,
            textX + textWidth + padding,
            textY + padding
        )

        canvas.drawRoundRect(backgroundRect, 8f, 8f, textBackgroundPaint)

        // Draw text dengan warna putih
        canvas.drawText(text, textX, textY, textPaint)
    }

    fun setResults(results: List<YoloModelBinding.Detection>) {
        // Konversi hasil deteksi ke format yang kompatibel dengan metode existing
        val boxesAndLabels = results.map { detection ->
            detection.boundingBox to "${detection.label} (${String.format("%.0f", detection.confidence * 100)}%)"
        }
        
        setBoxesAndLabels(boxesAndLabels)
    }
}