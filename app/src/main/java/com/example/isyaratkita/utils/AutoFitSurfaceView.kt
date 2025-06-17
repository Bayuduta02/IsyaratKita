package com.example.isyaratkita.utils

import android.content.Context
import android.util.AttributeSet
import android.view.SurfaceView
import kotlin.math.roundToInt

/**
 * SurfaceView yang otomatis menyesuaikan aspect ratio
 * berdasarkan ukuran preview kamera
 */
class AutoFitSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : SurfaceView(context, attrs, defStyleAttr) {

    private var aspectRatio = 0f

    /**
     * Set aspect ratio untuk surface view
     * @param width lebar preview
     * @param height tinggi preview
     */
    fun setAspectRatio(width: Int, height: Int) {
        require(width > 0 && height > 0) { "Size cannot be negative." }
        aspectRatio = width.toFloat() / height.toFloat()
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)

        if (aspectRatio == 0f) return

        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = MeasureSpec.getSize(heightMeasureSpec)

        val newWidth: Int
        val newHeight: Int

        val actualRatio = if (width > height) aspectRatio else 1f / aspectRatio

        if (width < height * actualRatio) {
            newWidth = width
            newHeight = (width / actualRatio).roundToInt()
        } else {
            newWidth = (height * actualRatio).roundToInt()
            newHeight = height
        }

        setMeasuredDimension(newWidth, newHeight)
    }
}