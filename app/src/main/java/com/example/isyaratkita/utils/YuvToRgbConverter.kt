package com.example.isyaratkita.utils

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import java.nio.ByteBuffer

/**
 * YuvToRgbConverter tanpa RenderScript — kompatibel Android 14+
 * RenderScript sudah deprecated di Android 12 dan dihapus di Android 14.
 *
 * Menggunakan konversi YUV_420_888 → RGB manual via ByteBuffer.
 */
class YuvToRgbConverter {

    // Tidak perlu Context lagi — constructor kosong
    constructor()

    // Overload dengan Context agar tidak perlu ubah KameraGestureActivity
    constructor(@Suppress("UNUSED_PARAMETER") context: android.content.Context)

    private var yuvBytes: ByteArray? = null
    private var argbArray: IntArray? = null

    @Synchronized
    fun yuvToRgb(image: Image, output: Bitmap) {
        require(image.format == ImageFormat.YUV_420_888) {
            "Format gambar harus YUV_420_888, dapat: ${image.format}"
        }

        val width = image.width
        val height = image.height
        val pixelCount = width * height

        // Reuse array — tidak alokasi tiap frame
        if (argbArray == null || argbArray!!.size != pixelCount) {
            argbArray = IntArray(pixelCount)
        }

        val planes = image.planes

        // Plane 0: Y (luma)
        val yBuffer: ByteBuffer = planes[0].buffer
        val yRowStride = planes[0].rowStride

        // Plane 1: U (Cb)
        val uBuffer: ByteBuffer = planes[1].buffer
        val uRowStride = planes[1].rowStride
        val uPixelStride = planes[1].pixelStride

        // Plane 2: V (Cr)
        val vBuffer: ByteBuffer = planes[2].buffer
        val vRowStride = planes[2].rowStride
        val vPixelStride = planes[2].pixelStride

        val argb = argbArray!!

        for (row in 0 until height) {
            for (col in 0 until width) {
                val yIndex = row * yRowStride + col

                // UV disampling 2x2
                val uvRow = row / 2
                val uvCol = col / 2
                val uIndex = uvRow * uRowStride + uvCol * uPixelStride
                val vIndex = uvRow * vRowStride + uvCol * vPixelStride

                val y = (yBuffer.get(yIndex).toInt() and 0xFF)
                val u = (uBuffer.get(uIndex).toInt() and 0xFF) - 128
                val v = (vBuffer.get(vIndex).toInt() and 0xFF) - 128

                // Konversi YUV → RGB (BT.601)
                val r = clamp(y + (1.370705f * v).toInt())
                val g = clamp(y - (0.337633f * u).toInt() - (0.698001f * v).toInt())
                val b = clamp(y + (1.732446f * u).toInt())

                argb[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        output.setPixels(argb, 0, width, 0, 0, width, height)
    }

    private fun clamp(value: Int): Int = when {
        value < 0   -> 0
        value > 255 -> 255
        else        -> value
    }
}