package com.example.isyaratkita.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.renderscript.*
import java.nio.ByteBuffer

/**
 * Helper class untuk konversi YUV420_888 ke RGB menggunakan RenderScript
 * untuk performa yang optimal
 */
class YuvToRgbConverter(context: Context) {
    private val rs = RenderScript.create(context)
    private val scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    private var pixelCount: Int = -1
    private lateinit var yuvBuffer: ByteBuffer
    private lateinit var inputAllocation: Allocation
    private lateinit var outputAllocation: Allocation

    @Synchronized
    fun yuvToRgb(image: Image, output: Bitmap) {
        // Pastikan format image adalah YUV_420_888
        require(image.format == ImageFormat.YUV_420_888) {
            "Invalid image format: ${image.format}"
        }

        val currentPixelCount = image.width * image.height
        if (currentPixelCount != pixelCount) {
            pixelCount = currentPixelCount

            // Buat buffer untuk YUV data
            yuvBuffer = ByteBuffer.allocateDirect(
                currentPixelCount * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8
            )

            // Buat allocations untuk input dan output
            inputAllocation = Allocation.createSized(rs, Element.U8(rs), yuvBuffer.array().size)
            outputAllocation = Allocation.createFromBitmap(rs, output)
        }

        // Convert Image ke ByteBuffer
        imageToByteBuffer(image, yuvBuffer.array())

        // Copy data ke input allocation
        inputAllocation.copyFrom(yuvBuffer.array())

        // Set input allocation untuk script
        scriptYuvToRgb.setInput(inputAllocation)

        // Run script untuk konversi
        scriptYuvToRgb.forEach(outputAllocation)

        // Copy hasil ke output bitmap
        outputAllocation.copyTo(output)
    }

    private fun imageToByteBuffer(image: Image, outputBuffer: ByteArray) {
        require(image.format == ImageFormat.YUV_420_888) {
            "Invalid image format: ${image.format}"
        }

        val imageCrop = image.cropRect
        val imagePlanes = image.planes

        imagePlanes.forEachIndexed { planeIndex, plane ->
            val outputStride: Int
            var outputOffset: Int

            when (planeIndex) {
                0 -> {
                    outputStride = 1
                    outputOffset = 0
                }
                1 -> {
                    outputStride = 2
                    outputOffset = image.width * image.height + 1
                }
                2 -> {
                    outputStride = 2
                    outputOffset = image.width * image.height
                }
                else -> return@forEachIndexed
            }

            val planeBuffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride

            val planeCrop = if (planeIndex == 0) {
                imageCrop
            } else {
                android.graphics.Rect(
                    imageCrop.left / 2,
                    imageCrop.top / 2,
                    imageCrop.right / 2,
                    imageCrop.bottom / 2
                )
            }

            val planeWidth = planeCrop.width()
            val planeHeight = planeCrop.height()

            val rowBuffer = ByteArray(plane.rowStride)

            val rowLength = if (pixelStride == 1 && outputStride == 1) {
                planeWidth
            } else {
                (planeWidth - 1) * pixelStride + 1
            }

            for (row in 0 until planeHeight) {
                planeBuffer.position(
                    (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride
                )

                if (pixelStride == 1 && outputStride == 1) {
                    planeBuffer.get(outputBuffer, outputOffset, rowLength)
                    outputOffset += rowLength
                } else {
                    planeBuffer.get(rowBuffer, 0, rowLength)
                    for (col in 0 until planeWidth) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                        outputOffset += outputStride
                    }
                }
            }
        }
    }

    fun close() {
        if (::inputAllocation.isInitialized) {
            inputAllocation.destroy()
        }
        if (::outputAllocation.isInitialized) {
            outputAllocation.destroy()
        }
        scriptYuvToRgb.destroy()
        rs.destroy()
    }
}