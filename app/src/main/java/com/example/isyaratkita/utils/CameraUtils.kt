package com.example.isyaratkita.utils

import android.graphics.Point
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Size

/**
 * Extension functions dan utility untuk Camera2 API
 */

/**
 * Get optimal preview size berdasarkan display size dan camera capabilities
 */
fun getPreviewOutputSize(
    display: android.view.Display,
    characteristics: CameraCharacteristics,
    surfaceClass: Class<*>,
    targetClass: Class<*>? = null
): Size {
    val screenSize = Point()
    display.getSize(screenSize)
    val hdScreen = screenSize.x >= 720 || screenSize.y >= 720
    val maxSize = if (hdScreen) Size(1280, 720) else Size(640, 480)

    val config = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
    val allSizes = config.getOutputSizes(surfaceClass)

    return allSizes
        .sortedWith(compareBy { it.height * it.width })
        .first { it.height * it.width <= maxSize.height * maxSize.width }
}

/**
 * Get camera orientation untuk menentukan rotasi yang diperlukan
 */
fun getCameraOrientation(characteristics: CameraCharacteristics): Int {
    return characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
}

/**
 * Check apakah camera mendukung autofocus
 */
fun supportsAutofocus(characteristics: CameraCharacteristics): Boolean {
    val afModes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
    return afModes?.contains(CameraCharacteristics.CONTROL_AF_MODE_AUTO) == true ||
            afModes?.contains(CameraCharacteristics.CONTROL_AF_MODE_CONTINUOUS_PICTURE) == true
}

/**
 * Get semua camera IDs yang tersedia
 */
fun CameraManager.getAvailableCameraIds(): Array<String> {
    return try {
        cameraIdList
    } catch (e: Exception) {
        emptyArray()
    }
}

/**
 * Check apakah camera adalah front camera
 */
fun CameraManager.isFrontCamera(cameraId: String): Boolean {
    return try {
        val characteristics = getCameraCharacteristics(cameraId)
        val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
        facing == CameraCharacteristics.LENS_FACING_FRONT
    } catch (e: Exception) {
        false
    }
}

/**
 * Get camera ID untuk back atau front camera
 */
fun CameraManager.getCameraId(useFrontCamera: Boolean): String? {
    return try {
        cameraIdList.firstOrNull { id ->
            val characteristics = getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (useFrontCamera) {
                facing == CameraCharacteristics.LENS_FACING_FRONT
            } else {
                facing == CameraCharacteristics.LENS_FACING_BACK
            }
        }
    } catch (e: Exception) {
        null
    }
}

/**
 * Compute transform matrix untuk preview
 */
fun computeTransformationMatrix(
    viewWidth: Int,
    viewHeight: Int,
    imageWidth: Int,
    imageHeight: Int,
    applyRotation: Int,
    maintainAspectRatio: Boolean
): android.graphics.Matrix {
    val matrix = android.graphics.Matrix()

    if (applyRotation != 0) {
        // Rotate around center
        matrix.postRotate(applyRotation.toFloat(), viewWidth / 2f, viewHeight / 2f)
    }

    if (maintainAspectRatio) {
        val viewRatio = viewWidth.toFloat() / viewHeight
        val imageRatio = imageWidth.toFloat() / imageHeight
        val scaleX: Float
        val scaleY: Float

        if (viewRatio > imageRatio) {
            // View is wider, scale to fit height
            scaleY = viewHeight.toFloat() / imageHeight
            scaleX = scaleY * imageRatio / viewRatio
        } else {
            // View is taller, scale to fit width
            scaleX = viewWidth.toFloat() / imageWidth
            scaleY = scaleX * viewRatio / imageRatio
        }

        matrix.preScale(scaleX, scaleY, viewWidth / 2f, viewHeight / 2f)
    }

    return matrix
}