package com.example.isyaratkita

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.isyaratkita.utils.AutoFitSurfaceView
import com.example.isyaratkita.utils.CameraSizes.getPreviewOutputSize
import com.example.isyaratkita.utils.YuvToRgbConverter
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class KameraGestureActivity : AppCompatActivity() {

    private lateinit var surfaceView: AutoFitSurfaceView
    private lateinit var gestureText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var fpsText: TextView
    private lateinit var cameraDevice: CameraDevice
    private lateinit var cameraCaptureSession: CameraCaptureSession
    private lateinit var previewSize: Size
    private lateinit var previewRequestBuilder: CaptureRequest.Builder
    private lateinit var closeButton: ImageButton
    private lateinit var switchCameraButton: ImageButton
    private lateinit var overlayView: OverlayView
    private lateinit var imageReader: ImageReader
    private lateinit var yuvToRgbConverter: YuvToRgbConverter
    private var modelBinding: YoloModelBinding? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var inferenceThread: HandlerThread? = null
    private var inferenceHandler: Handler? = null

    private val cameraOpenCloseLock = Semaphore(1)
    private val isProcessingFrame = AtomicBoolean(false)

    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f

    private var frameSkipCounter = 0
    private val FRAME_SKIP_RATE = 2 // Proses setiap 3 frame, bisa disesuaikan

    private val cameraManager: CameraManager by lazy {
        getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private var currentCameraId: String = ""
    private var isBackCamera: Boolean = true

    // --- PERBAIKAN ---
    // Menggunakan konstanta dari Model.kt sebagai satu-satunya sumber kebenaran
    // untuk ukuran input model.
    private val modelInputSize = 640

    companion object {
        private const val TAG = "KameraGestureActivity"
        private const val CAMERA_REQUEST_CODE = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_FULLSCREEN or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY

        setContentView(R.layout.camera_activity)
        initViews()
        setupClickListeners()
        setupSurfaceView()
        initializeModel()
    }

    private fun initViews() {
        surfaceView = findViewById(R.id.camera_preview)
        gestureText = findViewById(R.id.gesture_text)
        confidenceText = findViewById(R.id.confidence_text)
        fpsText = findViewById(R.id.fps_text)
        closeButton = findViewById(R.id.btn_close)
        switchCameraButton = findViewById(R.id.btn_switch_camera)
        overlayView = findViewById(R.id.overlay_view)
    }

    private fun setupClickListeners() {
        closeButton.setOnClickListener { finish() }
        switchCameraButton.setOnClickListener {
            isBackCamera = !isBackCamera
            closeCamera()
            checkCameraPermissionAndOpen()
        }
    }

    private fun setupSurfaceView() {
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                checkCameraPermissionAndOpen()
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                overlayView.setPreviewSize(width, height)
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                closeCamera()
            }
        })
    }

    private fun initializeModel() {
        try {
            // Periksa keberadaan file model dan label terlebih dahulu
            val assetManager = assets
            val modelExists = assetManager.list("")?.contains("model.tflite") ?: false
            val labelsExists = assetManager.list("")?.contains("labels.txt") ?: false

            if (!modelExists || !labelsExists) {
                val errorMsg = when {
                    !modelExists && !labelsExists -> "File model.tflite dan labels.txt tidak ditemukan"
                    !modelExists -> "File model.tflite tidak ditemukan"
                    else -> "File labels.txt tidak ditemukan"
                }
                throw Exception(errorMsg)
            }

            yuvToRgbConverter = YuvToRgbConverter(this)
            modelBinding = YoloModelBinding(this)
            gestureText.text = "Model siap"
        } catch (e: Exception) {
            gestureText.text = "Error: ${e.message}"
            Log.e(TAG, "Error initializing model: ${e.message}", e)
            Toast.makeText(this, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        try {
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }

            currentCameraId = cameraManager.cameraIdList.first { id ->
                val characteristics = cameraManager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (isBackCamera) facing == CameraCharacteristics.LENS_FACING_BACK
                else facing == CameraCharacteristics.LENS_FACING_FRONT
            }

            val characteristics = cameraManager.getCameraCharacteristics(currentCameraId)
            previewSize = getPreviewOutputSize(windowManager.defaultDisplay, characteristics, SurfaceHolder::class.java)
            surfaceView.setAspectRatio(previewSize.width, previewSize.height)

            // --- PERBAIKAN ---
            // ImageReader diset ke ukuran preview untuk menangkap gambar dengan kualitas lebih baik
            // Penskalaan akan dilakukan kemudian
            imageReader = ImageReader.newInstance(previewSize.width, previewSize.height, ImageFormat.YUV_420_888, 3)

            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                frameSkipCounter++
                if (frameSkipCounter % (FRAME_SKIP_RATE + 1) == 0) {
                    if (isProcessingFrame.compareAndSet(false, true)) {
                        inferenceHandler?.post { processImage(image) }
                    } else {
                        image.close()
                    }
                } else {
                    image.close()
                }
            }, backgroundHandler)

            startBackgroundThread()
            cameraManager.openCamera(currentCameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    cameraDevice = camera
                    createCameraPreviewSession()
                }
                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    camera.close()
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpenCloseLock.release()
                    camera.close()
                    finish()
                }
            }, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Error opening camera: ${e.message}")
        }
    }

    // --- PERBAIKAN ---
    // Logika proses gambar disederhanakan dan penskalaan yang salah dihapus.
    private fun processImage(image: android.media.Image) {
        try {
            val startTime = System.currentTimeMillis()
            val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            yuvToRgbConverter.yuvToRgb(image, bitmap)

            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)
            bitmap.recycle() // Recycle bitmap asli setelah di-scaling

            modelBinding?.let { model ->
                val (results, inferenceTime) = model.detect(scaledBitmap)

                // Logika penskalaan yang salah di sini DIHAPUS.
                // 'results' sekarang berisi koordinat relatif terhadap gambar 640x640.
                // Biarkan OverlayView yang menangani penskalaan ke layar.

                val totalProcessingTime = System.currentTimeMillis() - startTime

                runOnUiThread {
                    // Kirim 'results' yang asli langsung ke OverlayView
                    overlayView.setResults(results)

                    if (results.isNotEmpty()) {
                        val topResult = results.first()
                        gestureText.text = topResult.label.uppercase()
                        confidenceText.text = "Akurasi: ${String.format("%.1f", topResult.confidence * 100)}%"
                    } else {
                        gestureText.text = "Mendeteksi..."
                        confidenceText.text = "Akurasi: --"
                    }
                    updateFps()
                    fpsText.text = "FPS: ${String.format("%.1f", currentFps)} | ${totalProcessingTime}ms"
                }
            }
            scaledBitmap.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
        } finally {
            image.close()
            isProcessingFrame.set(false)
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val surface = surfaceView.holder.surface
            val imageReaderSurface = imageReader.surface
            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            previewRequestBuilder.addTarget(surface)
            previewRequestBuilder.addTarget(imageReaderSurface)

            cameraDevice.createCaptureSession(listOf(surface, imageReaderSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        updatePreview()
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@KameraGestureActivity, "Camera preview failed", Toast.LENGTH_SHORT).show()
                    }
                }, backgroundHandler)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updatePreview() {
        try {
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            cameraCaptureSession.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Fungsi lifecycle lainnya (onResume, onPause, dll) tidak diubah
    // ... (sisa kode seperti onRequestPermissionsResult, start/stopBackgroundThread, dll.)
    // ... Anda bisa menyalin sisa kode dari file asli Anda.
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread?.looper!!)
        inferenceThread = HandlerThread("InferenceThread", Thread.MAX_PRIORITY).also { it.start() }
        inferenceHandler = Handler(inferenceThread?.looper!!)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        inferenceThread?.quitSafely()
        try {
            backgroundThread?.join()
            inferenceThread?.join()
            backgroundThread = null
            inferenceThread = null
            backgroundHandler = null
            inferenceHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            if (::cameraCaptureSession.isInitialized) cameraCaptureSession.close()
            if (::cameraDevice.isInitialized) cameraDevice.close()
            if (::imageReader.isInitialized) imageReader.close()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        if (surfaceView.holder.surface.isValid) {
            checkCameraPermissionAndOpen()
        }
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        modelBinding?.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Izin kamera ditolak", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun checkCameraPermissionAndOpen() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE)
        } else {
            openCamera()
        }
    }

    private fun updateFps() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        val elapsedTime = currentTime - lastFpsTime

        if (elapsedTime >= 1000) {
            currentFps = frameCount.toFloat() / (elapsedTime / 1000f)
            frameCount = 0
            lastFpsTime = currentTime
        }
    }
}