package com.example.isyaratkita

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.RectF
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
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
    private val FRAME_SKIP_RATE = 2

    private val cameraManager: CameraManager by lazy {
        getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private var currentCameraId: String = ""
    private var isBackCamera: Boolean = true

    // PERBAIKAN: Gunakan 320 sesuai model.tflite Anda
    private val modelInputSize = 320

    companion object {
        private const val TAG = "KameraGestureActivity"
        private const val CAMERA_REQUEST_CODE = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // PERBAIKAN: Gunakan WindowInsetsController modern
        setupImmersiveMode()

        setContentView(R.layout.camera_activity)
        initViews()
        setupClickListeners()
        setupSurfaceView()
        initializeModel()
    }

    // PERBAIKAN: Setup immersive mode dengan API modern
    private fun setupImmersiveMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val controller = WindowCompat.getInsetsController(window, window.decorView)
        controller.apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
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
                Log.d(TAG, "surfaceCreated")
                checkCameraPermissionAndOpen()
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                Log.d(TAG, "surfaceChanged: ${width}x${height}")
                overlayView.setPreviewSize(width, height)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.d(TAG, "surfaceDestroyed")
                closeCamera()
            }
        })
    }

    private fun initializeModel() {
        try {
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
            Log.i(TAG, "Model initialized successfully")
        } catch (e: Exception) {
            gestureText.text = "Error: ${e.message}"
            Log.e(TAG, "Error initializing model: ${e.message}", e)
            Toast.makeText(this, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
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
            previewSize = getPreviewOutputSize(
                windowManager.defaultDisplay,
                characteristics,
                SurfaceHolder::class.java
            )

            // PERBAIKAN: Pastikan aspect ratio selalu di-set
            surfaceView.post {
                surfaceView.setAspectRatio(previewSize.width, previewSize.height)
                Log.d(TAG, "Preview size set: ${previewSize.width}x${previewSize.height}")
            }

            imageReader = ImageReader.newInstance(
                previewSize.width,
                previewSize.height,
                ImageFormat.YUV_420_888,
                3
            )

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

            cameraManager.openCamera(currentCameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    cameraDevice = camera
                    createCameraPreviewSession()
                    Log.i(TAG, "Camera opened successfully")
                }

                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    camera.close()
                    Log.w(TAG, "Camera disconnected")
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpenCloseLock.release()
                    camera.close()
                    Log.e(TAG, "Camera error: $error")
                    finish()
                }
            }, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Error opening camera: ${e.message}", e)
            cameraOpenCloseLock.release()
        }
    }

    private fun processImage(image: android.media.Image) {
        try {
            val startTime = System.currentTimeMillis()

            // Convert YUV to RGB
            val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            yuvToRgbConverter.yuvToRgb(image, bitmap)

            // Scale to model input size (320x320)
            val scaledBitmap = Bitmap.createScaledBitmap(
                bitmap,
                modelInputSize,
                modelInputSize,
                true
            )
            bitmap.recycle()

            modelBinding?.let { model ->
                val (results, inferenceTime) = model.detect(scaledBitmap)

                // PERBAIKAN: Scale coordinates dari model space (320x320) ke preview space
                // Detection.boundingBox adalah RectF dalam koordinat 320x320
                val scaleX = previewSize.width.toFloat() / modelInputSize
                val scaleY = previewSize.height.toFloat() / modelInputSize

                val scaledResults = results.map { detection ->
                    YoloModelBinding.Detection(
                        boundingBox = RectF(
                            detection.boundingBox.left * scaleX,
                            detection.boundingBox.top * scaleY,
                            detection.boundingBox.right * scaleX,
                            detection.boundingBox.bottom * scaleY
                        ),
                        label = detection.label,
                        confidence = detection.confidence
                    )
                }

                val totalProcessingTime = System.currentTimeMillis() - startTime

                runOnUiThread {
                    overlayView.setResults(scaledResults)

                    if (scaledResults.isNotEmpty()) {
                        val topResult = scaledResults.first()
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
            Log.e(TAG, "Error processing image: ${e.message}", e)
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

            cameraDevice.createCaptureSession(
                listOf(surface, imageReaderSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        updatePreview()
                        Log.i(TAG, "Camera preview session configured")
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Camera preview session configuration failed")
                        Toast.makeText(
                            this@KameraGestureActivity,
                            "Camera preview failed",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating preview session: ${e.message}", e)
        }
    }

    private fun updatePreview() {
        try {
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            )
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_ON
            )
            cameraCaptureSession.setRepeatingRequest(
                previewRequestBuilder.build(),
                null,
                backgroundHandler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error updating preview: ${e.message}", e)
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread?.looper!!)
        inferenceThread = HandlerThread("InferenceThread", Thread.MAX_PRIORITY).also { it.start() }
        inferenceHandler = Handler(inferenceThread?.looper!!)
        Log.d(TAG, "Background threads started")
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
            Log.d(TAG, "Background threads stopped")
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping background threads", e)
        }
    }

    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            if (::cameraCaptureSession.isInitialized) {
                cameraCaptureSession.close()
            }
            if (::cameraDevice.isInitialized) {
                cameraDevice.close()
            }
            if (::imageReader.isInitialized) {
                imageReader.close()
            }
            Log.d(TAG, "Camera closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing camera", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    override fun onResume() {
        super.onResume()
        // PERBAIKAN: Re-apply immersive mode setiap onResume
        setupImmersiveMode()
        startBackgroundThread()

        if (surfaceView.holder.surface.isValid) {
            checkCameraPermissionAndOpen()
        }
        Log.d(TAG, "onResume")
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
        Log.d(TAG, "onPause")
    }

    override fun onDestroy() {
        super.onDestroy()
        modelBinding?.close()
        Log.d(TAG, "onDestroy")
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
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE
            )
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