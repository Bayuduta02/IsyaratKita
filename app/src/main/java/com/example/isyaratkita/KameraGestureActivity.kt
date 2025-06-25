package com.example.isyaratkita

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
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
import com.example.isyaratkita.utils.YuvToRgbConverter
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.max

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

    // Background thread untuk image processing
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    // Semaphore untuk mencegah concurrent processing
    private val cameraOpenCloseLock = Semaphore(1)
    private var isProcessingFrame = false

    // FPS tracking
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f

    private val cameraManager: CameraManager by lazy {
        getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private var currentCameraId: String = ""
    private var isBackCamera: Boolean = true

    companion object {
        private const val CAMERA_REQUEST_CODE = 1001
        private const val MAX_PREVIEW_WIDTH = 1920
        private const val MAX_PREVIEW_HEIGHT = 1080
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

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                closeCamera()
            }
        })
    }

    private fun initializeModel() {
        try {
            yuvToRgbConverter = YuvToRgbConverter(this)
            // Initialize model binding
            modelBinding = YoloModelBinding(this)
            gestureText.text = "Model initialized"
        } catch (e: Exception) {
            gestureText.text = "Error initializing model: ${e.message}"
            e.printStackTrace()
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread?.looper!!)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
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
                Toast.makeText(this, "Izin kamera ditolak. Tidak bisa menjalankan kamera.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        try {
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }

            // Find camera dengan facing yang diinginkan
            currentCameraId = cameraManager.cameraIdList.first { id ->
                val characteristics = cameraManager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (isBackCamera) {
                    facing == CameraCharacteristics.LENS_FACING_BACK
                } else {
                    facing == CameraCharacteristics.LENS_FACING_FRONT
                }
            }

            val characteristics = cameraManager.getCameraCharacteristics(currentCameraId)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

            // Pilih preview size yang optimal
            previewSize = chooseOptimalSize(
                map.getOutputSizes(SurfaceHolder::class.java),
                surfaceView.width,
                surfaceView.height,
                Size(MAX_PREVIEW_WIDTH, MAX_PREVIEW_HEIGHT)
            )

            surfaceView.setAspectRatio(previewSize.width, previewSize.height)
            overlayView.setPreviewSize(previewSize.width, previewSize.height)

            // Setup ImageReader untuk processing
            imageReader = ImageReader.newInstance(
                previewSize.width,
                previewSize.height,
                ImageFormat.YUV_420_888,
                2
            )

            imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

            cameraManager.openCamera(currentCameraId, cameraStateCallback, backgroundHandler)

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error opening camera: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private val imageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        // Skip frame jika masih memproses frame sebelumnya
        if (isProcessingFrame) return@OnImageAvailableListener

        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener

        isProcessingFrame = true

        // Process in background thread to avoid UI jank
        backgroundHandler?.post {
            try {
                // Convert YUV to RGB efficiently
                val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                yuvToRgbConverter.yuvToRgb(image, bitmap)
                
                // Close image as soon as possible to free resources
                image.close()
                
                // Process detection
                processDetection(bitmap)
                
            } catch (e: Exception) {
                e.printStackTrace()
                image.close()
            } finally {
                isProcessingFrame = false
            }
        }
    }

    private fun processDetection(bitmap: Bitmap) {
        try {
            // Resize bitmap for faster processing if needed
            val resizedBitmap = resizeBitmapIfNeeded(bitmap)
            
            // Run detection with our new model binding
            val (detections, inferenceTime) = modelBinding?.detect(resizedBitmap) ?: Pair(emptyList(), 0L)

            // Update FPS
            updateFPS()

            // Update UI on main thread
            runOnUiThread {
                updateDetectionResults(detections, inferenceTime)
            }

        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                gestureText.text = "Detection error: ${e.message}"
            }
        }
    }
    
    private fun resizeBitmapIfNeeded(bitmap: Bitmap): Bitmap {
        // If bitmap is too large, resize it for faster processing
        // YOLOv8 works well with 640x640, so we don't need larger images
        val maxSize = 640
        
        if (bitmap.width > maxSize || bitmap.height > maxSize) {
            val ratio = maxSize.toFloat() / max(bitmap.width, bitmap.height)
            val newWidth = (bitmap.width * ratio).toInt()
            val newHeight = (bitmap.height * ratio).toInt()
            
            return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        }
        
        return bitmap
    }

    private fun updateDetectionResults(results: List<YoloModelBinding.Detection>, inferenceTime: Long) {
        if (results.isNotEmpty()) {
            val bestResult = results.maxByOrNull { it.confidence }
            bestResult?.let { result ->
                gestureText.text = result.label.uppercase()
                confidenceText.text = "Confidence: ${String.format("%.2f", result.confidence * 100)}%"

                // Update overlay dengan bounding boxes
                val boxes = results.map {
                    it.boundingBox to "${it.label} (${String.format("%.2f", it.confidence)})"
                }
                overlayView.setBoxesAndLabels(boxes)
            }
        } else {
            gestureText.text = getString(R.string.detecting_gesture)
            confidenceText.text = "Confidence: --"
            overlayView.clearBoxes()
        }

        // Update processing info with inference time
        fpsText.text = "FPS: $currentFps | ${inferenceTime}ms"
    }

    private fun updateFPS() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsTime >= 1000) {
            currentFps = frameCount.toFloat()
            frameCount = 0
            lastFpsTime = currentTime
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(device: CameraDevice) {
            cameraOpenCloseLock.release()
            cameraDevice = device
            createCameraPreviewSession()
        }

        override fun onDisconnected(device: CameraDevice) {
            cameraOpenCloseLock.release()
            device.close()
        }

        override fun onError(device: CameraDevice, error: Int) {
            cameraOpenCloseLock.release()
            device.close()
            runOnUiThread {
                Toast.makeText(this@KameraGestureActivity, "Camera error: $error", Toast.LENGTH_SHORT).show()
            }
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
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@KameraGestureActivity, "Camera preview failed", Toast.LENGTH_SHORT).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updatePreview() {
        try {
            // Set autofocus mode for better image quality
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            )
            
            // Set optimal exposure for real-time processing
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_ON
            )
            
            // Set optimal frame rate untuk real-time processing (20-30 FPS)
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                android.util.Range(20, 30)
            )
            
            // Set video stabilization if available
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
            )
            
            // Set optimal JPEG quality (for image capture if needed)
            previewRequestBuilder.set(
                CaptureRequest.JPEG_QUALITY,
                95.toByte()
            )

            cameraCaptureSession.setRepeatingRequest(
                previewRequestBuilder.build(),
                null,
                backgroundHandler
            )
        } catch (e: Exception) {
            e.printStackTrace()
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
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    private fun chooseOptimalSize(
        choices: Array<Size>,
        textureViewWidth: Int,
        textureViewHeight: Int,
        maxSize: Size
    ): Size {
        val bigEnough = mutableListOf<Size>()
        val notBigEnough = mutableListOf<Size>()
        val w = maxSize.width
        val h = maxSize.height

        for (option in choices) {
            if (option.width <= w && option.height <= h) {
                if (option.width >= textureViewWidth && option.height >= textureViewHeight) {
                    bigEnough.add(option)
                } else {
                    notBigEnough.add(option)
                }
            }
        }

        return when {
            bigEnough.size > 0 -> bigEnough.minByOrNull { it.width * it.height }!!
            notBigEnough.size > 0 -> notBigEnough.maxByOrNull { it.width * it.height }!!
            else -> choices[0]
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
}