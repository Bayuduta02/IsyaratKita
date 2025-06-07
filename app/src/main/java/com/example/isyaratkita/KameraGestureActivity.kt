package com.example.isyaratkita

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
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
import com.example.skripsi.utils.AutoFitSurfaceView
import com.example.skripsi.utils.YuvToRgbConverter
import com.example.skripsi.utils.getPreviewOutputSize
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import android.app.AlertDialog
import android.util.Log

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
    private var objectDetector: ObjectDetector? = null

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
        private const val TAG = "KameraGestureActivity"
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
        initializeDetector()
    }

    private fun initViews() {
        surfaceView = findViewById(R.id.camera_preview)
        gestureText = findViewById(R.id.gesture_text)
        confidenceText = findViewById(R.id.confidence_text) // Tambahkan di layout
        fpsText = findViewById(R.id.fps_text) // Tambahkan di layout
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

    private fun initializeDetector() {
        try {
            yuvToRgbConverter = YuvToRgbConverter(this)
            
            // Initialize TFLite model with custom options
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(5)  // Limit detection results
                .setNumThreads(4)  // Use multiple threads for better performance
                .build()
                
            objectDetector = ObjectDetector(
                context = this,
                modelPath = "sign_language_ssd_mobilenetv2.tflite",
                options = options
            )

            // Verify model initialization
            if (objectDetector == null) {
                throw Exception("Failed to initialize object detector")
            }

            gestureText.text = "Detector initialized"
            Log.d(TAG, "Object detector initialized successfully")
            
        } catch (e: Exception) {
            val errorMsg = "Error initializing detector: ${e.message}"
            gestureText.text = errorMsg
            Log.e(TAG, errorMsg, e)
            e.printStackTrace()
            
            // Show error dialog to user
            runOnUiThread {
                AlertDialog.Builder(this)
                    .setTitle("Initialization Error")
                    .setMessage("Failed to initialize gesture detector. Please restart the app.")
                    .setPositiveButton("OK") { _, _ -> finish() }
                    .show()
            }
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
        if (isProcessingFrame) {
            reader.acquireLatestImage()?.close()
            return@OnImageAvailableListener
        }

        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        var bitmap: Bitmap? = null

        isProcessingFrame = true

        try {
            // Convert YUV to RGB
            bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            yuvToRgbConverter.yuvToRgb(image, bitmap)

            // Process detection
            processDetection(bitmap)

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            image.close()
            bitmap?.recycle()
            isProcessingFrame = false
        }
    }

    private fun processDetection(bitmap: Bitmap) {
        try {
            val startTime = System.currentTimeMillis()

            // Minimum confidence threshold
            val CONFIDENCE_THRESHOLD = 0.5f

            // Run detection
            val results = objectDetector?.detect(bitmap)?.filter { it.score >= CONFIDENCE_THRESHOLD }
                ?: emptyList()

            val processingTime = System.currentTimeMillis() - startTime

            // Update UI on main thread
            runOnUiThread {
                if (results.isNotEmpty()) {
                    // Sort by confidence and get the best result
                    val bestResult = results.maxByOrNull { it.score }
                    bestResult?.let { result ->
                        // Update UI with detection result
                        gestureText.text = result.label.uppercase()
                        confidenceText.text = String.format("Confidence: %.1f%%", result.score * 100)

                        // Update overlay with all detected boxes above threshold
                        val boxes = results.map { detection ->
                            detection.boundingBox to "${detection.label}\n${String.format("%.1f%%", detection.score * 100)}"
                        }
                        overlayView.setBoxesAndLabels(boxes)
                    }
                } else {
                    gestureText.text = getString(R.string.detecting_gesture)
                    confidenceText.text = "Confidence: --"
                    overlayView.clearBoxes()
                }

                // Update FPS and processing time
                fpsText.text = String.format("FPS: %.1f | %dms", currentFps, processingTime)
            }

        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                gestureText.text = "Detection error: ${e.message}"
            }
        }
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
            // Optimize camera settings for real-time detection
            previewRequestBuilder.apply {
                // Set auto-focus mode
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                
                // Optimize auto-exposure for faster frame processing
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AE_ANTIBANDING_MODE, CaptureRequest.CONTROL_AE_ANTIBANDING_MODE_AUTO)
                
                // Set optimal frame rate range for real-time detection (30fps)
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, android.util.Range(30, 30))
                
                // Optimize auto-white-balance
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
            }

            cameraCaptureSession.setRepeatingRequest(
                previewRequestBuilder.build(),
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        // Monitor frame capture rate
                        updateFPS()
                    }
                },
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
        try {
            // Clean up camera resources
            closeCamera()
            stopBackgroundThread()
            
            // Clean up YuvToRgbConverter
            if (::yuvToRgbConverter.isInitialized) {
                yuvToRgbConverter.destroy()
            }
            
            // Clean up object detector
            objectDetector?.close()
            
            // Clean up any remaining resources
            if (::imageReader.isInitialized) {
                imageReader.close()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            super.onDestroy()
        }
    }
}