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
import android.view.Surface
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
import kotlin.math.abs

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
    private var sensorOrientation: Int = 0
    private var screenWidth: Int = 0
    private var screenHeight: Int = 0

    companion object {
        private const val TAG = "KameraGestureActivity"
        private const val CAMERA_REQUEST_CODE = 1001
        // Kita pilih resolusi yang tinggi untuk kualitas yang lebih baik
        private const val MAX_PREVIEW_WIDTH = 1920
        private const val MAX_PREVIEW_HEIGHT = 1080
        private const val DESIRED_PREVIEW_FPS = 30
        
        // Rasio aspek 16:9
        private const val ASPECT_RATIO_16_9 = 16.0f / 9.0f
        
        // Toleransi untuk rasio aspek
        private const val ASPECT_RATIO_TOLERANCE = 0.1f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_FULLSCREEN or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY

        setContentView(R.layout.camera_activity)

        // Dapatkan ukuran layar
        val displayMetrics = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(displayMetrics)
        screenWidth = displayMetrics.widthPixels
        screenHeight = displayMetrics.heightPixels
        Log.d(TAG, "Screen size: $screenWidth x $screenHeight")

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
        
        // Tambahkan logging untuk memverifikasi bahwa views ditemukan
        Log.d(TAG, "SurfaceView: $surfaceView")
        Log.d(TAG, "GestureText: $gestureText")
        Log.d(TAG, "ConfidenceText: $confidenceText")
        Log.d(TAG, "FPSText: $fpsText")
        Log.d(TAG, "CloseButton: $closeButton")
        Log.d(TAG, "SwitchCameraButton: $switchCameraButton")
        Log.d(TAG, "OverlayView: $overlayView")
    }

    private fun setupClickListeners() {
        // Tombol close untuk menutup aktivitas
        closeButton.setOnClickListener { 
            Log.d(TAG, "Close button clicked")
            finish() 
        }
        
        // Tombol switch camera untuk berganti antara kamera depan dan belakang
        switchCameraButton.setOnClickListener { 
            Log.d(TAG, "Switch camera button clicked")
            isBackCamera = !isBackCamera
            
            // Tutup kamera saat ini
            closeCamera()
            
            // Buka kamera dengan facing yang baru
            checkCameraPermissionAndOpen()
        }
    }

    private fun setupSurfaceView() {
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                checkCameraPermissionAndOpen()
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                Log.d(TAG, "Surface changed: $width x $height")
            }
            
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
            
            // Dapatkan orientasi sensor kamera
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            Log.d(TAG, "Sensor orientation: $sensorOrientation")
            
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

            // Pilih preview size yang optimal dengan rasio 16:9
            val previewSizes = map.getOutputSizes(SurfaceHolder::class.java)
            
            // Gunakan utilitas CameraSizes untuk mendapatkan ukuran preview optimal
            previewSize = getPreviewOutputSize(
                windowManager.defaultDisplay, 
                characteristics, 
                SurfaceHolder::class.java
            )
            
            Log.d(TAG, "Selected preview size: ${previewSize.width} x ${previewSize.height}")
            
            // Sesuaikan rasio aspek surface view
            surfaceView.setAspectRatio(previewSize.width, previewSize.height)
            
            // Siapkan ImageReader untuk processing frame
            imageReader = ImageReader.newInstance(
                previewSize.width, 
                previewSize.height, 
                ImageFormat.YUV_420_888, 
                2 // Buffer frames
            )
            
            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                image?.let { processImage(it) }
            }, backgroundHandler)

            // Mulai background thread untuk processing
            startBackgroundThread()

            // Buka kamera
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
            e.printStackTrace()
            Toast.makeText(this, "Error opening camera: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun needsSwappedDimensions(screenOrientation: Int): Boolean {
        var swappedDimensions = false
        when (screenOrientation) {
            Surface.ROTATION_0, Surface.ROTATION_180 -> {
                if (sensorOrientation == 90 || sensorOrientation == 270) {
                    swappedDimensions = true
                }
            }
            Surface.ROTATION_90, Surface.ROTATION_270 -> {
                if (sensorOrientation == 0 || sensorOrientation == 180) {
                    swappedDimensions = true
                }
            }
        }
        
        Log.d(TAG, "Screen orientation: $screenOrientation, sensor orientation: $sensorOrientation")
        Log.d(TAG, "Swapped dimensions: $swappedDimensions")
        
        return swappedDimensions
    }

    private fun processImage(image: android.media.Image) {
        // Hindari proses bersamaan
        if (isProcessingFrame) {
            image.close()
            return
        }
        
        isProcessingFrame = true
        
        try {
            // Konversi YUV ke Bitmap
            val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            yuvToRgbConverter.yuvToRgb(image, bitmap)
            
            // Proses dengan model
            modelBinding?.let { model ->
                val (results, inferenceTime) = model.detect(bitmap)
                
                // Update UI di thread utama
                runOnUiThread {
                    // Update overlay dengan hasil deteksi
                    overlayView.setResults(results)
                    
                    // Update teks gesture dan confidence
                    if (results.isNotEmpty()) {
                        val topResult = results.first()
                        gestureText.text = topResult.label
                        confidenceText.text = "Confidence: ${String.format("%.2f", topResult.confidence * 100)}%"
                    } else {
                        gestureText.text = "Detecting gesture..."
                        confidenceText.text = "Confidence: --"
                    }
                    
                    // Update FPS dan inference time
                    fpsText.text = "FPS: ${String.format("%.1f", currentFps)} | ${inferenceTime}ms"
                    
                    // Hitung FPS
                    updateFps()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
            e.printStackTrace()
        } finally {
            image.close()
            isProcessingFrame = false
        }
    }

    private fun updateFps() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        val elapsedTime = currentTime - lastFpsTime
        
        if (elapsedTime > 1000) {
            currentFps = frameCount.toFloat() / (elapsedTime / 1000f)
            fpsText.text = "FPS: ${String.format("%.1f", currentFps)} | ${elapsedTime}ms"
            
            // Reset counter
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
                android.util.Range(20, DESIRED_PREVIEW_FPS)
            )
            
            // Set video stabilization if available
            previewRequestBuilder.set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
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

    /**
     * Memilih ukuran preview yang optimal dengan preferensi rasio 16:9
     */
    private fun chooseOptimalSize(
        choices: Array<Size>,
        width: Int,
        height: Int,
        maxSize: Size
    ): Size {
        // Daftar semua ukuran yang tersedia untuk debugging
        Log.d(TAG, "Available preview sizes: ${choices.joinToString { "${it.width}x${it.height}" }}")
        
        // Filter ukuran yang tidak melebihi batasan maksimal
        val validSizes = choices.filter { 
            it.width <= maxSize.width && it.height <= maxSize.height
        }
        
        if (validSizes.isEmpty()) {
            Log.e(TAG, "Tidak dapat menemukan ukuran preview yang sesuai")
            return choices.sortedBy { 
                abs(it.width * it.height - width * height) 
            }.first()
        }
        
        // Cari ukuran yang memiliki rasio 16:9
        val aspectRatio16by9Sizes = validSizes.filter { size ->
            val ratio = size.width.toFloat() / size.height.toFloat()
            val isRatio16by9 = abs(ratio - ASPECT_RATIO_16_9) < ASPECT_RATIO_TOLERANCE
            Log.d(TAG, "Size ${size.width}x${size.height}, ratio: $ratio, is 16:9: $isRatio16by9")
            isRatio16by9
        }
        
        // Logika pemilihan: prioritaskan rasio 16:9, kemudian pilih yang terbesar
        return when {
            // Jika ada ukuran dengan rasio 16:9, pilih yang terbesar
            aspectRatio16by9Sizes.isNotEmpty() -> {
                val largestSize = aspectRatio16by9Sizes.maxByOrNull { it.width * it.height }!!
                Log.d(TAG, "Menggunakan ukuran preview 16:9: ${largestSize.width}x${largestSize.height}")
                largestSize
            }
            // Jika tidak ada, cari ukuran terbesar yang tersedia
            else -> {
                val bestSize = validSizes.maxByOrNull { it.width * it.height }!!
                Log.d(TAG, "Tidak ada ukuran dengan rasio 16:9, menggunakan ukuran terbesar: ${bestSize.width}x${bestSize.height}")
                bestSize
            }
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