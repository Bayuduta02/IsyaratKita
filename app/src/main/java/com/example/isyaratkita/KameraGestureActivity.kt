package com.example.isyaratkita

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Build
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

    // FPS counter
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f

    private val cameraManager: CameraManager by lazy {
        getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private var isBackCamera: Boolean = true

    private val modelInputSize = 640

    // OPTIMASI: Reuse bitmap — tidak alokasi baru setiap frame
    private var rgbBitmap: Bitmap? = null
    private var scaledBitmap: Bitmap? = null
    private val scaleMatrix = Matrix()

    companion object {
        private const val TAG = "KameraGestureActivity"
        private const val CAMERA_REQUEST_CODE = 1001

        // OPTIMASI: Gunakan resolusi kamera lebih kecil → YUV conversion lebih cepat
        // 640x480 atau 1280x720 jauh lebih cepat dari 1920x1080
        private val PREFERRED_PREVIEW_SIZE = Size(1280, 720)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupImmersiveMode()
        setContentView(R.layout.camera_activity)
        initViews()
        setupClickListeners()
        setupSurfaceView()
        initializeModel()
    }

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
            // Cek hanya file TFLite — label sudah embed di dalam model YOLO26
            val assetList = assets.list("") ?: emptyArray()
            if (!assetList.contains("model.tflite"))
                throw Exception("File model.tflite tidak ditemukan di assets")

            yuvToRgbConverter = YuvToRgbConverter(this)
            modelBinding = YoloModelBinding(this)
            gestureText.text = "Model siap"
            Log.i(TAG, "Model initialized successfully")
        } catch (e: Exception) {
            gestureText.text = "Error: ${e.message}"
            Log.e(TAG, "Error: ${e.message}", e)
            Toast.makeText(this, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        try {
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS))
                throw RuntimeException("Timeout waiting to lock camera.")

            val cameraId = cameraManager.cameraIdList.first { id ->
                val facing = cameraManager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.LENS_FACING)
                if (isBackCamera) facing == CameraCharacteristics.LENS_FACING_BACK
                else facing == CameraCharacteristics.LENS_FACING_FRONT
            }

            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // OPTIMASI: Pilih resolusi lebih kecil untuk mempercepat YUV conversion
            previewSize = chooseBestPreviewSize(characteristics)
            Log.i(TAG, "Selected preview size: ${previewSize.width}x${previewSize.height}")

            surfaceView.post {
                surfaceView.setAspectRatio(previewSize.width, previewSize.height)
            }

            // OPTIMASI: Inisialisasi reusable bitmap sesuai preview size
            rgbBitmap = Bitmap.createBitmap(previewSize.width, previewSize.height, Bitmap.Config.ARGB_8888)
            scaledBitmap = Bitmap.createBitmap(modelInputSize, modelInputSize, Bitmap.Config.ARGB_8888)

            imageReader = ImageReader.newInstance(previewSize.width, previewSize.height, ImageFormat.YUV_420_888, 3)

            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                // OPTIMASI: Tidak ada frame skip — langsung cek apakah sedang proses
                if (isProcessingFrame.compareAndSet(false, true)) {
                    inferenceHandler?.post { processImage(image) }
                } else {
                    image.close()
                }
            }, backgroundHandler)

            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    cameraDevice = camera
                    createCameraPreviewSession()
                    Log.i(TAG, "Camera opened: $cameraId")
                }
                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    camera.close()
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

    /**
     * OPTIMASI: Pilih resolusi kamera terkecil yang >= modelInputSize (640)
     * Prioritas: 1280x720 > 960x540 > 640x480
     * Menghindari 1920x1080 yang lambat untuk di-convert
     */
    private fun chooseBestPreviewSize(characteristics: CameraCharacteristics): Size {
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return PREFERRED_PREVIEW_SIZE

        val availableSizes = map.getOutputSizes(ImageFormat.YUV_420_888)
            ?.sortedBy { it.width * it.height }
            ?: return PREFERRED_PREVIEW_SIZE

        Log.d(TAG, "Available sizes: ${availableSizes.joinToString { "${it.width}x${it.height}" }}")

        // Cari ukuran terkecil yang lebar dan tingginya >= modelInputSize
        val preferred = availableSizes.firstOrNull {
            it.width >= modelInputSize && it.height >= modelInputSize
        }

        return preferred ?: availableSizes.last()
    }

    private fun processImage(image: android.media.Image) {
        try {
            val t0 = System.currentTimeMillis()

            // OPTIMASI: Gunakan rgbBitmap yang sudah dialokasi — tidak buat baru
            val rgb = rgbBitmap ?: return
            yuvToRgbConverter.yuvToRgb(image, rgb)

            // OPTIMASI: Scale manual dengan Canvas ke scaledBitmap yang sudah ada
            val scaled = scaledBitmap ?: return
            val canvas = android.graphics.Canvas(scaled)
            scaleMatrix.setScale(
                modelInputSize.toFloat() / rgb.width,
                modelInputSize.toFloat() / rgb.height
            )
            canvas.drawBitmap(rgb, scaleMatrix, null)

            modelBinding?.let { model ->
                val (results, _) = model.detect(scaled)
                val totalMs = System.currentTimeMillis() - t0

                runOnUiThread {
                    overlayView.setResults(results)
                    if (results.isNotEmpty()) {
                        val top = results.first()
                        gestureText.text = top.label.uppercase()
                        confidenceText.text = "Akurasi: ${String.format("%.1f", top.confidence * 100)}%"
                    } else {
                        gestureText.text = "Mendeteksi..."
                        confidenceText.text = "Akurasi: --"
                    }
                    updateFps()
                    fpsText.text = "FPS: ${String.format("%.1f", currentFps)} | ${totalMs}ms"
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing: ${e.message}", e)
        } finally {
            image.close()
            isProcessingFrame.set(false)
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val surface = surfaceView.holder.surface
            val readerSurface = imageReader.surface

            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            previewRequestBuilder.addTarget(surface)
            previewRequestBuilder.addTarget(readerSurface)

            // OPTIMASI: Set frame rate eksplisit
            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                android.util.Range(15, 30))

            cameraDevice.createCaptureSession(
                listOf(surface, readerSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        updatePreview()
                        Log.i(TAG, "Preview session configured")
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Preview session failed")
                        Toast.makeText(this@KameraGestureActivity, "Camera preview failed", Toast.LENGTH_SHORT).show()
                    }
                }, backgroundHandler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating session: ${e.message}", e)
        }
    }

    private fun updatePreview() {
        try {
            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            cameraCaptureSession.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "Error updating preview: ${e.message}", e)
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
        inferenceThread = HandlerThread("InferenceThread", Thread.MAX_PRIORITY).also { it.start() }
        inferenceHandler = Handler(inferenceThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        inferenceThread?.quitSafely()
        try {
            backgroundThread?.join()
            inferenceThread?.join()
        } catch (e: InterruptedException) {
            Log.e(TAG, "Error stopping threads", e)
        } finally {
            backgroundThread = null; inferenceThread = null
            backgroundHandler = null; inferenceHandler = null
        }
    }

    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()
            if (::cameraCaptureSession.isInitialized) cameraCaptureSession.close()
            if (::cameraDevice.isInitialized) cameraDevice.close()
            if (::imageReader.isInitialized) imageReader.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing camera", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    override fun onResume() {
        super.onResume()
        setupImmersiveMode()
        startBackgroundThread()
        if (surfaceView.holder.surface.isValid) checkCameraPermissionAndOpen()
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        modelBinding?.close()
        rgbBitmap?.recycle(); rgbBitmap = null
        scaledBitmap?.recycle(); scaledBitmap = null
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) openCamera()
            else { Toast.makeText(this, "Izin kamera ditolak", Toast.LENGTH_SHORT).show(); finish() }
        }
    }

    private fun checkCameraPermissionAndOpen() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        } else {
            openCamera()
        }
    }

    private fun updateFps() {
        frameCount++
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTime
        if (elapsed >= 1000) {
            currentFps = frameCount * 1000f / elapsed
            frameCount = 0
            lastFpsTime = now
        }
    }
}