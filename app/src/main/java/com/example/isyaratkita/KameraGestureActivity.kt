package com.example.isyaratkita

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.example.isyaratkita.utils.orientForInference
import com.example.isyaratkita.utils.toRgbaBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Kamera modern via CameraX — kompatibel HP flagship seperti iQOO Neo 10.
 * ImageAnalysis langsung 640×640 RGBA, tanpa konversi YUV manual.
 */
class KameraGestureActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var gestureText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var fpsText: TextView
    private lateinit var closeButton: ImageButton
    private lateinit var switchCameraButton: ImageButton
    private lateinit var overlayView: OverlayView

    private var modelBinding: YoloModelBinding? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private val isProcessingFrame = AtomicBoolean(false)
    private val modelInputSize = 640

    private var inputBitmap: Bitmap? = null

    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f

    companion object {
        private const val TAG = "KameraGestureActivity"
        private const val CAMERA_REQUEST_CODE = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupImmersiveMode()
        setContentView(R.layout.camera_activity)
        initViews()
        setupClickListeners()
        initializeModel()
    }

    private fun setupImmersiveMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.getInsetsController(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun initViews() {
        previewView = findViewById(R.id.camera_preview)
        gestureText = findViewById(R.id.gesture_text)
        confidenceText = findViewById(R.id.confidence_text)
        fpsText = findViewById(R.id.fps_text)
        closeButton = findViewById(R.id.btn_close)
        switchCameraButton = findViewById(R.id.btn_switch_camera)
        overlayView = findViewById(R.id.overlay_view)

        previewView.implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
    }

    private fun setupClickListeners() {
        closeButton.setOnClickListener { finish() }
        switchCameraButton.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            startCamera()
        }
    }

    private fun initializeModel() {
        try {
            val assetList = assets.list("") ?: emptyArray()
            if (!assetList.contains("model.tflite")) {
                throw Exception("File model.tflite tidak ditemukan di assets")
            }

            val binding = YoloModelBinding(this)
            if (!binding.isReady) {
                throw Exception(binding.initError ?: "Model gagal dimuat")
            }
            modelBinding = binding
            inputBitmap = Bitmap.createBitmap(modelInputSize, modelInputSize, Bitmap.Config.ARGB_8888)
            gestureText.text = "Model siap"
            Log.i(TAG, "Model initialized successfully")
        } catch (e: Exception) {
            gestureText.text = "Error: ${e.message}"
            Log.e(TAG, "Error: ${e.message}", e)
            Toast.makeText(this, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onResume() {
        super.onResume()
        setupImmersiveMode()
        if (analysisExecutor == null || analysisExecutor!!.isShutdown) {
            analysisExecutor = Executors.newSingleThreadExecutor()
        }
        checkCameraPermissionAndStart()
    }

    override fun onPause() {
        cameraProvider?.unbindAll()
        analysisExecutor?.shutdownNow()
        analysisExecutor = null
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        modelBinding?.close()
        inputBitmap?.recycle()
        inputBitmap = null
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, "Izin kamera ditolak", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun checkCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE
            )
        } else {
            startCamera()
        }
    }

    private fun startCamera() {
        val executor = analysisExecutor ?: return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                cameraProvider = providerFuture.get()
                bindCameraUseCases(executor)
            } catch (e: Exception) {
                Log.e(TAG, "Camera provider error: ${e.message}", e)
                runOnUiThread {
                    Toast.makeText(this, "Kamera gagal: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases(executor: ExecutorService) {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        val rotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
        val targetSize = Size(modelInputSize, modelInputSize)

        val preview = Preview.Builder()
            .setTargetRotation(rotation)
            .build()
            .also { it.surfaceProvider = previewView.surfaceProvider }

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(targetSize)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(rotation)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(executor) { imageProxy ->
                    if (!isProcessingFrame.compareAndSet(false, true)) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    try {
                        processFrame(imageProxy)
                    } catch (e: Exception) {
                        Log.e(TAG, "Frame error: ${e.message}", e)
                    } finally {
                        isProcessingFrame.set(false)
                        imageProxy.close()
                    }
                }
            }

        val selector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        try {
            provider.bindToLifecycle(this, selector, preview, imageAnalysis)
            Log.i(TAG, "CameraX bound → lens=${if (lensFacing == CameraSelector.LENS_FACING_BACK) "back" else "front"}, analysis=${targetSize.width}x${targetSize.height}")
        } catch (e: Exception) {
            Log.e(TAG, "Bind failed: ${e.message}", e)
            Toast.makeText(this, "Kamera tidak tersedia: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun processFrame(imageProxy: androidx.camera.core.ImageProxy) {
        val t0 = System.currentTimeMillis()
        val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
        val rotation = imageProxy.imageInfo.rotationDegrees

        var rgba = imageProxy.toRgbaBitmap()
        rgba = rgba.orientForInference(rotation, mirror)

        val modelInput = prepareModelInput(rgba)
        rgba.recycle()

        modelBinding?.let { model ->
            val (results, _) = model.detect(modelInput)
            val totalMs = System.currentTimeMillis() - t0

            runOnUiThread {
                overlayView.setPreviewSize(previewView.width, previewView.height)
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
    }

    /** Letterbox ke 640×640 — sama seperti saat training YOLO. */
    private fun prepareModelInput(source: Bitmap): Bitmap {
        val output = inputBitmap ?: Bitmap.createBitmap(modelInputSize, modelInputSize, Bitmap.Config.ARGB_8888)
        inputBitmap = output

        val canvas = Canvas(output)
        canvas.drawColor(Color.BLACK)

        val scale = minOf(
            modelInputSize.toFloat() / source.width,
            modelInputSize.toFloat() / source.height
        )
        val matrix = Matrix()
        matrix.postScale(scale, scale)
        matrix.postTranslate(
            (modelInputSize - source.width * scale) / 2f,
            (modelInputSize - source.height * scale) / 2f
        )
        canvas.drawBitmap(source, matrix, null)
        return output
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
