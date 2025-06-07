package com.example.isyaratkita

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class ObjectDetector(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null
    private val inputSize = 224  // Ukuran input sesuai model Colab
    private val numClasses = 32  // Jumlah kelas sesuai labelMap
    private val scoreThreshold = 0.5f

    // Mapping indeks ke label - sesuai urutan training di Colab
    private val labelMap = mapOf(
        0 to "a", 1 to "apa", 2 to "b", 3 to "c", 4 to "d", 5 to "e",
        6 to "f", 7 to "g", 8 to "h", 9 to "halo", 10 to "i", 11 to "j",
        12 to "k", 13 to "kabar", 14 to "l", 15 to "m", 16 to "n",
        17 to "nama", 18 to "o", 19 to "p", 20 to "perkenalkan",
        21 to "q", 22 to "r", 23 to "s", 24 to "saya", 25 to "t",
        26 to "u", 27 to "v", 28 to "w", 29 to "x", 30 to "y", 31 to "z"
    )

    init {
        try {
            // Load model dari assets
            val model = FileUtil.loadMappedFile(context, modelPath)
            
            // Konfigurasi interpreter
            val options = Interpreter.Options().apply {
                setNumThreads(4)  // Menggunakan 4 thread untuk performa
                setUseNNAPI(true) // Menggunakan Neural Network API jika tersedia
            }
            
            interpreter = Interpreter(model, options)
            
            // Tampilkan informasi model untuk debugging
            printModelInfo()
            
        } catch (e: Exception) {
            println("Error saat memuat model: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun printModelInfo() {
        interpreter?.let { interpreter ->
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor1 = interpreter.getOutputTensor(0) // Class probabilities [1,32]
            val outputTensor2 = interpreter.getOutputTensor(1) // Bounding box [1,4]

            println("\n=== INFORMASI MODEL ===")
            println("Input shape: ${inputTensor.shape().contentToString()}")
            println("Output1 (Class) shape: ${outputTensor1.shape().contentToString()}")
            println("Output2 (BBox) shape: ${outputTensor2.shape().contentToString()}")
            println("=====================\n")
        }
    }

    data class DetectionResult(
        val boundingBox: RectF,
        val label: String,
        val score: Float
    )

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize gambar ke ukuran input model
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // Alokasi ByteBuffer untuk input
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        
        // Konversi bitmap ke ByteBuffer dan normalisasi ke [0,1]
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixel in pixels) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        
        return byteBuffer
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        if (interpreter == null) {
            println("Interpreter belum diinisialisasi")
            return emptyList()
        }

        try {
            // Preprocess gambar input
            val inputBuffer = preprocessImage(bitmap)

            // Siapkan array output
            val classOutput = Array(1) { FloatArray(numClasses) }  // [1,32] untuk probabilitas kelas
            val bboxOutput = Array(1) { FloatArray(4) }           // [1,4] untuk bounding box

            // Jalankan inferensi
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = classOutput  // Output pertama: probabilitas kelas
            outputs[1] = bboxOutput   // Output kedua: bounding box

            interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

            // Post-process hasil deteksi
            val results = mutableListOf<DetectionResult>()
            
            // Ambil hasil prediksi
            val classProbs = classOutput[0]  // Probabilitas untuk setiap kelas
            val bbox = bboxOutput[0]         // Koordinat bounding box (normalized)
            
            // Cari kelas dengan probabilitas tertinggi
            var maxClassIdx = 0
            var maxProb = classProbs[0]
            for (i in 1 until classProbs.size) {
                if (classProbs[i] > maxProb) {
                    maxProb = classProbs[i]
                    maxClassIdx = i
                }
            }

            // Jika confidence melebihi threshold
            if (maxProb >= scoreThreshold) {
                // Konversi koordinat normalized ke pixel coordinates
                val xmin = bbox[0] * bitmap.width
                val ymin = bbox[1] * bitmap.height
                val xmax = bbox[2] * bitmap.width
                val ymax = bbox[3] * bitmap.height

                // Pastikan koordinat dalam range yang valid
                val boundingBox = RectF(
                    max(0f, min(xmin, bitmap.width.toFloat())),
                    max(0f, min(ymin, bitmap.height.toFloat())),
                    max(0f, min(xmax, bitmap.width.toFloat())),
                    max(0f, min(ymax, bitmap.height.toFloat()))
                )

                // Ambil label dari map
                val label = labelMap[maxClassIdx] ?: "unknown"
                
                // Tambahkan hasil deteksi
                results.add(DetectionResult(boundingBox, label, maxProb))
            }

            return results

        } catch (e: Exception) {
            e.printStackTrace()
            println("Error saat deteksi: ${e.message}")
            return emptyList()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    private fun validateModelConfiguration() {
        interpreter?.let { interpreter ->
            try {
                println("\n=== INFORMASI MODEL TFLITE ===")
                
                // Periksa Input
                val inputTensor = interpreter.getInputTensor(0)
                val inputShape = inputTensor.shape()
                println("\nINFORMASI INPUT:")
                println("- Shape: ${inputShape.contentToString()}")
                println("- Data Type: ${inputTensor.dataType()}")
                println("- Quantization: ${inputTensor.quantizationParams()}")
                
                // Validasi Input Shape
                if (inputShape.contentEquals(intArrayOf(1, 224, 224, 3))) {
                    println("✅ Input shape sesuai dengan yang diharapkan")
                } else {
                    println("❌ Input shape tidak sesuai! Harusnya [1, 224, 224, 3]")
                }

                // Periksa Outputs
                println("\nINFORMASI OUTPUT:")
                
                // Output 1 - Class Probabilities (diubah dari Bounding Box)
                val outputTensor1 = interpreter.getOutputTensor(0)
                val outputShape1 = outputTensor1.shape()
                println("\nOutput 1 (Class Probabilities):")
                println("- Shape: ${outputShape1.contentToString()}")
                println("- Data Type: ${outputTensor1.dataType()}")
                println("- Quantization: ${outputTensor1.quantizationParams()}")
                
                // Output 2 - Bounding Box (diubah dari Class Probabilities)
                val outputTensor2 = interpreter.getOutputTensor(1)
                val outputShape2 = outputTensor2.shape()
                println("\nOutput 2 (Bounding Box):")
                println("- Shape: ${outputShape2.contentToString()}")
                println("- Data Type: ${outputTensor2.dataType()}")
                println("- Quantization: ${outputTensor2.quantizationParams()}")

                // Validasi Output Shapes
                val isOutput1Valid = outputShape1[1] == 32 // [batch_size, 32] untuk class probs
                val isOutput2Valid = outputShape2[1] == 4  // [batch_size, 4] untuk bbox
                
                if (isOutput1Valid) {
                    println("✅ Output 1 shape sesuai untuk klasifikasi")
                } else {
                    println("❌ Output 1 shape tidak sesuai! Harusnya [1, 32]")
                }
                
                if (isOutput2Valid) {
                    println("✅ Output 2 shape sesuai untuk bounding box")
                } else {
                    println("❌ Output 2 shape tidak sesuai! Harusnya [1, 4]")
                }

                // Periksa jumlah label
                println("\nVALIDASI LABEL:")
                println("- Jumlah kelas di model: ${outputShape1[1]}")
                println("- Jumlah label di labelMap: ${labelMap.size}")
                if (outputShape1[1] == labelMap.size) {
                    println("✅ Jumlah label sesuai")
                } else {
                    println("❌ Jumlah label tidak sesuai!")
                }

                println("\n=== SELESAI ===\n")

            } catch (e: Exception) {
                println("\n❌ Error saat validasi model: ${e.message}")
                e.printStackTrace()
            }
        } ?: println("\n❌ Interpreter tidak tersedia")
    }
}