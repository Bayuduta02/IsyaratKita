# IsyaratKita - Aplikasi Deteksi Bahasa Isyarat dengan YOLOv11

Aplikasi Android untuk mendeteksi bahasa isyarat Indonesia menggunakan model deep learning YOLOv11.

## Fitur

- Deteksi bahasa isyarat secara real-time menggunakan kamera
- Visualisasi bounding box dan label hasil deteksi
- Dukungan untuk model YOLOv11 (TensorFlow Lite)
- Penanganan orientasi layar otomatis
- Switching kamera depan/belakang

## Setup Model

1. Pastikan model YOLOv11 sudah dikonversi ke format TensorFlow Lite (.tflite)
2. Tempatkan file model di `app/src/main/assets/best_float32.tflite`
3. Pastikan file label ada di `app/src/main/assets/labels.txt` dengan format satu label per baris

## Konfigurasi Model

Untuk menyesuaikan konfigurasi YOLOv11, buka file `YOLOv11Config.kt`:

```kotlin
object YOLOv11Config {
    // Model properties
    const val MODEL_FILE = "best_float32.tflite"
    const val LABELS_FILE = "labels.txt"
    
    // Input properties
    const val INPUT_SIZE = 640
    const val CHANNELS = 3
    
    // Detection properties
    const val MAX_DETECTIONS = 10
    const val CONFIDENCE_THRESHOLD = 0.5f
    const val NMS_THRESHOLD = 0.5f
    
    // Output shape
    const val DETECTION_COUNT = 8400
}
```

## Penggunaan

1. Jalankan aplikasi
2. Dari home screen, pilih menu "Kamera"
3. Arahkan kamera ke bahasa isyarat yang ingin dideteksi
4. Hasil deteksi akan ditampilkan secara real-time

## Troubleshooting

Jika deteksi tidak berjalan dengan baik:
- Pastikan model dan labels.txt sudah benar
- Coba sesuaikan nilai threshold di YOLOv11Config.kt
- Pastikan pencahayaan cukup terang
- Pastikan gesture bahasa isyarat berada dalam frame kamera

## Konversi Model

Untuk mengkonversi model YOLOv11 ke format TFLite:

```python
import tensorflow as tf

# Load model YOLOv11 (dalam format .pt atau .onnx)
# Konversi ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

# Simpan model
with open('best_float32.tflite', 'wb') as f:
    f.write(tflite_model)
``` 