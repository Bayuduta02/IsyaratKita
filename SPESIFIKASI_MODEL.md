# 📋 SPESIFIKASI MODEL - IsyaratKita

## Model Information (dari metadata)

```yaml
Task: detect
Framework: Ultralytics YOLOv8
Version: 8.3.204
Date: 2025-10-05
File: model_test 2.tflite (3.127 KB / 3127 bytes)
```

## Input Specification

| Parameter | Value |
|-----------|-------|
| **Input Size** | 320 x 320 |
| **Channels** | 3 (RGB) |
| **Format** | int8 quantized |
| **Data Type** | UINT8 [0-255] |
| **Normalization** | None (raw pixel values) |
| **Batch Size** | 1 |

## Output Specification

| Parameter | Value |
|-----------|-------|
| **Output Shape** | [1, 30, 2100] |
| **Format** | [batch, features, boxes] |
| **Features** | 30 = 4 (bbox) + 26 (classes) |
| **Boxes** | 2100 anchors untuk 320x320 |
| **Bbox Format** | [x_center, y_center, width, height] |
| **Coordinates** | Normalized [0-1] |

## Classes (26 huruf A-Z)

```
0: a    7: h    14: o    21: v
1: b    8: i    15: p    22: w
2: c    9: j    16: q    23: x
3: d    10: k   17: r    24: y
4: e    11: l   18: s    25: z
5: f    12: m   19: t
6: g    13: n   20: u
```

## Detection Configuration

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| **NMS** | false | Manual NMS required (sudah diimplementasi) |
| **Confidence Threshold** | 0.25 | Min. confidence untuk deteksi |
| **IoU Threshold** | 0.45 | Threshold untuk NMS |
| **Stride** | 32 | Feature extraction stride |
| **half** | false | No FP16 |
| **int8** | true | INT8 quantization |

## Performance Optimization

### Hardware Acceleration

1. **GPU Delegate** (dicoba dulu)
   - Biasanya lambat untuk int8
   - Fallback to CPU jika gagal

2. **CPU with XNNPACK** (optimal)
   - Best untuk int8 quantized
   - Multi-threaded (max 4 threads)

### Frame Processing

- **Frame Skip**: 2 (process every 3rd frame)
- **Async Processing**: Ya (dedicated inference thread)
- **Expected FPS**: 15-25 fps
- **Expected Inference Time**: 30-60ms

## File Structure

```
app/src/main/assets/
├── model.tflite          # YOLOv8 int8 320x320 (3.127 KB)
└── labels.txt            # 26 baris (a-z)
```

## Coordinate Transformation

### Flow dari Camera ke Display:

1. **Camera Preview** (contoh: 1920x1080)
   ↓ YUV → RGB conversion

2. **RGB Bitmap** (1920x1080)
   ↓ Scale down

3. **Model Input** (320x320 UINT8)
   ↓ Inference

4. **Model Output** (normalized coords [0-1])
   ↓ Scale to pixel coords

5. **Detection Boxes** (320x320 pixel coords)
   ↓ Scale to preview size

6. **Display Boxes** (1920x1080 pixel coords)
   ↓ Scale to view size

7. **Screen Display** (view width x height)

### Scaling Formula:

```kotlin
// Model space (320x320) → Preview space (previewWidth x previewHeight)
val scaleX = previewSize.width.toFloat() / 320
val scaleY = previewSize.height.toFloat() / 320

// Preview space → View space (automatic via OverlayView)
val viewScaleX = viewWidth.toFloat() / 320
val viewScaleY = viewHeight.toFloat() / 320
```

## Expected Log Output

### Saat Startup (Model Loading):

```
I/Model: ✓ Model loaded: 3KB
D/Model: Attempting to use GPU Delegate.
W/Model: GPU delegate failed: ... Using CPU with XNNPACK.
I/Model: ✓ Model initialized with CPU (XNNPACK enabled).
I/Model: Model ready: 320x320, channels=3, format=NHWC, dataType=UINT8
I/KameraGestureActivity: ✓ Model initialized successfully with input size: 320x320
```

### Saat Inference (Deteksi):

```
I/Model: → Output shape: [1, 30, 2100] -> 26 classes, 2100 boxes
D/Model: First detection: x=0.512, y=0.345, w=0.123, h=0.156 ->
         x1=131.0, y1=78.0, x2=170.0, y2=128.0, score=0.87, class=a
D/Model: Candidates found: 3 (before NMS)
D/Model: ⏱ Prep:3ms | Inf:35ms | Post:5ms | Total:43ms | Det:2
```

### Jika Tidak Ada Deteksi:

```
D/Model: ⚠️ No detections above confidence threshold (0.25)
```

## Troubleshooting

### Model Size Check:

```bash
# Cek ukuran file
ls -lh app/src/main/assets/model.tflite

# Output yang benar:
# -rw-r--r-- 1 user user 3.1K Oct 5 12:47 model.tflite
```

### File Integrity:

```bash
# Cek apakah file valid TFLite
file app/src/main/assets/model.tflite

# Output yang benar:
# model.tflite: TensorFlow Lite model
```

### Common Issues:

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| Model crash saat load | File tidak valid | Upload model yang benar (3.127 KB) |
| Tidak ada deteksi | Threshold terlalu tinggi | Turunkan CONF_THRESHOLD ke 0.15 |
| Bounding box salah | Scaling tidak konsisten | Semua sudah 320x320 ✓ |
| FPS rendah | GPU delegate slow | Sudah fallback ke CPU XNNPACK ✓ |
| Output shape error | Model tidak sesuai | Cek metadata, harus 320x320 |

## Testing Steps

1. ✅ Upload file `model_test 2.tflite` → `model.tflite`
2. ✅ Verify file size: 3127 bytes
3. ✅ Build: `./gradlew clean assembleDebug`
4. ✅ Install dan jalankan aplikasi
5. ✅ Monitor logcat: `adb logcat | grep Model`
6. ✅ Test gesture di depan kamera
7. ✅ Verify bounding box muncul dan akurat

## Expected Results

### Dengan file model yang benar:

- ✅ Model load dalam <1 detik
- ✅ Output shape: [1, 30, 2100]
- ✅ Inference time: 30-60ms
- ✅ FPS: 15-25
- ✅ Bounding box akurat
- ✅ Label benar (a-z)
- ✅ Confidence score >0.25

---

**Status:** ✅ Kode sudah benar untuk model 320x320 int8 quantized
**Action Required:** Upload file model yang valid (3.127 KB)
**Last Updated:** 2025-10-05
