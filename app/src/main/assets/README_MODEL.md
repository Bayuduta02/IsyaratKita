# TFLite Model Setup Instructions

## Problem
Your app currently has a placeholder `model.tflite` file (20 bytes) that needs to be replaced with an actual trained model.

## Required Model Specifications

Your TFLite model must meet these requirements:

1. **Input Shape**: 640x640x3 (width x height x channels)
2. **Format**: YOLOv8 or similar object detection model exported to TFLite
3. **Output Format**: One of:
   - `[1, max_detections, 6]` - (x1, y1, x2, y2, confidence, class_index)
   - `[1, boxes, features]` - (x_center, y_center, width, height, class_scores...)

## Steps to Add Your Model

### Option 1: Using Android Studio
1. Train or obtain your gesture recognition model
2. Export it to TensorFlow Lite format (`.tflite`)
3. In Android Studio:
   - Navigate to `app/src/main/assets/`
   - Delete the existing `model.tflite` placeholder
   - Drag and drop your trained model file (must be named `model.tflite`)
4. Rebuild the project

### Option 2: Using Command Line
```bash
# From project root
cd app/src/main/assets/
rm model.tflite
cp /path/to/your/trained_model.tflite model.tflite
```

## Updating Labels

Update `labels.txt` to match your model's classes:
```bash
# Example: If your model detects sign language gestures
cd app/src/main/assets/
nano labels.txt
```

Replace with your actual class names (one per line):
```
hello
goodbye
yes
no
please
thank_you
...
```

## Training Your Own Model

If you need to train a gesture detection model:

### Using Google Teachable Machine (Easy)
1. Go to https://teachablemachine.withgoogle.com/
2. Choose "Image Project"
3. Collect gesture images for each class
4. Train the model
5. Export as "TensorFlow Lite" (not "TensorFlow.js")
6. Download and use the `.tflite` file

### Using YOLOv8 (Advanced)
```bash
# Install ultralytics
pip install ultralytics

# Train YOLOv8 model
yolo detect train data=gestures.yaml model=yolov8n.pt epochs=100

# Export to TFLite
yolo export model=best.pt format=tflite imgsz=640
```

## Verification

After adding your model, run the app and check logcat:
```bash
adb logcat | grep -E "(Model|TFLite)"
```

You should see:
- ✅ "Model buffer loaded: XXXXX bytes" (where XXXXX > 1000000)
- ✅ "Model initialized successfully"
- ✅ "Input=640x640"

If you see errors, the model file is still invalid.

## Expected Model Size

A typical gesture detection model should be:
- **Minimum**: 1 MB (simple models)
- **Typical**: 5-20 MB (balanced models)
- **Large**: 20-100 MB (complex models)

Your current placeholder is only 20 bytes, which confirms it needs replacement.

## Troubleshooting

### Error: "Model file is too small"
- Your model file is a placeholder. Replace it with an actual trained model.

### Error: "Model file not found"
- Make sure the file is named exactly `model.tflite`
- Check it's in the correct folder: `app/src/main/assets/`

### Error: "Unsupported input tensor shape"
- Your model's input size doesn't match. Retrain or export with 640x640 input size.

### App shows "Mendeteksi..." but no detections
- Labels might not match model outputs
- Model confidence threshold might be too high (currently 0.30)
- Model might not be properly trained

## Need Help?

Check the logcat output for detailed error messages:
```bash
adb logcat -s Model YoloModelBinding KameraGestureActivity
```
