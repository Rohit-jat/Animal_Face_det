# Dual-Model Animal Emotion Detection System

## Overview

This system now supports **two different algorithms** for animal emotion detection:

1. **YOLOv8 + CNN** - YOLOv8 for face detection + CNN for emotion classification
2. **CNN Only** - CNN-based pipeline for both detection and classification

This dual-model approach allows you to compare both methods for your research paper.

---

## Features

### 🎯 Toggle Switch
- Easy model selection via radio button in the sidebar
- Switch between YOLOv8 and CNN models instantly
- No need to restart the application

### 📊 Performance Metrics
- **Inference Time**: Shows how long each prediction takes
- **FPS (Frames Per Second)**: Estimated processing speed
- Real-time performance comparison

### 🔬 Comparison Mode
- Side-by-side visualization of both models
- Comparative metrics table
- Speed difference analysis
- Perfect for research paper data collection

### 🎨 Visual Indicators
- Color-coded bounding boxes:
  - Green = Happy emotion
  - Red = Sad emotion
- Model labels on each detection
- Clear UI indicators showing active model

---

## How to Use

### Basic Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Select Your Model**
   - In the sidebar, choose between:
     - "YOLOv8 + CNN" (Better accuracy)
     - "CNN Only" (Faster inference)

3. **Upload an Image**
   - Click "Upload Image"
   - Select an animal photo
   - Click "Analyze Emotions"

4. **View Results**
   - See detected faces with emotion labels
   - Check inference time and FPS
   - Compare predictions

### Comparison Mode (For Research)

1. Enable **"Comparison Mode"** checkbox in sidebar
2. Upload and analyze an image
3. View side-by-side results from both models
4. Examine the comparison metrics table
5. Note which model performs better for your use case

---

## Technical Details

### Model Architecture

#### YOLOv8 + CNN Pipeline
- **Detection**: YOLOv8 (Ultralytics)
  - Pre-trained on COCO dataset
  - Detects cats and dogs (classes 15, 16)
  - High accuracy, moderate speed

- **Classification**: MobileNetV2-based CNN
  - Transfer learning from ImageNet
  - Binary classification (Happy/Sad)
  - Optimized for speed and accuracy

#### CNN Only Pipeline
- **Detection**: Haar Cascades + Contour Analysis
  - OpenCV's pre-trained face cascade
  - Fallback to contour detection
  - Fast but less accurate than YOLO

- **Classification**: Same MobileNetV2 CNN
  - Identical emotion classifier
  - Consistent predictions across pipelines

### File Structure

```
game/
├── app.py                 # Main Streamlit application
├── detect.py             # YOLOv8 detection module
├── cnn_detector.py       # NEW: CNN-only detection module
├── train_emotion.py      # CNN model training script
├── test_cnn_module.py    # NEW: CNN module test script
├── models/
│   ├── yolov8.pt        # YOLOv8 weights
│   └── emotion_model.h5 # CNN emotion classifier
└── dataset/
    ├── happy/           # Training images (happy)
    └── sad/             # Training images (sad)
```

---

## Performance Comparison

### Expected Characteristics

| Metric | YOLOv8 + CNN | CNN Only |
|--------|--------------|----------|
| **Accuracy** | High | Moderate |
| **Speed** | Moderate (~0.2-0.5s) | Fast (~0.1-0.3s) |
| **Face Detection** | Precise bounding boxes | Approximate regions |
| **Best For** | Research accuracy | Real-time applications |

### Research Paper Metrics

The system provides these measurable metrics:

1. **Inference Time per Image**
2. **Faces Detected per Image**
3. **Confidence Scores**
4. **Model Agreement Rate** (in comparison mode)

---

## API Reference

### AnimalEmotionClassifier Class

```python
classifier = AnimalEmotionClassifier(
    emotion_model_path='models/emotion_model.h5',
    yolo_model_path='models/yolov8.pt'
)
```

**Methods:**
- `load_models()` - Load all required models
- `process_image(image, model_type='yolo')` - Process image with selected model
  - Returns: `(result_image, results, inference_time)`

### CNNEmotionDetector Class

```python
cnn_detector = CNNEmotionDetector(
    emotion_model_path='models/emotion_model.h5'
)
```

**Methods:**
- `load_models()` - Load CNN and Haar cascade
- `detect_faces(image)` - Detect faces using Haar cascades
- `predict_emotion(face_image)` - Classify emotion
- `detect_and_classify(image)` - Complete pipeline
- `draw_results(image, results, model_type)` - Visualize results

---

## Testing

### Test CNN Module
```bash
python test_cnn_module.py
```

Expected output:
```
✅ CNN emotion model loaded successfully
✅ Haar cascade loaded successfully
✅ CNN Detector Test Complete!
```

### Test Full Application
```bash
streamlit run app.py
```

The app will automatically:
1. Load both YOLOv8 and CNN models
2. Display success messages
3. Show model selection interface

---

## Troubleshooting

### "Emotion model not found"
- Train the model first: `python train_emotion.py`
- Ensure `models/emotion_model.h5` exists

### "No faces detected"
- Try the other model (CNN may work better for some images)
- Ensure animal faces are clearly visible
- Check image resolution and lighting

### Slow inference time
- CNN Only mode is typically faster
- Reduce image resolution if needed
- Close other GPU-intensive applications

---

## Data Collection for Research Paper

### Suggested Methodology

1. **Prepare Test Dataset**
   - Collect 50-100 animal images
   - Label ground truth emotions

2. **Run Both Models**
   - Use comparison mode
   - Record metrics for each image

3. **Measure:**
   - Accuracy (vs ground truth)
   - Inference time
   - False positive rate
   - Confidence score distribution

4. **Analyze:**
   - Which model is more accurate?
   - Speed vs accuracy trade-offs
   - Cases where models disagree

### Export Results

You can manually record results from the comparison table:
- Faces detected count
- Inference times
- Confidence scores
- Model agreement/disagreement

---

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] CSV export of results
- [ ] Webcam real-time comparison
- [ ] Additional emotion classes (Angry, Neutral, etc.)
- [ ] More animal species support
- [ ] Model ensemble approach

---

## Citation

If you use this system in your research:

```
@software{AnimalEmotionDetection2024,
  title = {Dual-Model Animal Emotion Detection System},
  description = {Comparative analysis of YOLOv8 and CNN approaches},
  year = {2024}
}
```

---

## License

This project is for educational and research purposes.
