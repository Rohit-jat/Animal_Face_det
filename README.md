# Animal Face Emotion Classifier

This application detects animal faces (dogs and cats) in images and classifies their facial expressions as "Happy" or "Sad" using AI.

## Features

- Animal face detection using YOLOv8
- Emotion classification using transfer learning (MobileNetV2)
- Streamlit web interface for easy interaction
- Support for image upload
- Visualization of results with bounding boxes

## Directory Structure

```
├── dataset/
│   ├── happy/
│   └── sad/
├── models/
│   ├── yolov8.pt
│   └── emotion_model.h5
├── app.py
├── train_emotion.py
├── detect.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Dataset

Organize your dataset in the following structure:
```
dataset/
├── happy/
│   ├── happy_animal_1.jpg
│   ├── happy_animal_2.jpg
│   └── ...
└── sad/
    ├── sad_animal_1.jpg
│   ├── sad_animal_2.jpg
    └── ...
```

You can use the `create_dataset.py` script to help organize your images:
```
python create_dataset.py
```

### 2. Download/Prepare Models

Download a pre-trained YOLOv8 model or prepare a custom animal face detection model:
```
python download_models.py
```

### 3. Train the Emotion Classifier

Train the emotion classification model:
```
python train_emotion.py
```

This will:
- Create and train a MobileNetV2-based model
- Save the trained model as `models/emotion_model.h5`
- Generate training history plots

### 4. Run the Application

Start the Streamlit web application:
```
streamlit run app.py
```

Then:
1. Upload an image containing animal faces
2. Click "Analyze Emotions"
3. View the detected animals with their emotion classifications

## How It Works

1. **Animal Face Detection**: Uses YOLOv8 to detect animal faces in images
2. **Face Cropping**: Extracts detected face regions from the image
3. **Emotion Classification**: Uses a transfer learning model based on MobileNetV2 to classify emotions as "Happy" or "Sad"
4. **Visualization**: Displays results with bounding boxes and emotion labels

## Requirements

- Python 3.7+
- See `requirements.txt` for detailed package requirements

## Notes

- For best results, use clear images with visible animal faces
- The emotion classification model needs to be trained before use
- The YOLOv8 model for animal face detection should be specifically trained for this task
- For production use, consider training a custom model specifically for animal face detection