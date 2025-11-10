import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from detect import AnimalFaceDetector

# Emotion labels
EMOTION_LABELS = ['Happy', 'Sad']

class BatchProcessor:
    def __init__(self, emotion_model_path='models/emotion_model.h5'):
        """
        Initialize the batch processor.
        """
        self.emotion_model_path = emotion_model_path
        self.emotion_model = None
        self.detector = AnimalFaceDetector()
        
    def load_models(self):
        """
        Load both detection and emotion classification models.
        """
        # Load emotion classification model
        if os.path.exists(self.emotion_model_path):
            try:
                self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
                print("Emotion classification model loaded successfully!")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                return False
        else:
            print("Emotion model not found. Please train the model first.")
            return False
        
        # Load detection model
        try:
            self.detector.load_model()
            print("Animal face detection model loaded!")
        except Exception as e:
            print(f"Error loading detection model: {e}")
            return False
            
        return True
    
    def preprocess_image(self, image):
        """
        Preprocess image for emotion classification.
        """
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict_emotion(self, face_image):
        """
        Predict emotion for a face image.
        Returns emotion label and confidence.
        """
        if self.emotion_model is None:
            return "Model Not Loaded", 0.0
            
        # Preprocess image
        processed_image = self.preprocess_image(face_image)
        
        # Predict
        predictions = self.emotion_model.predict(processed_image)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        return EMOTION_LABELS[predicted_class], confidence
    
    def process_image(self, image_path):
        """
        Process a single image and return results.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        
        # Process each detected face
        results = []
        for face in faces:
            # Crop face
            x1, y1, x2, y2 = face['bbox']
            face_img = image[y1:y2, x1:x2]
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_img)
            
            # Store results
            results.append({
                'animal': face.get('label', 'Animal'),
                'emotion': emotion,
                'confidence': confidence,
                'bbox': face['bbox']
            })
            
            # Draw bounding box and label on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{face.get('label', 'Animal')} - {emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, results
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a directory.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png']
        
        # Process each image
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in extensions):
                input_path = os.path.join(input_dir, filename)
                print(f"Processing {filename}...")
                
                # Process image
                result = self.process_image(input_path)
                if result is not None:
                    processed_image, results = result
                    
                    # Save processed image
                    output_path = os.path.join(output_dir, f"processed_{filename}")
                    cv2.imwrite(output_path, processed_image)
                    
                    # Print results
                    print(f"  Results for {filename}:")
                    for i, r in enumerate(results):
                        print(f"    {r['animal']}: {r['emotion']} ({r['confidence']:.2f})")
                    
                    if not results:
                        print(f"    No animal faces detected")
                else:
                    print(f"  Failed to process {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process animal face emotion detection')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed images')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Load models
    if not processor.load_models():
        print("Failed to load models. Exiting.")
        exit(1)
    
    # Process directory
    processor.process_directory(args.input, args.output)
    print("Batch processing complete!")