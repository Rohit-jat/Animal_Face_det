import cv2
import numpy as np
import tensorflow as tf
import time
import os

class CNNEmotionDetector:
    """
    CNN-based face detection and emotion classification.
    Uses Haar cascades for face localization and CNN for emotion classification.
    """
    
    def __init__(self, emotion_model_path='models/emotion_model.h5'):
        """
        Initialize the CNN emotion detector.
        """
        self.emotion_model_path = emotion_model_path
        self.emotion_model = None
        self.face_cascade = None
        self.cat_face_cascade = None
        self.emotion_labels = ['Happy', 'Sad']
        
    def load_models(self):
        """
        Load CNN emotion model and Haar cascades for animal face detection.
        """
        # Load Haar cascade for cat face detection (primary)
        try:
            # Try to load OpenCV's pre-trained cat face cascade
            cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
            if os.path.exists(cat_cascade_path):
                self.cat_face_cascade = cv2.CascadeClassifier(cat_cascade_path)
                print("Cat face Haar cascade loaded successfully")
            else:
                print("Cat face cascade not found, trying alternate cascade")
                # Try alternate cat eye cascade
                cat_eye_path = cv2.data.haarcascades + 'haarcascade_catface.xml'
                if os.path.exists(cat_eye_path):
                    self.cat_face_cascade = cv2.CascadeClassifier(cat_eye_path)
                    print("Cat eye Haar cascade loaded successfully")
                else:
                    print("Cat cascades not found, using fallback detection")
                    self.cat_face_cascade = None
        except Exception as e:
            print(f"Error loading cat face cascade: {e}")
            self.cat_face_cascade = None
        
        # Load CNN emotion model
        if os.path.exists(self.emotion_model_path):
            try:
                self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
                print("CNN emotion model loaded successfully")
            except Exception as e:
                print(f"Error loading CNN emotion model: {e}")
                self.emotion_model = None
        else:
            print(f"CNN emotion model not found at {self.emotion_model_path}")
            self.emotion_model = None
    
    def detect_faces(self, image, min_face_size=(30, 30)):
        """
        Detect animal faces using cat face Haar cascade.
        Returns list of bounding boxes.
        """
        faces = []
        
        if self.cat_face_cascade is not None:
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect cat/animal faces using specialized cascade
            detected_faces = self.cat_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process detected faces
            for (x, y, w, h) in detected_faces:
                faces.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.9,  # Default confidence for Haar cascade
                    'label': 'cat'
                })
        else:
            # Fallback: Use simple contour detection optimized for animals
            faces = self._detect_faces_contour(image)
        
        return faces
    
    def _detect_faces_contour(self, image):
        """
        Fallback face detection using contour analysis optimized for animals.
        This is used when Haar cascade is not available.
        """
        faces = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be animal faces
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 80000:  # Adjusted thresholds for animal faces
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Animal faces are typically rounder/oval
                # Check if aspect ratio is roughly face-like (0.6 to 1.4)
                if 0.6 < aspect_ratio < 1.4:
                    # Additional check: circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        # Animal faces tend to have moderate circularity
                        if 0.3 < circularity < 0.9:
                            faces.append({
                                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                                'confidence': 0.7,
                                'label': 'animal'
                            })
        
        return faces
    
    def preprocess_image(self, image):
        """
        Preprocess face image for CNN emotion classification.
        Ensures proper resizing and normalization for the model.
        """
        # Resize to model input size (224x224)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Ensure 3 channels (BGR or RGB)
        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # RGBA to BGR
            image = image[:, :, :3]
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_emotion(self, face_image):
        """
        Predict emotion for a face image using CNN.
        Returns emotion label and confidence.
        """
        if self.emotion_model is None:
            return "Model Not Loaded", 0.0
        
        # Preprocess image
        processed_image = self.preprocess_image(face_image)
        
        # Predict with timing
        start_time = time.time()
        predictions = self.emotion_model.predict(processed_image, verbose=0)
        inference_time = time.time() - start_time
        
        # Get prediction
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        return self.emotion_labels[predicted_class], confidence, inference_time
    
    def crop_face(self, image, bbox, padding=20):
        """
        Crop face region with appropriate padding for animal faces.
        Ensures we capture the full face area for emotion analysis.
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Add proportional padding (20% of face size)
        padding_x = int(face_width * 0.2)
        padding_y = int(face_height * 0.2)
        
        # Apply padding with boundary checks
        y1_padded = max(0, y1 - padding_y)
        y2_padded = min(image.shape[0], y2 + padding_y)
        x1_padded = max(0, x1 - padding_x)
        x2_padded = min(image.shape[1], x2 + padding_x)
        
        return image[y1_padded:y2_padded, x1_padded:x2_padded]
    
    def detect_and_classify(self, image):
        """
        Complete pipeline: detect animal faces and classify emotions.
        For each detected face:
        1. Crop the face region with padding
        2. Resize to CNN input size (224x224)
        3. Normalize pixel values
        4. Pass through CNN for emotion prediction
        Returns results with bounding boxes, emotions, and timing info.
        """
        start_time = time.time()
        
        # Detect animal faces
        faces = self.detect_faces(image)
        
        # Classify emotions for each detected face
        results = []
        total_inference_time = 0
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            
            # Crop face region with padding
            face_img = self.crop_face(image, face['bbox'])
            
            # Skip if face is too small after cropping
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue
            
            # Predict emotion (includes preprocessing inside)
            emotion, confidence, emotion_time = self.predict_emotion(face_img)
            total_inference_time += emotion_time
            
            results.append({
                'bbox': face['bbox'],
                'emotion': emotion,
                'confidence': float(confidence),
                'inference_time': emotion_time,
                'label': face.get('label', 'cat')
            })
        
        total_time = time.time() - start_time
        
        return results, total_time
    
    def draw_results(self, image, results, model_type="CNN"):
        """
        Draw bounding boxes and emotion labels on image.
        Shows animal type, emotion, and confidence for each detection.
        """
        output_image = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            label_text = result.get('label', 'cat')  # Get animal label
            
            # Color based on emotion
            color = (0, 255, 0) if emotion == 'Happy' else (0, 0, 255)
            
            # Draw bounding box with thicker line for visibility
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with animal type and emotion
            label = f"{model_type} - {label_text.capitalize()} - {emotion}: {confidence:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return output_image


# Example usage
if __name__ == "__main__":
    detector = CNNEmotionDetector()
    detector.load_models()
    print("CNN emotion detector initialized.")
