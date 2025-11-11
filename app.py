import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Import our modules
from detect import AnimalFaceDetector

# Emotion labels
EMOTION_LABELS = ['Happy', 'Sad']

class AnimalEmotionClassifier:
    def __init__(self, emotion_model_path='models/emotion_model.h5'):
        """
        Initialize the animal emotion classifier.
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
                st.success("Emotion classification model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading emotion model: {e}")
                st.info("Please train the emotion model first using train_emotion.py")
        else:
            st.warning("Emotion model not found. Please train the model first.")
            st.info("Expected path: " + self.emotion_model_path)
        
        # Load detection model
        try:
            self.detector.load_model()
            st.success("Animal face detection model loaded!")
        except Exception as e:
            st.error(f"Error loading detection model: {e}")
    
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
    
    def process_image(self, image):
        """
        Process an image: detect faces, classify emotions, draw results.
        """
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.detector.detect_faces(image_cv)
        
        # Process each detected face
        results = []
        for face in faces:
            # Crop face
            x1, y1, x2, y2 = face['bbox']
            face_img = image_cv[y1:y2, x1:x2]
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_img)
            
            # Store results
            results.append({
                'bbox': face['bbox'],
                'animal': face.get('label', 'Animal'),
                'emotion': emotion,
                'confidence': confidence
            })
            
            # Draw bounding box and label on image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{face.get('label', 'Animal')} - {emotion}: {confidence:.2f}"
            cv2.putText(image_cv, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert back to PIL format for Streamlit
        result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return result_image, results

def main():
    st.title("Animal Face Emotion Classifier")
    st.write("Detect animal faces and classify their emotions as Happy or Sad")
    
    # Initialize classifier
    classifier = AnimalEmotionClassifier()
    
    # Load models
    with st.spinner("Loading models..."):
        classifier.load_models()
    
    # Sidebar
    st.sidebar.header("Options")
    app_mode = st.sidebar.selectbox(
        "Choose the mode",
        ["Upload Image", "Webcam (Coming Soon)", "Instructions"]
    )
    
    if app_mode == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            if st.button("Analyze Emotions"):
                with st.spinner("Analyzing..."):
                    try:
                        result_image, results = classifier.process_image(image)
                        
                        # Display results
                        st.image(result_image, caption="Processed Image", use_column_width=True)
                        
                        # Display emotion results
                        if results:
                            st.subheader("Detected Animals and Emotions:")
                            for i, result in enumerate(results):
                                animal = result['animal']
                                emotion = result['emotion']
                                confidence = result['confidence']
                                st.write(f"{animal.capitalize()}: {emotion} (Confidence: {confidence:.2f})")
                        else:
                            st.warning("No animal faces detected in the image.")
                            
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
    
    elif app_mode == "Webcam (Coming Soon)":
        st.header("Webcam Mode")
        st.info("Webcam functionality will be implemented in a future version.")
        st.write("For now, please use the 'Upload Image' option.")
        
    # elif app_mode == "Instructions":
    #     st.header("How to Use This Application")
        
    #     st.subheader("1. Setup")
    #     st.write("""
    #     1. Install required packages: `pip install -r requirements.txt`
    #     2. Prepare your dataset of animal face images organized in:
    #        ```
    #        dataset/
    #        ├── happy/
    #        └── sad/
    #        ```
    #     """)
        
    #     st.subheader("2. Train the Emotion Classifier")
    #     st.write("""
    #     Run the training script to create your emotion classification model:
    #     ```
    #     python train_emotion.py
    #     ```
    #     This will create `models/emotion_model.h5` which is used for emotion prediction.
    #     """)
        
    #     st.subheader("3. Run the Application")
    #     st.write("""
    #     Start the Streamlit web application:
    #     ```
    #     streamlit run app.py
    #     ```
    #     """)
        
    #     st.subheader("4. Using the Application")
    #     st.write("""
    #     1. Select "Upload Image" from the sidebar
    #     2. Upload an image containing animal faces
    #     3. Click "Analyze Emotions"
    #     4. View the detected animals with their emotion classifications
    #     """)

if __name__ == "__main__":
    main()