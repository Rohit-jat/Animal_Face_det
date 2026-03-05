import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

# Import our modules
from detect import AnimalFaceDetector
from cnn_detector import CNNEmotionDetector

# Emotion labels
EMOTION_LABELS = ['Happy', 'Sad']

class AnimalEmotionClassifier:
    def __init__(self, emotion_model_path='models/emotion_model.h5', yolo_model_path='models/yolov8.pt'):
        """
        Initialize the animal emotion classifier with both YOLO and CNN models.
        """
        self.emotion_model_path = emotion_model_path
        self.yolo_model_path = yolo_model_path
        self.emotion_model = None
        self.yolo_detector = AnimalFaceDetector(yolo_model_path)
        self.cnn_detector = CNNEmotionDetector(emotion_model_path)
        self.current_model_type = 'yolo'  # Default to YOLO
        
    def load_models(self):
        """
        Load both detection and emotion classification models.
        """
        # Load emotion classification model (shared by both pipelines)
        if os.path.exists(self.emotion_model_path):
            try:
                self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
            except Exception as e:
                st.error(f"Error loading emotion model: {e}")
                st.info("Please train the emotion model first using train_emotion.py")
        else:
            st.warning("Emotion model not found. Please train the model first.")
            st.info("Expected path: " + self.emotion_model_path)
        
        # Load YOLO detector
        try:
            self.yolo_detector.load_model()
        except Exception as e:
            st.error(f"Error loading YOLO detection model: {e}")
        
        # Load CNN detector
        try:
            self.cnn_detector.load_models()
        except Exception as e:
            st.error(f"Error loading CNN detector: {e}")
    
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
    
    def predict_emotion_cnn(self, face_image):
        """
        Predict emotion for a face image using CNN.
        Returns emotion label and confidence.
        """
        if self.emotion_model is None:
            return "Model Not Loaded", 0.0
            
        # Preprocess image
        processed_image = self.preprocess_image(face_image)
        
        # Predict
        predictions = self.emotion_model.predict(processed_image, verbose=0)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        return EMOTION_LABELS[predicted_class], confidence
    
    def process_image_yolo(self, image):
        """
        Process image using YOLOv8 detection + CNN emotion classification.
        """
        start_time = time.time()
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces using YOLO
        faces = self.yolo_detector.detect_faces(image_cv)
        
        # Process each detected face
        results = []
        for face in faces:
            # Crop face
            x1, y1, x2, y2 = face['bbox']
            face_img = image_cv[y1:y2, x1:x2]
            
            # Skip if face is too small
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue
            
            # Predict emotion using CNN
            emotion, confidence = self.predict_emotion_cnn(face_img)
            
            # Store results
            results.append({
                'bbox': face['bbox'],
                'animal': face.get('label', 'Animal'),
                'emotion': emotion,
                'confidence': float(confidence)
            })
            
            # Draw bounding box and label on image
            color = (0, 255, 0) if emotion == 'Happy' else (0, 0, 255)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            label = f"YOLO - {face.get('label', 'Animal')} - {emotion}: {confidence:.2f}"
            cv2.putText(image_cv, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        inference_time = time.time() - start_time
        
        # Convert back to PIL format for Streamlit
        result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return result_image, results, inference_time
    
    def process_image_cnn(self, image):
        """
        Process image using CNN-only pipeline.
        """
        start_time = time.time()
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Use CNN detector for both detection and classification
        results, _ = self.cnn_detector.detect_and_classify(image_cv)
        
        # Add animal label to results
        for result in results:
            result['animal'] = 'Animal'
        
        # Draw results on image
        result_image_array = self.cnn_detector.draw_results(image_cv, results, model_type="CNN")
        result_image = cv2.cvtColor(result_image_array, cv2.COLOR_BGR2RGB)
        
        inference_time = time.time() - start_time
        return result_image, results, inference_time
    
    def process_image(self, image, model_type='yolo'):
        """
        Process an image based on selected model type.
        Routes to appropriate pipeline.
        """
        if model_type == 'cnn':
            return self.process_image_cnn(image)
        else:  # default to yolo
            return self.process_image_yolo(image)

def main():
    st.title("Animal Face Emotion Classifier")
    st.write("Detect animal faces and classify their emotions as Happy or Sad")
    
    # Initialize classifier
    classifier = AnimalEmotionClassifier()
    
    # Load models
    with st.spinner("Loading models..."):
        classifier.load_models()
    
    # Show model loaded status in bottom-right corner
    st.markdown(
        """
        <style>
        .model-status {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: rgba(40, 167, 69, 0.9);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 9999;
            max-width: 300px;
        }
        </style>
        <div class="model-status">
            Models loaded successfully
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model selection in center of page
    st.header("Model Selection")
    st.write("Choose which model to use for emotion detection:")
    
    # Create a centered toggle using columns with labels
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Add YOLO and CNN labels on sides of toggle
        label_col1, label_col2, label_col3 = st.columns([1, 3, 1])
        with label_col1:
            st.markdown("<div style='text-align: right; padding-right: 10px; font-weight: bold;'>YOLOv8</div>", unsafe_allow_html=True)
        with label_col2:
            model_choice = st.toggle(
                "",
                value=False,
                help="Toggle OFF for YOLOv8+CNN (More Accurate) | Toggle ON for CNN Only (Faster)"
            )
        with label_col3:
            st.markdown("<div style='text-align: left; padding-left: 10px; font-weight: bold;'>CNN</div>", unsafe_allow_html=True)
    
    # Display current model info prominently
    if model_choice:
        st.info("CNN Only - Using CNN pipeline for both detection and classification (Faster)")
        model_type = 'cnn'
    else:
        st.success("YOLOv8 + CNN - Using YOLOv8 for detection + CNN for emotion classification (More Accurate)")
        model_type = 'yolo'
    
    # Sidebar
    st.sidebar.header("Options")
    
    # Add comparison mode option
    comparison_mode = st.sidebar.checkbox(
        "Comparison Mode (Side-by-Side)",
        help="Display results from both models simultaneously for comparison"
    )
    
    # Show instructions directly in sidebar (no expander)
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to Use")
    st.sidebar.markdown(
        """
        **1. Select Model**
        - Use the toggle button to choose your model:
          - **OFF (Left)**: YOLOv8 + CNN (More accurate)
          - **ON (Right)**: CNN Only (Faster processing)
        
        **2. Upload an Image**
        - Click on 'Upload Image' option
        - Choose a photo containing an animal's face
        
        **3. Analyze the Image**
        - Click on 'Analyze Emotions' button 
        
        **4. View Results**
        - See detected animal face(s) highlighted
        - Each face shows predicted emotion (Happy/Sad) with confidence
        - Inference time and FPS metrics displayed below
        
        **5. Comparison Mode (Optional)**
        - Enable 'Comparison Mode' checkbox
        - See both models' results side-by-side
        - Compare accuracy and speed 
        
        **6. Try Another Image**
        - Upload a new image to test again
        """
    )
    
    app_mode = st.sidebar.selectbox(
        "Choose the mode",
        ["Upload Image", "Webcam (Coming Soon)"]
    )
    
    if app_mode == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image
            if st.button("Analyze Emotions"):
                with st.spinner("Analyzing..."):
                    try:
                        if comparison_mode:
                            # Run both models for comparison
                            st.subheader("🔄 Running Both Models for Comparison...")
                            
                            # YOLO model
                            result_image_yolo, results_yolo, time_yolo = classifier.process_image(image, model_type='yolo')
                            
                            # CNN model
                            result_image_cnn, results_cnn, time_cnn = classifier.process_image(image, model_type='cnn')
                            
                            # Display side-by-side comparison
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(result_image_yolo, caption="YOLOv8 + CNN Model", use_container_width=True)
                                st.metric("YOLO Inference Time", f"{time_yolo:.3f}s")
                                if results_yolo:
                                    st.success(f"✅ YOLO detected {len(results_yolo)} face(s)")
                                    for i, res in enumerate(results_yolo):
                                        st.write(f"- {res['animal'].capitalize()}: {res['emotion']} ({res['confidence']:.2f})")
                            
                            with col2:
                                st.image(result_image_cnn, caption="CNN Only Model", use_container_width=True)
                                st.metric("CNN Inference Time", f"{time_cnn:.3f}s")
                                if results_cnn:
                                    st.success(f"✅ CNN detected {len(results_cnn)} face(s)")
                                    for i, res in enumerate(results_cnn):
                                        st.write(f"- Animal: {res['emotion']} ({res['confidence']:.2f})")
                            
                            # Comparison metrics table
                            st.subheader("📊 Model Comparison Metrics")
                            comparison_data = {
                                'Model': ['YOLOv8 + CNN', 'CNN Only'],
                                'Faces Detected': [len(results_yolo), len(results_cnn)],
                                'Inference Time': [f"{time_yolo:.3f}s", f"{time_cnn:.3f}s"],
                                'FPS (Est.)': [f"{1.0/time_yolo if time_yolo > 0 else 0:.1f}", f"{1.0/time_cnn if time_cnn > 0 else 0:.1f}"]
                            }
                            st.table(comparison_data)
                            
                            # Speed comparison
                            speed_diff = abs(time_yolo - time_cnn)
                            faster_model = "CNN Only" if time_cnn < time_yolo else "YOLOv8 + CNN"
                            st.info(f"⚡ **Speed Difference:** {speed_diff:.3f}s | Faster model: {faster_model}")
                            
                        else:
                            # Single model mode (existing code)
                            result_image, results, inference_time = classifier.process_image(image, model_type=model_type)
                            
                            # Display processed image
                            st.image(result_image, caption=f"Processed Image ({model_choice})", use_container_width=True)
                            
                            # Display performance metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Inference Time", f"{inference_time:.3f}s")
                            with col2:
                                fps = 1.0 / inference_time if inference_time > 0 else 0
                                st.metric("FPS (Estimated)", f"{fps:.1f}")
                            
                            # Display emotion results
                            if results:
                                st.subheader("Detected Animals and Emotions:")
                                for i, result in enumerate(results):
                                    animal = result['animal']
                                    emotion = result['emotion']
                                    confidence = result['confidence']
                                    st.write(f"**Detection {i+1}:** {animal.capitalize()} - {emotion} (Confidence: {confidence:.2f})")
                                
                                # Additional model-specific info
                                st.sidebar.success(f"✅ Detected {len(results)} face(s) using {model_choice}")
                            else:
                                st.warning("No animal faces detected in the image.")
                                st.sidebar.info("💡 Try adjusting the image or using a different model")
                            
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
    
    elif app_mode == "Webcam (Coming Soon)":
        st.header("Webcam Mode")
        st.info("Webcam functionality will be implemented in a future version.")
        st.write("For now, please use the 'Upload Image' option.")

if __name__ == "__main__":
    main()