import cv2
import numpy as np
from ultralytics import YOLO
import os

class AnimalFaceDetector:
    def __init__(self, model_path='models/yolov8.pt'):
        """
        Initialize the animal face detector.
        """
        # For now, we'll use a placeholder - in practice, you would need a model
        # trained specifically for animal face detection
        self.model = None
        self.model_path = model_path
        
    def load_model(self):
        """
        Load the YOLOv8 model for animal face detection.
        """
        # In a real implementation, you would load a custom trained model
        # For this example, we'll use a pre-trained model and filter for animals
        try:
            # Try to load a custom model if it exists
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                # Fallback to a general purpose model
                # Note: This is a placeholder - you would need a model trained for animal faces
                print("Custom model not found. Using general object detection model.")
                self.model = YOLO('yolov8n.pt')  # This requires internet connection to download
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using a basic face detection approach as fallback.")
            self.model = None
    
    def detect_faces(self, image):
        """
        Detect animal faces in an image.
        Returns list of bounding boxes.
        """
        if self.model is not None:
            # Use YOLOv8 for detection
            results = self.model(image)
            boxes = []
            
            # Process results
            for result in results:
                # For animal detection, we need to filter for animal classes
                # COCO dataset classes: 15 = cat, 16 = dog
                animal_classes = [15, 16]  # cat and dog classes in COCO
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for animal classes
                        if class_id in animal_classes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Convert class ID to label
                            class_label = "cat" if class_id == 15 else "dog"
                            
                            boxes.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'label': class_label
                            })
            
            return boxes
        else:
            # Fallback to basic face detection using Haar cascades
            # This is just a placeholder implementation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Note: You would need a specific cascade for animal faces
            # This is just for demonstration
            return []
    
    def crop_face(self, image, bbox):
        """
        Crop the face region from the image using bounding box.
        """
        x1, y1, x2, y2 = bbox
        # Add some padding around the face
        padding = 20
        y1 = max(0, y1 - padding)
        y2 = min(image.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def process_image(self, image_path):
        """
        Process an image to detect animal faces and crop them.
        Returns list of cropped faces and their bounding boxes.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Crop faces
        cropped_faces = []
        for face in faces:
            cropped = self.crop_face(image, face['bbox'])
            cropped_faces.append({
                'image': cropped,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'label': face['label']
            })
        
        return cropped_faces

# Example usage
if __name__ == "__main__":
    detector = AnimalFaceDetector()
    detector.load_model()
    
    # This is just an example - you would need actual images
    # faces = detector.process_image('path_to_image.jpg')
    print("Animal face detector initialized.")