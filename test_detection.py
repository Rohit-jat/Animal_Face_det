import cv2
import numpy as np
from detect import AnimalFaceDetector

def test_detection():
    """
    Test the animal face detection module.
    """
    print("Testing Animal Face Detection Module")
    print("=" * 40)
    
    # Initialize detector
    detector = AnimalFaceDetector()
    
    # Load model
    try:
        detector.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create a simple test image (just a blank image for now)
    # In a real scenario, you would load an actual image with animals
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [255, 255, 255]  # White background
    
    # Add a simple shape to simulate an animal
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 0), -1)
    
    # Try to detect faces
    try:
        faces = detector.detect_faces(test_image)
        print(f"✓ Detection completed. Found {len(faces)} faces")
        
        # Print results
        for i, face in enumerate(faces):
            print(f"  Face {i+1}:")
            print(f"    Bounding box: {face['bbox']}")
            print(f"    Confidence: {face['confidence']:.2f}")
            print(f"    Label: {face.get('label', 'Unknown')}")
            
    except Exception as e:
        print(f"✗ Error during detection: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_detection()