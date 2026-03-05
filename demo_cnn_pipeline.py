"""
Demonstration of the updated CNN-only pipeline for animal emotion detection.
This script shows how the cat face cascade detects animal faces before emotion classification.
"""

from cnn_detector import CNNEmotionDetector
import cv2
import numpy as np

def demo_cnn_pipeline():
    """Demonstrate the CNN-only pipeline with cat face detection."""
    
    print("=" * 60)
    print("CNN-Only Animal Emotion Detection Pipeline Demo")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing CNN detector...")
    detector = CNNEmotionDetector()
    
    # Load models
    print("2. Loading models...")
    detector.load_models()
    
    # Check what was loaded
    if detector.cat_face_cascade is not None:
        print("   ✅ Cat face Haar cascade loaded")
        print(f"      - Using: haarcascade_frontalcatface.xml")
    else:
        print("   ⚠️  Cat cascade not available, using fallback")
    
    if detector.emotion_model is not None:
        print("   ✅ CNN emotion model loaded")
        print(f"      - Input size: 224x224")
        print(f"      - Classes: Happy, Sad")
    else:
        print("   ❌ Emotion model not loaded")
        return
    
    # Test with sample image (if available)
    print("\n3. Pipeline Overview:")
    print("-" * 60)
    print("Step 1: Load image (BGR format from OpenCV)")
    print("Step 2: Convert to grayscale for face detection")
    print("Step 3: Apply cat face Haar cascade detector")
    print("Step 4: For each detected face:")
    print("   a) Crop face region with 20% padding")
    print("   b) Resize to 224x224 pixels")
    print("   c) Normalize pixel values to [0, 1]")
    print("   d) Add batch dimension")
    print("   e) Pass through CNN model")
    print("   f) Extract emotion label and confidence")
    print("Step 5: Draw bounding boxes and labels")
    print("Step 6: Display results")
    print("-" * 60)
    
    # Show detection parameters
    print("\n4. Detection Parameters:")
    print(f"   - Scale factor: 1.1")
    print(f"   - Min neighbors: 5")
    print(f"   - Minimum face size: 30x30 pixels")
    print(f"   - Padding: 20% of face dimensions")
    
    # Example workflow
    print("\n5. Example Workflow:")
    print("-" * 60)
    print("# Load an image")
    print("image = cv2.imread('your_cat_photo.jpg')")
    print("")
    print("# Run detection and classification")
    print("results, total_time = detector.detect_and_classify(image)")
    print("")
    print("# Results include:")
    print("# - bbox: [x1, y1, x2, y2] bounding box")
    print("# - emotion: 'Happy' or 'Sad'")
    print("# - confidence: 0.0 to 1.0")
    print("# - inference_time: time for CNN prediction")
    print("# - label: 'cat' or 'animal'")
    print("-" * 60)
    
    print("\n✅ Pipeline ready for use!")
    print("\nTo test with your own image:")
    print("  python -c \"from cnn_detector import CNNEmotionDetector; d = CNNEmotionDetector(); d.load_models(); import cv2; img = cv2.imread('your_image.jpg'); results, _ = d.detect_and_classify(img); print(results)\"")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        demo_cnn_pipeline()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
