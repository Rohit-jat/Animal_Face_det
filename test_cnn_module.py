"""
Test script for CNN-only emotion detector module.
This verifies that the CNN detector can be loaded and initialized properly.
"""

from cnn_detector import CNNEmotionDetector
import cv2
import numpy as np

def test_cnn_detector():
    """Test the CNN emotion detector initialization and basic functionality."""
    
    print("=" * 50)
    print("Testing CNN Emotion Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = CNNEmotionDetector()
    
    # Load models
    print("\n1. Loading models...")
    detector.load_models()
    
    if detector.emotion_model is None:
        print("❌ CNN emotion model not loaded. Please train the model first.")
        return False
    else:
        print("✅ CNN emotion model loaded successfully")
    
    if detector.face_cascade is None:
        print("⚠️  Haar cascade not available, will use contour detection fallback")
    else:
        print("✅ Haar cascade loaded successfully")
    
    # Test with a dummy image
    print("\n2. Testing face detection on dummy image...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    faces = detector.detect_faces(dummy_image)
    print(f"   Detected {len(faces)} face(s) in dummy image")
    
    # Test emotion prediction
    print("\n3. Testing emotion classification...")
    if len(faces) > 0:
        x1, y1, x2, y2 = faces[0]['bbox']
        face_crop = dummy_image[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            emotion, confidence, inf_time = detector.predict_emotion(face_crop)
            print(f"   Predicted emotion: {emotion}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Inference time: {inf_time:.4f}s")
    
    # Test complete pipeline
    print("\n4. Testing complete detect_and_classify pipeline...")
    results, total_time = detector.detect_and_classify(dummy_image)
    print(f"   Total inference time: {total_time:.4f}s")
    print(f"   Results: {len(results)} face(s) processed")
    
    print("\n" + "=" * 50)
    print("✅ CNN Detector Test Complete!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_cnn_detector()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
