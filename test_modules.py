import os
import sys

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        import detect
        print("✓ detect module imported successfully")
    except Exception as e:
        print(f"✗ Error importing detect module: {e}")
    
    try:
        import train_emotion
        print("✓ train_emotion module imported successfully")
    except Exception as e:
        print(f"✗ Error importing train_emotion module: {e}")
    
    try:
        import app
        print("✓ app module imported successfully")
    except Exception as e:
        print(f"✗ Error importing app module: {e}")

def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = ['dataset', 'models', 'dataset/happy', 'dataset/sad']
    
    for dir_path in required_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")

def test_requirements():
    """Test that required packages can be imported."""
    required_packages = [
        'cv2', 'numpy', 'tensorflow', 'ultralytics', 
        'PIL', 'streamlit'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'ultralytics':
                import ultralytics
            elif package == 'PIL':
                from PIL import Image
            elif package == 'streamlit':
                import streamlit
            print(f"✓ Package imported: {package}")
        except ImportError as e:
            print(f"✗ Error importing package {package}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error with package {package}: {e}")

if __name__ == "__main__":
    print("Testing Animal Face Emotion Classifier Modules")
    print("=" * 50)
    
    print("\n1. Testing imports:")
    test_imports()
    
    print("\n2. Testing directory structure:")
    test_directory_structure()
    
    print("\n3. Testing package imports:")
    test_requirements()
    
    print("\n" + "=" * 50)
    print("Testing complete!")