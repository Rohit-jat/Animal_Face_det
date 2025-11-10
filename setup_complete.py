import os
import sys

def check_setup():
    """
    Check if all required components are properly set up.
    """
    print("Animal Face Emotion Classifier - Setup Verification")
    print("=" * 55)
    
    # Check directories
    print("\n1. Directory Structure:")
    required_dirs = ['dataset', 'models', 'dataset/happy', 'dataset/sad']
    for dir_path in required_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
    
    # Check model files
    print("\n2. Model Files:")
    model_files = ['models/yolov8n.pt']
    for model_file in model_files:
        full_path = os.path.join(os.getcwd(), model_file)
        if os.path.exists(full_path):
            print(f"  ✓ {model_file}")
        else:
            print(f"  ✗ {model_file} (missing)")
    
    # Check Python packages
    print("\n3. Python Packages:")
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('tensorflow', 'tensorflow'),
        ('ultralytics', 'ultralytics'),
        ('PIL', 'Pillow'),
        ('streamlit', 'streamlit')
    ]
    
    for package_import, package_name in required_packages:
        try:
            if package_import == 'cv2':
                import cv2
                print(f"  ✓ {package_name} (OpenCV)")
            elif package_import == 'numpy':
                import numpy
                print(f"  ✓ {package_name}")
            elif package_import == 'tensorflow':
                import tensorflow
                print(f"  ✓ {package_name} (version: {tensorflow.__version__})")
            elif package_import == 'ultralytics':
                import ultralytics
                print(f"  ✓ {package_name}")
            elif package_import == 'PIL':
                from PIL import Image
                print(f"  ✓ {package_name} (Pillow)")
            elif package_import == 'streamlit':
                import streamlit
                print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name} (not installed or import error)")
        except Exception as e:
            print(f"  ✗ {package_name} (error: {e})")
    
    # Check Python scripts
    print("\n4. Python Scripts:")
    script_files = ['app.py', 'train_emotion.py', 'detect.py', 'download_models.py']
    for script_file in script_files:
        full_path = os.path.join(os.getcwd(), script_file)
        if os.path.exists(full_path):
            print(f"  ✓ {script_file}")
        else:
            print(f"  ✗ {script_file} (missing)")
    
    print("\n" + "=" * 55)
    print("Setup verification complete!")
    print("\nTo train the emotion classifier:")
    print("  python train_emotion.py")
    print("\nTo run the application:")
    print("  streamlit run app.py")
    print("\nTo organize your dataset:")
    print("  python create_dataset.py")

if __name__ == "__main__":
    check_setup()