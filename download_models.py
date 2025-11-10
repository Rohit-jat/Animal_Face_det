import os
import urllib.request
from ultralytics import YOLO

def download_yolo_model(model_name='yolov8n.pt', destination_folder='models'):
    """
    Download a pre-trained YOLOv8 model.
    Note: This downloads a general object detection model, not specifically trained for animal faces.
    For production use, you would need to train a custom model for animal face detection.
    """
    # Create models directory if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    try:
        # Load and download the model (this will download if not already present)
        model = YOLO(model_name)
        model_path = os.path.join(destination_folder, model_name)
        
        # Save the model
        model.export(format='pt')
        
        print(f"YOLOv8 model downloaded and saved to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading YOLOv8 model: {e}")
        print("Please ensure you have internet connection and ultralytics installed.")
        return None

def create_custom_model_placeholder(model_path='models/yolov8.pt'):
    """
    Create a placeholder file for a custom animal face detection model.
    In practice, you would train this model specifically for animal faces.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create a simple placeholder file
    with open(model_path, 'w') as f:
        f.write("# Placeholder for custom animal face detection model\n")
        f.write("# In practice, this would be a trained YOLOv8 model\n")
        f.write("# specifically for detecting animal faces.\n")
    
    print(f"Created placeholder model file at {model_path}")

if __name__ == "__main__":
    print("Model Download Utility")
    print("1. Download pre-trained YOLOv8 model (general object detection)")
    print("2. Create placeholder for custom animal face detection model")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        model_name = input("Enter model name (default: yolov8n.pt): ") or 'yolov8n.pt'
        download_yolo_model(model_name)
    elif choice == '2':
        model_path = input("Enter model path (default: models/yolov8.pt): ") or 'models/yolov8.pt'
        create_custom_model_placeholder(model_path)
    else:
        print("Invalid choice")