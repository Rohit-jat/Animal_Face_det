import os
from ultralytics import YOLO

def download_yolo_model(model_name='yolov8n.pt', destination_folder='models'):
    """
    Download a pre-trained YOLOv8 model.
    """
    # Create models directory if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    try:
        # Load and download the model (this will download if not already present)
        model = YOLO(model_name)
        print(f"YOLOv8 model {model_name} downloaded successfully!")
        return model
    except Exception as e:
        print(f"Error downloading YOLOv8 model: {e}")
        print("Please ensure you have internet connection and ultralytics installed.")
        return None

if __name__ == "__main__":
    print("Downloading YOLOv8 model...")
    model = download_yolo_model()
    if model:
        print("Model downloaded successfully!")
    else:
        print("Failed to download model.")