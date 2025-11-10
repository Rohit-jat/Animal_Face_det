import os
import shutil
from PIL import Image
import random

def create_sample_dataset(dataset_path='dataset', num_samples=10):
    """
    Create a sample dataset structure with placeholder images.
    This is for demonstration purposes only.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.join(dataset_path, 'happy'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'sad'), exist_ok=True)
    
    # Create sample images (just solid color images for demonstration)
    emotions = ['happy', 'sad']
    
    for emotion in emotions:
        for i in range(num_samples):
            # Create a simple colored image
            if emotion == 'happy':
                # Yellow image for happy
                color = (255, 255, 0)
            else:
                # Blue image for sad
                color = (0, 0, 255)
                
            # Create image
            image = Image.new('RGB', (224, 224), color)
            
            # Save image
            image_path = os.path.join(dataset_path, emotion, f'{emotion}_sample_{i+1}.jpg')
            image.save(image_path)
    
    print(f"Sample dataset created with {num_samples} images per emotion class.")

def organize_dataset(source_folder, dataset_path='dataset'):
    """
    Organize images from a source folder into the dataset structure.
    Assumes filenames contain 'happy' or 'sad' to determine their class.
    """
    # Create directories if they don't exist
    happy_path = os.path.join(dataset_path, 'happy')
    sad_path = os.path.join(dataset_path, 'sad')
    os.makedirs(happy_path, exist_ok=True)
    os.makedirs(sad_path, exist_ok=True)
    
    # Process images in source folder
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist.")
        return
    
    happy_count = 0
    sad_count = 0
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(source_folder, filename)
            
            # Determine class based on filename
            if 'happy' in filename.lower():
                dst_path = os.path.join(happy_path, filename)
                shutil.copy(src_path, dst_path)
                happy_count += 1
            elif 'sad' in filename.lower():
                dst_path = os.path.join(sad_path, filename)
                shutil.copy(src_path, dst_path)
                sad_count += 1
            else:
                print(f"Skipping {filename} - couldn't determine emotion class")
    
    print(f"Organized dataset: {happy_count} happy images, {sad_count} sad images")

if __name__ == "__main__":
    print("Dataset Creation Utility")
    print("1. Create sample dataset")
    print("2. Organize existing images")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        num_samples = input("Enter number of samples per class (default 10): ")
        try:
            num_samples = int(num_samples) if num_samples else 10
        except ValueError:
            num_samples = 10
        create_sample_dataset(num_samples=num_samples)
    elif choice == '2':
        source_folder = input("Enter path to source folder: ")
        organize_dataset(source_folder)
    else:
        print("Invalid choice")