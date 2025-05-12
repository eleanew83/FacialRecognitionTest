#!/usr/bin/env python3
import os
import subprocess
import sys

def install_labelimg():
    """Install LabelImg annotation tool"""
    print("Installing LabelImg annotation tool...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "labelImg"])
        print("LabelImg installed successfully!")
        print("\nTo use LabelImg for annotation:")
        print("1. Run: labelImg")
        print("2. Open Directory: Navigate to your images folder (yolo_detection/images/train)")
        print("3. Change Save Directory: Set to yolo_detection/labels/train")
        print("4. In LabelImg, go to View > Auto Save mode")
        print("5. Set Format to YOLO in the left panel")
        print("6. Draw bounding boxes around macaque faces")
        print("7. Press 'w' to save and move to next image")
        print("8. Label 'macaque_face' for all boxes")
    except Exception as e:
        print(f"Error installing LabelImg: {e}")
        print("You can try manual installation:")
        print("pip install labelImg")

def setup_cvat_local():
    """Instructions for setting up CVAT locally"""
    print("CVAT Setup Instructions (more powerful but requires Docker):")
    print("1. Clone CVAT: git clone https://github.com/opencv/cvat")
    print("2. cd cvat")
    print("3. docker-compose up -d")
    print("4. Access CVAT at: http://localhost:8080")
    print("5. Create a project and upload your images")
    print("6. Label the images and export in YOLO format")
    print("See full instructions at: https://github.com/opencv/cvat/blob/develop/site/content/en/docs/getting_started.md")

def create_sample_images_for_annotation():
    """Create a sample subset of images for faster annotation"""
    import random
    import shutil
    from pathlib import Path
    
    YOLO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SOURCE_DIR = os.path.join(os.path.dirname(YOLO_BASE), "macaque_split_data")
    SAMPLE_DIR = os.path.join(YOLO_BASE, "annotation_sample")
    
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
    
    # Get all image paths
    all_images = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    # Select a subset for manual annotation (around 100 images)
    if len(all_images) > 100:
        sample_images = random.sample(all_images, 100)
    else:
        sample_images = all_images
    
    # Copy images to sample directory
    for img_path in sample_images:
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(SAMPLE_DIR, filename))
    
    print(f"Created a sample of {len(sample_images)} images in {SAMPLE_DIR}")
    print(f"Annotate these images with LabelImg and use them to train an initial model")
    print(f"To use LabelImg with this sample directory:")
    print(f"labelImg {SAMPLE_DIR}")

def use_roboflow():
    """Instructions for using Roboflow for annotation"""
    print("Roboflow Setup Instructions (cloud-based, easier to use):")
    print("1. Sign up at: https://roboflow.com/")
    print("2. Create a new project")
    print("3. Upload your images")
    print("4. Annotate using their web interface")
    print("5. Export in YOLO format")
    print("\nRoboflow also offers:")
    print("- Annotation assistance")
    print("- Dataset versioning")
    print("- Data augmentation")
    print("- Model training")

def main():
    print("=== Macaque Face Annotation Tools ===")
    print("Choose an annotation approach:")
    print("1. Install LabelImg (local annotation tool)")
    print("2. Create sample image set for annotation")
    print("3. Instructions for CVAT (more powerful)")
    print("4. Instructions for Roboflow (cloud-based)")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        install_labelimg()
    elif choice == "2":
        create_sample_images_for_annotation()
    elif choice == "3":
        setup_cvat_local()
    elif choice == "4":
        use_roboflow()
    else:
        print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main() 