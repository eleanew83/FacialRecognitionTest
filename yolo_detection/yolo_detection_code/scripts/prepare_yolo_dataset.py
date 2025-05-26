import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths setup for running from scripts directory
script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_code_dir = os.path.dirname(script_dir)
yolo_dir = os.path.dirname(yolo_code_dir)
root_dir = os.path.dirname(yolo_dir)

SOURCE_DIR = os.path.join(root_dir, "macaque_split_data")
DESTINATION_DIR = os.path.join(root_dir, "yolo_detection", "yolo_detection_data")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def process_images():
    """Process macaque images and generate YOLO format annotations"""
    print("Loading pretrained model for face detection...")
    model_path = os.path.join(script_dir, "yolov8n.pt")
    print(f"Model path: {model_path}")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Destination directory: {DESTINATION_DIR}")
    
    # Check if the source directory exists
    train_dir = os.path.join(SOURCE_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory does not exist: {train_dir}")
        return
    
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(DESTINATION_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DESTINATION_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(DESTINATION_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DESTINATION_DIR, 'labels', 'val'), exist_ok=True)
    
    model = YOLO(model_path)  # Using YOLOv8 nano model
    
    # Collect all image paths from subdirectories (case-insensitive extension search)
    all_images = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            # Case-insensitive check for image extensions
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images in macaque_split_data/train")
    
    # Shuffle and split the dataset
    random.shuffle(all_images)
    train_size = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]
    
    # Process train and validation images
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"Processing {split} images...")
        for img_path in tqdm(image_list):
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                height, width = img.shape[:2]
                
                # Generate unique filename
                filename = os.path.basename(img_path)
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_{hash(img_path) % 10000:04d}.jpg"
                
                # Run inference with standard YOLOv8 model
                results = model(img, verbose=False)
                
                # Find person detections (class 0 in COCO)
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls.item())
                        if cls == 0:  # Person class, which might include faces
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf.item()
                            if confidence > 0.3:  # Confidence threshold
                                detections.append((x1, y1, x2, y2))
                
                # If no detections, skip or use the whole image
                if not detections:
                    # For macaques, we'll assume the entire image has a macaque
                    # and use a central region as an initial guess
                    center_x, center_y = width / 2, height / 2
                    box_w, box_h = width * 0.25, height * 0.25  # Smaller boxes work better for macaque faces
                    x1 = max(0, center_x - box_w / 2)
                    y1 = max(0, center_y - box_h / 2)
                    x2 = min(width, center_x + box_w / 2)
                    y2 = min(height, center_y + box_h / 2)
                    detections = [(x1, y1, x2, y2)]
                
                # Save the image
                img_save_path = os.path.join(DESTINATION_DIR, 'images', split, new_filename)
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(img_save_path, img)
                
                # Create YOLO format annotation (class x_center y_center width height)
                label_save_path = os.path.join(DESTINATION_DIR, 'labels', split, 
                                           os.path.splitext(new_filename)[0] + '.txt')
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                
                with open(label_save_path, 'w') as f:
                    for x1, y1, x2, y2 in detections:
                        # Convert to YOLO format (normalized)
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        
                        # Class 0 for macaque face
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def main():    
    # Download YOLOv8 model if not present
    model_path = os.path.join(script_dir, "yolov8n.pt")
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 nano model...")
        os.system(f"wget -O {model_path} https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    
    process_images()
    print("Dataset preparation completed!")
    
    # Print statistics
    try:
        train_images = len(os.listdir(os.path.join(DESTINATION_DIR, 'images', 'train')))
        val_images = len(os.listdir(os.path.join(DESTINATION_DIR, 'images', 'val')))
        print(f"Train images: {train_images}")
        print(f"Validation images: {val_images}")
    except Exception as e:
        print(f"Error getting statistics: {e}")

if __name__ == "__main__":
    main() 