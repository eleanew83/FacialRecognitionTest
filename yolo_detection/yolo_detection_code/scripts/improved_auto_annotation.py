#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import random
import shutil

def install_dependencies():
    """Install necessary dependencies"""
    try:
        import pip
        pip.main(['install', 'mediapipe', 'tqdm'])
        print("Dependencies installed successfully")
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("Try manually installing: pip install mediapipe tqdm")

def detect_faces_mediapipe(image_path, confidence=0.5):
    """
    Detect faces using MediaPipe (works well for primates too)
    
    Args:
        image_path: Path to image
        confidence: Confidence threshold
        
    Returns:
        List of bounding boxes in [x1, y1, x2, y2] format
    """
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=confidence)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []
            
        height, width = image.shape[:2]
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(image_rgb)
        
        # Extract bounding boxes
        bboxes = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * width))
                y1 = max(0, int(bbox.ymin * height))
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                x2 = min(width, x1 + w)
                y2 = min(height, y1 + h)
                
                # Add some margin (20% larger)
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                x1 = max(0, x1 - margin_w)
                y1 = max(0, y1 - margin_h)
                x2 = min(width, x2 + margin_w)
                y2 = min(height, y2 + margin_h)
                
                bboxes.append([x1, y1, x2, y2])
                
        return bboxes
    except ImportError:
        print("MediaPipe not installed. Please install with: pip install mediapipe")
        return []
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def detect_faces_haar(image_path):
    """
    Detect faces using Haar cascades
    
    Args:
        image_path: Path to image
        
    Returns:
        List of bounding boxes in [x1, y1, x2, y2] format
    """
    # Load Haar cascade for frontal face
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return []
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert to [x1, y1, x2, y2] format
    bboxes = []
    height, width = image.shape[:2]
    for (x, y, w, h) in faces:
        # Add some margin (20% larger)
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(width, x + w + margin_w)
        y2 = min(height, y + h + margin_h)
        
        bboxes.append([x1, y1, x2, y2])
    
    return bboxes

def annotate_images(source_dir, output_dir, detector='mediapipe', confidence=0.5):
    """
    Annotate macaque faces using specified detector
    
    Args:
        source_dir: Directory containing macaque images
        output_dir: Directory to save annotated images and labels
        detector: Face detection method ('mediapipe' or 'haar')
        confidence: Confidence threshold for face detection
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get all image paths
    all_images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images in {source_dir}")
    
    # Annotate each image
    success_count = 0
    for img_path in tqdm(all_images):
        try:
            # Detect faces based on selected detector
            if detector == 'mediapipe':
                bboxes = detect_faces_mediapipe(img_path, confidence)
            else:
                bboxes = detect_faces_haar(img_path)
            
            # If no faces detected, use a heuristic approach - central region
            if not bboxes:
                # Read image to get dimensions
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                
                # Use central region (50% of image)
                center_x, center_y = width / 2, height / 2
                box_w, box_h = width * 0.5, height * 0.5
                x1 = max(0, int(center_x - box_w / 2))
                y1 = max(0, int(center_y - box_h / 2))
                x2 = min(width, int(center_x + box_w / 2))
                y2 = min(height, int(center_y + box_h / 2))
                
                bboxes = [[x1, y1, x2, y2]]
            
            # Copy image to output directory
            image_filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(image_filename)
            
            # Create unique filename to avoid conflicts
            output_filename = f"{base_name}_{hash(img_path) % 10000:04d}{ext}"
            output_image_path = os.path.join(images_dir, output_filename)
            
            # Copy image
            shutil.copy(img_path, output_image_path)
            
            # Write YOLO format labels
            label_filename = f"{os.path.splitext(output_filename)[0]}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            # Read image for dimensions
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            height, width = image.shape[:2]
            
            with open(label_path, 'w') as f:
                for x1, y1, x2, y2 in bboxes:
                    # Convert to YOLO format (normalized)
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Class 0 for macaque face
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")
            
            success_count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Successfully annotated {success_count} images")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")
    
    # Create train-val split
    create_train_val_split(images_dir, labels_dir, output_dir)
    
    return output_dir

def create_train_val_split(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    Split annotated data into train and validation sets
    
    Args:
        images_dir: Directory containing annotated images
        labels_dir: Directory containing labels
        output_dir: Base output directory
        train_ratio: Ratio of training data (0-1)
    """
    # Create train/val directories
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            path = os.path.join(output_dir, folder, split)
            os.makedirs(path, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files to train/val directories
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in files:
            # Move image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split, img_file)
            shutil.copy(src_img, dst_img)
            
            # Move corresponding label
            label_file = f"{os.path.splitext(img_file)[0]}.txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, 'labels', split, label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
    
    # Create dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
names:
  0: macaque_face
"""
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    
    print(f"Created train-val split: {len(train_files)} train, {len(val_files)} val")
    print(f"Dataset config saved to: {os.path.join(output_dir, 'dataset.yaml')}")

def main():
    parser = argparse.ArgumentParser(description="Automated annotation of macaque faces")
    parser.add_argument("--source", type=str, default="../macaque_split_data",
                        help="Directory containing macaque images")
    parser.add_argument("--output", type=str, default="../yolo_detection",
                        help="Output directory for annotations")
    parser.add_argument("--detector", type=str, choices=['mediapipe', 'haar'], default='mediapipe',
                        help="Face detection method")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold for face detection")
    parser.add_argument("--install", action='store_true',
                        help="Install dependencies first")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install:
        install_dependencies()
    
    # Set paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.abspath(os.path.join(script_dir, args.source))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output))
    
    # Annotate images
    annotate_images(source_dir, output_dir, args.detector, args.confidence)

if __name__ == "__main__":
    main() 