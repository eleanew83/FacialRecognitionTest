#!/usr/bin/env python3

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Fixed absolute paths ===
YOLO_BASE = "/home/ylj20/FacialRecognitionTest"
SOURCE_DIR = os.path.join(YOLO_BASE, "macaque_split_data")
DATA_DIR = os.path.join(YOLO_BASE, "yolo_detection", "yolo_detection_data")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def create_dataset_structure():
    """Create the necessary directory structure for YOLOv8"""
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            path = os.path.join(DATA_DIR, folder, split)
            os.makedirs(path, exist_ok=True)

    # Create dataset.yaml for YOLO training
    yaml_content = f"""
path: {os.path.abspath(DATA_DIR)}
train: images/train
val: images/val

# Classes
names:
  0: macaque_face
"""
    with open(os.path.join(DATA_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

def process_images():
    """Process macaque images and generate YOLO format annotations"""
    print("Loading pretrained model for face detection...")

    yolo_detected_count = 0
    fallback_count = 0
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    model = YOLO(model_path)

    all_images = []
    for root, _, files in os.walk(os.path.join(SOURCE_DIR, "train")):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("❌ No images found! Ensure SOURCE_DIR/train contains images.")
        return

    random.shuffle(all_images)
    train_size = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]

    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"Processing {split} images...")
        for img_path in tqdm(image_list):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                height, width = img.shape[:2]
                filename = os.path.basename(img_path)
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_{hash(img_path) % 10000:04d}.jpg"

                results = model(img, verbose=False)

                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls.item())
                        if cls == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf.item()
                            if confidence > 0.3:
                                detections.append((x1, y1, x2, y2))

                if not detections:
                    # Use fallback center box
                    center_x, center_y = width / 2, height / 2
                    box_w, box_h = width * 0.5, height * 0.5
                    x1 = max(0, center_x - box_w / 2)
                    y1 = max(0, center_y - box_h / 2)
                    x2 = min(width, center_x + box_w / 2)
                    y2 = min(height, center_y + box_h / 2)
                    detections = [(x1, y1, x2, y2)]
                    fallback_count += 1
                else:
                    yolo_detected_count += 1

                img_save_path = os.path.join(DATA_DIR, 'images', split, new_filename)
                cv2.imwrite(img_save_path, img)

                label_save_path = os.path.join(DATA_DIR, 'labels', split, os.path.splitext(new_filename)[0] + '.txt')
                with open(label_save_path, 'w') as f:
                    for x1, y1, x2, y2 in detections:
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\n🧾 Detection summary:")
    print(f"Images with YOLO-detected faces: {yolo_detected_count}")
    print(f"Images using fallback center box: {fallback_count}")

def main():
    # Clean existing output directory
    if os.path.exists(DATA_DIR):
        print(f"Removing existing data directory: {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    print(f"[INFO] SOURCE_DIR: {SOURCE_DIR}")
    print(f"[INFO] DATA_DIR:   {DATA_DIR}")

    create_dataset_structure()

    # Download model if not present
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 nano model...")
        os.system(f"wget -O {model_path} https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

    process_images()
    print("✅ Dataset preparation completed!")

    train_images = len(os.listdir(os.path.join(DATA_DIR, 'images', 'train')))
    val_images = len(os.listdir(os.path.join(DATA_DIR, 'images', 'val')))
    print(f"Train images: {train_images}")
    print(f"Validation images: {val_images}")

if __name__ == "__main__":
    main()
