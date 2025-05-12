#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_annotations(images_dir, labels_dir, output_dir=None):
    """
    Visualize YOLO format annotations
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        output_dir: Directory to save visualization (optional)
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Keep track of bad annotations
    bad_annotations = []
    
    # Process each image
    for img_file in image_files:
        # Get corresponding label file
        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        
        # Check if label file exists
        if not os.path.exists(label_file):
            print(f"WARNING: No label file for {img_file}")
            continue
        
        # Read image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Could not read image {img_path}")
            continue
        
        height, width = image.shape[:2]
        
        # Read annotations
        with open(label_file, 'r') as f:
            annotations = f.readlines()
        
        # Flag for bad annotations
        is_bad = False
        
        # Draw bounding boxes
        for annotation in annotations:
            parts = annotation.strip().split()
            if len(parts) != 5:
                print(f"WARNING: Invalid annotation format in {label_file}")
                continue
            
            class_id, x_center, y_center, w, h = map(float, parts)
            
            # Check if the annotation looks like a placeholder (centered box)
            is_center_placeholder = (abs(x_center - 0.5) < 0.01 and 
                                    abs(y_center - 0.5) < 0.01 and 
                                    abs(w - 0.5) < 0.01 and 
                                    abs(h - 0.5) < 0.01)
            
            if is_center_placeholder:
                is_bad = True
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            # Draw bounding box
            color = (0, 0, 255) if is_center_placeholder else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add class label
            label = f"macaque_face"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        # Save image if the annotation is bad
        if is_bad:
            bad_annotations.append((img_file, label_file))
            
            if output_dir:
                output_path = os.path.join(output_dir, f"bad_{img_file}")
                cv2.imwrite(output_path, image)
        elif output_dir:
            # Save visualization
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, image)
    
    # Print summary
    print(f"Found {len(bad_annotations)} images with likely placeholder annotations")
    if output_dir:
        print(f"Visualizations saved to {output_dir}")
    
    return bad_annotations

def fix_annotations(images_dir, labels_dir, method='mediapipe', confidence=0.3):
    """
    Fix bad annotations using a better face detector
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        method: Method to use for fixing ('mediapipe' or 'haar')
        confidence: Confidence threshold for detection
    """
    # First, find all bad annotations
    print("Checking for bad annotations...")
    bad_annotations = visualize_annotations(images_dir, labels_dir)
    
    if not bad_annotations:
        print("No bad annotations found!")
        return
    
    print(f"Fixing {len(bad_annotations)} bad annotations...")
    
    # Import required modules based on method
    if method == 'mediapipe':
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            detector = mp_face_detection.FaceDetection(min_detection_confidence=confidence)
        except ImportError:
            print("MediaPipe not installed. Please install with: pip install mediapipe")
            return
    else:
        # Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade_path)
    
    # Process each bad annotation
    fixed_count = 0
    for img_file, label_file in bad_annotations:
        # Read image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # Detect faces based on selected method
        bboxes = []
        if method == 'mediapipe':
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(image_rgb)
            
            if results.detections:
                for detection in results.detections:
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
        else:
            # Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                x1 = max(0, x - margin_w)
                y1 = max(0, y - margin_h)
                x2 = min(width, x + w + margin_w)
                y2 = min(height, y + h + margin_h)
                
                bboxes.append([x1, y1, x2, y2])
        
        # If no faces detected, use central region
        if not bboxes:
            center_x, center_y = width / 2, height / 2
            box_w, box_h = width * 0.4, height * 0.4  # Slightly smaller than default
            x1 = max(0, int(center_x - box_w / 2))
            y1 = max(0, int(center_y - box_h / 2))
            x2 = min(width, int(center_x + box_w / 2))
            y2 = min(height, int(center_y + box_h / 2))
            
            bboxes = [[x1, y1, x2, y2]]
        
        # Write fixed annotations
        with open(label_file, 'w') as f:
            for x1, y1, x2, y2 in bboxes:
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                # Class 0 for macaque face
                f.write(f"0 {x_center} {y_center} {w} {h}\n")
        
        fixed_count += 1
    
    print(f"Fixed {fixed_count} out of {len(bad_annotations)} bad annotations")

def check_dataset(dataset_dir):
    """
    Check a dataset for annotation issues
    
    Args:
        dataset_dir: Base directory of dataset with images/ and labels/ subdirectories
    """
    # Check for train and val splits
    for split in ['train', 'val']:
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"WARNING: {split} split not found")
            continue
        
        print(f"Checking {split} split...")
        # Create visualization directory
        vis_dir = os.path.join(dataset_dir, f'visualization_{split}')
        os.makedirs(vis_dir, exist_ok=True)
        
        bad_anns = visualize_annotations(images_dir, labels_dir, vis_dir)
        
        # Calculate statistics
        total_imgs = len([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if total_imgs > 0:
            bad_ratio = len(bad_anns) / total_imgs * 100
            print(f"  - Images with bad annotations: {len(bad_anns)}/{total_imgs} ({bad_ratio:.1f}%)")
        
            if bad_ratio > 50:
                print("  - WARNING: More than 50% of annotations look like placeholders!")
                print("  - Run with --fix to improve annotations")

def main():
    parser = argparse.ArgumentParser(description="Visualize and fix YOLO annotations")
    parser.add_argument("--dataset", type=str, default="../yolo_detection",
                       help="Base directory of dataset with images/ and labels/ subdirectories")
    parser.add_argument("--split", type=str, default="train",
                       help="Split to process (train or val)")
    parser.add_argument("--fix", action="store_true",
                       help="Fix bad annotations")
    parser.add_argument("--method", type=str, choices=['mediapipe', 'haar'], default='mediapipe',
                       help="Method to use for fixing")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    # Set paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, args.dataset))
    
    if args.fix:
        # Fix annotations for specified split
        images_dir = os.path.join(dataset_dir, 'images', args.split)
        labels_dir = os.path.join(dataset_dir, 'labels', args.split)
        fix_annotations(images_dir, labels_dir, args.method, args.confidence)
    else:
        # Check dataset
        check_dataset(dataset_dir)

if __name__ == "__main__":
    main() 