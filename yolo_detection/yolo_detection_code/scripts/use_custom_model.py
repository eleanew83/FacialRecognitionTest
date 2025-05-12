#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
import shutil
import random

def load_model(model_path, device='cpu'):
    """
    Load a YOLOv5 or YOLOv8 model from path
    """
    try:
        # Try loading as YOLOv8 model first
        from ultralytics import YOLO
        model = YOLO(model_path)
        model_type = "yolov8"
    except Exception as e:
        print(f"Failed to load as YOLOv8 model: {e}")
        try:
            # Try loading as YOLOv5 model
            import sys
            # Add path to yolov5 if you have it installed
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.to(device)
            model_type = "yolov5"
        except Exception as e:
            print(f"Failed to load as YOLOv5 model: {e}")
            print("Please make sure your model is in the correct format and dependencies are installed.")
            return None, None
    
    print(f"Successfully loaded model from {model_path} as {model_type}")
    return model, model_type

def detect_with_model(model, model_type, image_path, conf_threshold=0.3, device='cpu'):
    """
    Run inference with loaded model
    
    Returns:
        List of bounding boxes in [x1, y1, x2, y2] format
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []
        
        height, width = image.shape[:2]
        
        if model_type == "yolov8":
            # YOLOv8 inference
            results = model(image, conf=conf_threshold, verbose=False)
            bboxes = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bboxes.append([x1, y1, x2, y2])
        
        else:  # yolov5
            # YOLOv5 inference
            results = model(image)
            bboxes = []
            
            # Extract detections above threshold
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, xyxy)
                    bboxes.append([x1, y1, x2, y2])
        
        return bboxes
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return []

def process_dataset(dataset_dir, model_path, output_dir=None, conf_threshold=0.3, device='cpu'):
    """
    Process a dataset using a custom trained model
    
    Args:
        dataset_dir: Directory with images/train, images/val, etc.
        model_path: Path to trained YOLOv5 or YOLOv8 model
        output_dir: Output directory (default: same as dataset_dir)
        conf_threshold: Confidence threshold for detections
        device: Device to run model on ('cpu' or 'cuda:0')
    """
    if output_dir is None:
        output_dir = dataset_dir
    
    # Load model
    model, model_type = load_model(model_path, device)
    if model is None:
        return
    
    # Process train and val splits
    for split in ['train', 'val']:
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(output_dir, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist")
            continue
        
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images in {split} split...")
        
        # Process each image
        for img_file in tqdm(image_files):
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
            
            # Get image dimensions
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            height, width = image.shape[:2]
            
            # Detect objects
            bboxes = detect_with_model(model, model_type, img_path, conf_threshold, device)
            
            # If no detections, use center region
            if not bboxes:
                center_x, center_y = width / 2, height / 2
                box_w, box_h = width * 0.4, height * 0.4
                x1 = max(0, int(center_x - box_w / 2))
                y1 = max(0, int(center_y - box_h / 2))
                x2 = min(width, int(center_x + box_w / 2))
                y2 = min(height, int(center_y + box_h / 2))
                
                bboxes = [[x1, y1, x2, y2]]
            
            # Write YOLO format annotations
            with open(label_file, 'w') as f:
                for x1, y1, x2, y2 in bboxes:
                    # Convert to YOLO format (normalized)
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Class 0 for macaque face
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")
    
    print(f"Dataset processing completed. Annotations saved to {output_dir}/labels/")

def visualize_results(dataset_dir, output_dir):
    """
    Visualize the model's annotations and save them for manual review
    """
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    for split in ['train', 'val']:
        split_vis_dir = os.path.join(vis_dir, split)
        os.makedirs(split_vis_dir, exist_ok=True)
        
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(output_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select a random subset for visualization
        if len(image_files) > 100:
            image_files = random.sample(image_files, 100)
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
            
            if not os.path.exists(label_file):
                continue
            
            # Read image and labels
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            height, width = image.shape[:2]
            
            # Read annotations
            with open(label_file, 'r') as f:
                annotations = f.readlines()
            
            # Draw bounding boxes
            for annotation in annotations:
                parts = annotation.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, w, h = map(float, parts)
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add class label
                label = f"macaque_face"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            # Save visualization
            vis_path = os.path.join(split_vis_dir, img_file)
            cv2.imwrite(vis_path, image)
    
    print(f"Visualization saved to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Use a custom trained YOLO model for annotation")
    parser.add_argument("--dataset", type=str, default="../../yolo_detection",
                       help="Directory with images/train, images/val, etc.")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained YOLOv5 or YOLOv8 model")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: same as dataset)")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run model on ('cpu' or 'cuda:0')")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize results for manual review")
    
    args = parser.parse_args()
    
    # Set paths relative to script directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, args.dataset))
    
    output_dir = args.output
    if output_dir is not None:
        output_dir = os.path.abspath(os.path.join(script_dir, output_dir))
    
    # Process dataset with model
    process_dataset(
        dataset_dir=dataset_dir,
        model_path=args.model,
        output_dir=output_dir,
        conf_threshold=args.confidence,
        device=args.device
    )
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(dataset_dir, output_dir or dataset_dir)

if __name__ == "__main__":
    main() 