#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import shutil

# Global variables for detection models
MEDIAPIPE_MODEL = None
YOLO_MODEL = None
HAAR_MODEL = None

# Minimum acceptable box size (as a ratio of image dimensions)
MIN_BOX_SIZE_RATIO = 0.05
# Standard placeholder annotation values
PLACEHOLDER_ANNOTATION = (0.5, 0.5, 0.5, 0.5)

def load_mediapipe(conf=0.3):
    """Load MediaPipe face detection model"""
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        return mp_face_detection.FaceDetection(min_detection_confidence=conf)
    except ImportError:
        print("⚠️ MediaPipe not installed. Using fallback methods.")
        return None

def load_yolo(model_path=None):
    """Load YOLOv8 model for detection"""
    try:
        from ultralytics import YOLO
        if model_path and os.path.exists(model_path):
            return YOLO(model_path)
        else:
            # Use default YOLOv8n model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_model = os.path.join(script_dir, "yolov8n.pt")
            if os.path.exists(default_model):
                return YOLO(default_model)
            else:
                print("⚠️ YOLOv8 model not found. Using fallback methods.")
                return None
    except ImportError:
        print("⚠️ Ultralytics not installed. Using fallback methods.")
        return None

def load_haar():
    """Load Haar cascade classifier for face detection"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    except:
        print("⚠️ Haar cascade not available. Using fallback methods.")
        return None

def is_placeholder_annotation(x_center, y_center, w, h):
    """Check if annotation is exactly the placeholder (0.5, 0.5, 0.5, 0.5)"""
    # Use a small epsilon for float comparison
    epsilon = 1e-6
    return (abs(x_center - 0.5) < epsilon and 
            abs(y_center - 0.5) < epsilon and 
            abs(w - 0.5) < epsilon and 
            abs(h - 0.5) < epsilon)

def is_box_too_small(w, h):
    """Check if the box is too small (less than the minimum ratio)"""
    return w < MIN_BOX_SIZE_RATIO or h < MIN_BOX_SIZE_RATIO

def detect_face_mediapipe(image, confidence=0.3):
    """Detect faces using MediaPipe"""
    if MEDIAPIPE_MODEL is None:
        return []
    
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = MEDIAPIPE_MODEL.process(image_rgb)
    
    bboxes = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * width))
            y1 = max(0, int(bbox.ymin * height))
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Add margin for better face coverage (20% larger)
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.2)
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(width, x1 + w + 2*margin_w)
            y2 = min(height, y1 + h + 2*margin_h)
            
            conf = detection.score[0]
            bboxes.append((x1, y1, x2, y2, conf))
    
    return bboxes

def detect_face_yolo(image, confidence=0.3):
    """Detect faces using YOLOv8"""
    if YOLO_MODEL is None:
        return []
    
    results = YOLO_MODEL(image, conf=confidence, verbose=False)
    bboxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls.item())
            if cls == 0:  # Person class in COCO, might include faces
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                bboxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    
    return bboxes

def detect_face_haar(image):
    """Detect faces using Haar cascade"""
    if HAAR_MODEL is None:
        return []
    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = HAAR_MODEL.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    bboxes = []
    for (x, y, w, h) in faces:
        # Add margin for better face coverage (20% larger)
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(width, x + w + margin_w)
        y2 = min(height, y + h + margin_h)
        
        # Haar doesn't provide confidence, use 0.5 as default
        bboxes.append((x1, y1, x2, y2, 0.5))
    
    return bboxes

def get_central_box(image):
    """Get a central box as fallback"""
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2
    box_w, box_h = width * 0.5, height * 0.5
    x1 = max(0, int(center_x - box_w / 2))
    y1 = max(0, int(center_y - box_h / 2))
    x2 = min(width, int(center_x + box_w / 2))
    y2 = min(height, int(center_y + box_h / 2))
    
    return (x1, y1, x2, y2, 0.1)  # Low confidence for fallback

def select_best_detection(detections):
    """Select the best detection based on confidence and size"""
    if not detections:
        return None
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    return sorted_dets[0]

def detect_face_multi_method(image, confidence=0.3):
    """Try multiple face detection methods and return the best one"""
    detections = []
    
    # Try YOLO first
    yolo_dets = detect_face_yolo(image, confidence)
    if yolo_dets:
        detections.extend(yolo_dets)
        method = "YOLO"
    
    # If no YOLO detections, try Haar
    if not detections:
        haar_dets = detect_face_haar(image)
        if haar_dets:
            detections.extend(haar_dets)
            method = "Haar"
    
    # If still no detections, try MediaPipe with higher confidence
    if not detections:
        mp_dets = detect_face_mediapipe(image, 0.4)  # Use 0.4 confidence for MediaPipe
        if mp_dets:
            detections.extend(mp_dets)
            method = "MediaPipe"
    
    # If all methods failed, use central box
    if not detections:
        detections.append(get_central_box(image))
        method = "Fallback"
    
    # Select best detection
    best_det = select_best_detection(detections)
    return best_det, method

def _select_image_files(images_dir, limit=None):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if limit is not None and limit > 0 and len(image_files) > limit:
        image_files = image_files[:limit]
    return image_files


def visualize_annotations(images_dir, labels_dir, output_dir=None, limit=None):
    """Visualize existing annotations"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image_files = _select_image_files(images_dir, limit=limit)
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Track statistics
    multi_box_count = 0
    placeholder_count = 0

    for img_file in tqdm(image_files, desc="Visualizing annotations"):
        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        if not os.path.exists(label_file):
            continue

        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        height, width = image.shape[:2]

        with open(label_file, 'r') as f:
            annotations = f.readlines()
        
        # Check for multiple boxes
        if len(annotations) > 1:
            multi_box_count += 1
        
        # Check for placeholder annotations
        has_placeholder = False
        
        for annotation in annotations:
            parts = annotation.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, w, h = map(float, parts)
            
            # Check if this is a placeholder annotation
            if is_placeholder_annotation(x_center, y_center, w, h):
                has_placeholder = True
                placeholder_count += 1
            
            # Draw bounding box
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            
            # Use red for placeholder, green for good detection
            color = (0, 0, 255) if has_placeholder else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"macaque_face"
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if output_dir:
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, image)
    
    print(f"✅ Visualization complete")
    print(f"📊 Images with multiple boxes: {multi_box_count}")
    print(f"📊 Images with placeholder boxes: {placeholder_count}")
    
    return multi_box_count, placeholder_count

def fix_annotations(images_dir, labels_dir, confidence=0.3, limit=None):
    """Fix annotations using multiple detection methods"""
    global MEDIAPIPE_MODEL, YOLO_MODEL, HAAR_MODEL
    
    # Load detection models if not already loaded
    if MEDIAPIPE_MODEL is None:
        MEDIAPIPE_MODEL = load_mediapipe(confidence)
    
    if YOLO_MODEL is None:
        YOLO_MODEL = load_yolo()
    
    if HAAR_MODEL is None:
        HAAR_MODEL = load_haar()
    
    image_files = _select_image_files(images_dir, limit=limit)
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Check how many already have valid annotations (for resume functionality)
    already_fixed = 0
    placeholder_count = 0
    needs_processing = []
    
    for img_file in image_files:
        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        
        needs_fixing = True
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                
                # If exactly one annotation and not a placeholder and not too small, keep it
                if len(annotations) == 1:
                    parts = annotations[0].strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, w, h = map(float, parts)
                        
                        # Check if this is a placeholder annotation
                        if is_placeholder_annotation(x_center, y_center, w, h):
                            placeholder_count += 1
                            needs_fixing = True  # Force re-processing of placeholder annotations
                        elif not is_box_too_small(w, h):
                            needs_fixing = False
                            already_fixed += 1
            except Exception as e:
                # If there's an error reading the file, it needs fixing
                print(f"Error reading {label_file}: {e}")
                pass
        
        if needs_fixing:
            needs_processing.append(img_file)
    
    print(f"📊 Resume status: {already_fixed} already have good annotations")
    print(f"📊 Found {placeholder_count} placeholder annotations to be fixed")
    print(f"📊 Total {len(needs_processing)} images need processing")
    
    if len(needs_processing) == 0:
        print("✅ All annotations are already fixed!")
        return 0, {}
    
    # Statistics
    fixed_count = 0
    small_box_count = 0
    skipped_count = 0
    method_counts = {"MediaPipe": 0, "YOLO": 0, "Haar": 0, "Fallback": 0, "Placeholder (small box)": 0}
    
    for img_file in tqdm(needs_processing, desc="Fixing annotations"):
        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        img_path = os.path.join(images_dir, img_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            skipped_count += 1
            continue
        
        height, width = image.shape[:2]
        
        # Detect face using multiple methods
        best_detection, method = detect_face_multi_method(image, confidence)
        
        if best_detection:
            x1, y1, x2, y2, conf = best_detection
            
            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Check if box is too small, use placeholder if it is
            if is_box_too_small(w, h):
                x_center, y_center, w, h = PLACEHOLDER_ANNOTATION
                method = "Placeholder (small box)"
                small_box_count += 1
            
            # Write new annotation
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center} {y_center} {w} {h}\n")
            
            fixed_count += 1
            method_counts[method] += 1
    
    print(f"✅ Fixed {fixed_count} annotations")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} images (could not read)")
    print(f"📊 Boxes replaced with placeholder due to small size: {small_box_count}")
    print(f"📊 Detection methods used:")
    for method, count in method_counts.items():
        if count > 0:
            print(f"  - {method}: {count} images")
    
    return fixed_count, method_counts

def report_problematic_annotations(images_dir, labels_dir, limit=None):
    """Report only problematic annotation files without generating images."""
    image_files = _select_image_files(images_dir, limit=limit)
    print(f"Found {len(image_files)} images in {images_dir}")

    issues = []
    placeholder_count = 0
    multi_box_count = 0
    missing_label_count = 0
    invalid_label_count = 0
    small_box_count = 0

    for img_file in tqdm(image_files, desc="Scanning annotations"):
        label_file = os.path.join(labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
        if not os.path.exists(label_file):
            issues.append((img_file, "missing_label"))
            missing_label_count += 1
            continue

        try:
            with open(label_file, 'r') as f:
                annotations = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            issues.append((img_file, f"read_error:{e}"))
            invalid_label_count += 1
            continue

        if len(annotations) > 1:
            issues.append((img_file, "multiple_boxes"))
            multi_box_count += 1

        for annotation in annotations:
            parts = annotation.split()
            if len(parts) != 5:
                issues.append((img_file, "invalid_format"))
                invalid_label_count += 1
                continue

            _, x_center, y_center, w, h = map(float, parts)
            if is_placeholder_annotation(x_center, y_center, w, h):
                issues.append((img_file, "placeholder_box"))
                placeholder_count += 1
            elif is_box_too_small(w, h):
                issues.append((img_file, "box_too_small"))
                small_box_count += 1

    print("\n=== Problematic annotation summary ===")
    print(f"missing_label: {missing_label_count}")
    print(f"multiple_boxes: {multi_box_count}")
    print(f"placeholder_box: {placeholder_count}")
    print(f"box_too_small: {small_box_count}")
    print(f"invalid_label: {invalid_label_count}")
    print(f"total_problematic: {len(issues)}")

    if issues:
        print("\n=== Problematic files ===")
        for img_file, reason in issues:
            print(f"{img_file}\t{reason}")

    return issues

def main():
    parser = argparse.ArgumentParser(description="Visualize and fix YOLO annotations")
    parser.add_argument("--fix", action="store_true", help="Fix bad annotations")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--custom-model", type=str, help="Path to custom YOLO model (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Process the first N images per split")
    parser.add_argument("--report", action="store_true", help="Report only problematic files (no visualization)")
    args = parser.parse_args()

    dataset_dir = "/home/ylj20/FacialRecognitionTest/yolo_detection/yolo_detection_data"
    
    # Clean visualization directory if it exists (skip when reporting only)
    vis_base_dir = os.path.join(dataset_dir, 'visualization')
    if not args.report and os.path.exists(vis_base_dir):
        print(f"Cleaning visualization directory: {vis_base_dir}")
        shutil.rmtree(vis_base_dir)
    
    # Load models if fixing
    global MEDIAPIPE_MODEL, YOLO_MODEL, HAAR_MODEL
    if args.fix:
        print("🔍 Loading detection models...")
        MEDIAPIPE_MODEL = load_mediapipe(args.confidence)
        YOLO_MODEL = load_yolo(args.custom_model)
        HAAR_MODEL = load_haar()

    for split in ['train', 'val']:
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        vis_dir = os.path.join(dataset_dir, 'visualization', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"⚠️ {split} split not found, skipping")
            continue
        
        if args.report:
            print(f"\n🧾 Reporting issues for split: {split}")
            report_problematic_annotations(images_dir, labels_dir, limit=args.limit)
            continue

        os.makedirs(vis_dir, exist_ok=True)

        if args.fix:
            print(f"\n🔧 Fixing annotations for split: {split}")
            fixed_count, method_counts = fix_annotations(images_dir, labels_dir, args.confidence, limit=args.limit)
            
            # Visualize the fixed annotations
            print(f"\n👁️ Visualizing fixed annotations for split: {split}")
            visualize_annotations(images_dir, labels_dir, vis_dir, limit=args.limit)
        else:
            print(f"\n👁️ Visualizing annotations for split: {split}")
            multi_box_count, placeholder_count = visualize_annotations(images_dir, labels_dir, vis_dir, limit=args.limit)

            if multi_box_count > 0 or placeholder_count > 0:
                print("⚠️ Found issues with annotations. Run with --fix to repair them.")
                print("ℹ️  Use --report to print filenames of problematic files.")

if __name__ == "__main__":
    main()
