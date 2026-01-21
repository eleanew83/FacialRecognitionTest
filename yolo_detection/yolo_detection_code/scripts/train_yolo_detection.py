from ultralytics import YOLO
import torch
import os
import argparse
import shutil
from datetime import datetime

# Define base paths for the new structure
YOLO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the yolo base directory
MODEL_DIR = os.path.join(YOLO_BASE, "models")
RUNS_DIR = os.path.join(MODEL_DIR, "runs")
LEGACY_DIR = os.path.join(MODEL_DIR, "legacy")
LAST_RUN_FILE = os.path.join(MODEL_DIR, "latest_run.txt")
OUTPUT_DIR = os.path.join(YOLO_BASE, "output")
DATASET_DIR = "/home/ylj20/FacialRecognitionTest/yolo_detection/yolo_detection_data"  # Absolute path to dataset

def default_run_name(version: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    return f"macaque_face_detector_{date_str}_{version}"


def resolve_latest_run() -> str | None:
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
            run_name = f.read().strip()
            if run_name:
                return run_name
    if not os.path.isdir(RUNS_DIR):
        return None
    runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    if not runs:
        return None
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(RUNS_DIR, d)), reverse=True)
    return runs[0]


def train_model(epochs=100, batch_size=16, img_size=640, device="0", run_name: str | None = None):
    """
    Train a YOLOv8 model for macaque face detection
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size for the model
        device: Device to train on (0 for first GPU, cpu for CPU)
    """
    # Check for GPU
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create model directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(LEGACY_DIR, exist_ok=True)
    
    # Load a pretrained YOLO model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(
        data=os.path.join(DATASET_DIR, "dataset.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=RUNS_DIR,
        name=run_name,
        patience=20,  # Early stopping patience
        save=True,  # Save best checkpoint
        verbose=True
    )
    
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        f.write(run_name or "")

    print(f"Training completed. Model saved to {os.path.join(RUNS_DIR, run_name)}")
    return results

def crop_faces(detection_model_path, output_dir=None, confidence=0.3):
    """
    Use the trained detection model to crop macaque faces from images
    
    Args:
        detection_model_path: Path to the trained YOLOv8 model
        output_dir: Directory to save cropped faces
        confidence: Confidence threshold for detections
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "macaque_crops")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the trained model
    model = YOLO(detection_model_path)
    
    # Get all macaque images
    source_folder = os.path.join(os.path.dirname(YOLO_BASE), "macaque_split_data")
    all_images = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    # Process each image
    print(f"Processing {len(all_images)} images to crop faces...")
    for img_path in all_images:
        try:
            # Preserve split structure: split/macaque_id
            rel_path = os.path.relpath(img_path, source_folder)
            parts = rel_path.split(os.sep)
            if len(parts) < 3:
                continue
            split_name = parts[0]
            macaque_id = parts[1]
            macaque_output_dir = os.path.join(output_dir, split_name, macaque_id)
            if not os.path.exists(macaque_output_dir):
                os.makedirs(macaque_output_dir)
            
            # Run the model on the image
            results = model(img_path, conf=confidence)
            
            # Load the image once (needed for crops and label normalization)
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]

            # Collect all predicted boxes and track best for cropping
            all_boxes = []
            best_box = None
            best_conf = -1.0
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                for box in boxes:
                    conf = float(box.conf.item()) if box.conf is not None else 0.0
                    all_boxes.append((box, conf))
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

            if not all_boxes:
                continue

            # Get image basename
            base_filename = os.path.basename(img_path)
            name, ext = os.path.splitext(base_filename)

            # Save only the highest-confidence crop
            if best_box is None:
                continue
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_filename = f"{name}_crop0{ext}"
            crop_path = os.path.join(macaque_output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Face cropping completed. Cropped faces saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for macaque face detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device to train on (0 for GPU, cpu for CPU)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "crop", "both"], 
                       help="Mode: train model, crop faces, or both")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to detection model for cropping (defaults to best model)")
    parser.add_argument("--run-version", type=str, default="v1",
                        help="Version tag used in the run name (e.g., v1, v2)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name (default: YYYYMMDD_<run-version>)")
    
    args = parser.parse_args()
    
    trained_model_path = None
    
    if args.mode in ["train", "both"]:
        run_name = args.run_name or default_run_name(args.run_version)
        train_results = train_model(args.epochs, args.batch, args.img_size, args.device, run_name=run_name)
        trained_model_path = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
    
    if args.mode in ["crop", "both"]:
        # Create output directory only when cropping is requested
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        model_path = args.model if args.model else trained_model_path
        if model_path is None:
            latest_run = resolve_latest_run()
            if latest_run:
                model_path = os.path.join(RUNS_DIR, latest_run, "weights", "best.pt")
            else:
                model_path = None
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first or provide a valid model path with --model")
            exit(1)
            
        output_location = crop_faces(model_path)
        print(f"Cropped faces are available at: {output_location}") 