from ultralytics import YOLO
import torch
import os
import argparse
import shutil
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "animal-face-id"))
from src.datasets.split_integrity import file_fingerprint, fresh_output_directory, validate_manifest
from datetime import datetime

# Define base paths for the new structure
YOLO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the yolo base directory
SOURCE_SPLIT_DIR = "/home/ylj20/FacialRecognitionTest/macaque_split_data"
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


def train_model(
    epochs=100,
    batch_size=16,
    img_size=640,
    device="0",
    run_name: str | None = None,
    mosaic: float | None = None,
    mixup: float | None = None,
    hsv_h: float | None = None,
    hsv_s: float | None = None,
    hsv_v: float | None = None,
    fliplr: float | None = None,
):
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
    train_kwargs = {
        "data": os.path.join(DATASET_DIR, "dataset.yaml"),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": RUNS_DIR,
        "name": run_name,
        "patience": 20,  # Early stopping patience
        "save": True,  # Save best checkpoint
        "verbose": True,
    }
    # Only set augmentation overrides when explicitly provided
    if mosaic is not None:
        train_kwargs["mosaic"] = mosaic
    if mixup is not None:
        train_kwargs["mixup"] = mixup
    if hsv_h is not None:
        train_kwargs["hsv_h"] = hsv_h
    if hsv_s is not None:
        train_kwargs["hsv_s"] = hsv_s
    if hsv_v is not None:
        train_kwargs["hsv_v"] = hsv_v
    if fliplr is not None:
        train_kwargs["fliplr"] = fliplr

    results = model.train(
        **train_kwargs
    )
    
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        f.write(run_name or "")

    print(f"Training completed. Model saved to {os.path.join(RUNS_DIR, run_name)}")
    return results

def crop_faces(detection_model_path, output_dir=None, confidence=0.3, source_folder=None):
    """
    Use the trained detection model to crop macaque faces from images
    
    Args:
        detection_model_path: Path to the trained YOLOv8 model
        output_dir: Directory to save cropped faces
        confidence: Confidence threshold for detections
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "macaque_crops")
        
    source_folder = source_folder or SOURCE_SPLIT_DIR
    source_path = Path(source_folder).resolve()
    output_path = Path(output_dir).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    if source_path == output_path or source_path in output_path.parents or output_path in source_path.parents:
        raise ValueError("Source splits and crop output must be separate directories.")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Crop output is not empty; choose a fresh --output-dir: {output_path}")
    all_images = sorted(str(path) for path in source_path.rglob("*")
                        if path.is_file() and not path.name.startswith(".")
                        and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    source_manifest = {split: [] for split in ("train", "val", "test")}
    for image_path in all_images:
        relative = Path(image_path).relative_to(source_path)
        if len(relative.parts) != 3 or relative.parts[0] not in source_manifest:
            raise ValueError(f"Unexpected source image layout: {relative}")
        source_manifest[relative.parts[0]].append({"id": relative.parts[1], "path": str(relative),
                                                **file_fingerprint(image_path)})
    validate_manifest(source_manifest, source_path)
    if not all_images:
        raise ValueError("No source images found.")
    with fresh_output_directory(output_path) as staging_path:
        model = YOLO(detection_model_path)

        cropped_manifest = {split: [] for split in ("train", "val", "test")}
        errors = []
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
                macaque_output_dir = os.path.join(staging_path, split_name, macaque_id)
                if not os.path.exists(macaque_output_dir):
                    os.makedirs(macaque_output_dir)
            
                # Run the model on the image
                results = model(img_path, conf=confidence)
            
                # Load the image once (needed for crops and label normalization)
                import cv2
                img = cv2.imread(img_path)
                if img is None:
                    raise OSError(f"Could not read source image: {img_path}")
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
                if not cv2.imwrite(crop_path, crop):
                    raise OSError(f"Failed to write face crop: {crop_path}")
                cropped_manifest[split_name].append({
                    "id": macaque_id, "path": str(Path(crop_path).resolve().relative_to(staging_path)),
                    **file_fingerprint(crop_path),
                })
        
            except Exception as e:
                errors.append(f"{img_path}: {e}")
                print(f"Error processing {img_path}: {e}")
    
        # Different source images can produce an identical highest-confidence face.
        # Keep failed output available for review, but do not report a clean export.
        if errors:
            raise RuntimeError(f"Crop export failed for {len(errors)} images; first error: {errors[0]}")
        validate_manifest(cropped_manifest, staging_path, verify_content=True,
                          expected_labels={r["id"] for rows in source_manifest.values() for r in rows})
        (staging_path / "splits.json").write_text(json.dumps(cropped_manifest, indent=2) + "\n")
    print(f"Face cropping completed. Cropped faces saved to {output_path}")
    return str(output_path)

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
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold for cropping (default: 0.3)")
    parser.add_argument("--source-split-dir", default=SOURCE_SPLIT_DIR, help="Source train/val/test image folders")
    parser.add_argument("--output-dir", default=None, help="New or empty crop output directory")
    # Optional YOLO augmentation overrides (defaults = Ultralytics built-ins)
    parser.add_argument("--mosaic", type=float, default=None, help="Override mosaic prob (e.g., 0.5)")
    parser.add_argument("--mixup", type=float, default=None, help="Override mixup prob (e.g., 0.1)")
    parser.add_argument("--hsv-h", type=float, default=None, help="Override HSV-H (e.g., 0.015)")
    parser.add_argument("--hsv-s", type=float, default=None, help="Override HSV-S (e.g., 0.7)")
    parser.add_argument("--hsv-v", type=float, default=None, help="Override HSV-V (e.g., 0.4)")
    parser.add_argument("--fliplr", type=float, default=None, help="Override horizontal flip prob (e.g., 0.5)")
    
    args = parser.parse_args()
    
    trained_model_path = None
    
    if args.mode in ["train", "both"]:
        run_name = args.run_name or default_run_name(args.run_version)
        train_results = train_model(
            args.epochs,
            args.batch,
            args.img_size,
            args.device,
            run_name=run_name,
            mosaic=args.mosaic,
            mixup=args.mixup,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            fliplr=args.fliplr,
        )
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
            
        output_location = crop_faces(model_path, output_dir=args.output_dir, confidence=args.confidence,
                                     source_folder=args.source_split_dir)
        print(f"Cropped faces are available at: {output_location}") 