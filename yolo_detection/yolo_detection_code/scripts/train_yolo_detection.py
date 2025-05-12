from ultralytics import YOLO
import torch
import os
import argparse
import shutil

# Define base paths for the new structure
YOLO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the yolo base directory
MODEL_DIR = os.path.join(YOLO_BASE, "models")
OUTPUT_DIR = os.path.join(YOLO_BASE, "output")
DATASET_DIR = os.path.join(os.path.dirname(YOLO_BASE), "yolo_detection")  # Parent directory's yolo_detection

def train_model(epochs=100, batch_size=16, img_size=640, device="0"):
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
    
    # Create model directory
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
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
        project=MODEL_DIR,
        name="macaque_face_detector",
        patience=20,  # Early stopping patience
        save=True,  # Save best checkpoint
        verbose=True
    )
    
    # Validate the model
    model.val()
    
    print(f"Training completed. Model saved to {os.path.join(MODEL_DIR, 'macaque_face_detector')}")
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
            # Get the macaque ID from the folder name
            macaque_id = os.path.basename(os.path.dirname(img_path))
            macaque_output_dir = os.path.join(output_dir, macaque_id)
            if not os.path.exists(macaque_output_dir):
                os.makedirs(macaque_output_dir)
            
            # Run the model on the image
            results = model(img_path, conf=confidence)
            
            # Save the crops
            for i, result in enumerate(results):
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                
                # Get image basename
                base_filename = os.path.basename(img_path)
                name, ext = os.path.splitext(base_filename)
                
                # Save crops
                for j, box in enumerate(boxes):
                    # Get crop coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Load the image
                    import cv2
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Crop the image
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Save the crop
                    crop_filename = f"{name}_crop{j}{ext}"
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
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    trained_model_path = None
    
    if args.mode in ["train", "both"]:
        train_results = train_model(args.epochs, args.batch, args.img_size, args.device)
        trained_model_path = os.path.join(MODEL_DIR, "macaque_face_detector", "weights", "best.pt")
    
    if args.mode in ["crop", "both"]:
        model_path = args.model if args.model else trained_model_path
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "macaque_face_detector", "weights", "best.pt")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first or provide a valid model path with --model")
            exit(1)
            
        output_location = crop_faces(model_path)
        print(f"Cropped faces are available at: {output_location}") 