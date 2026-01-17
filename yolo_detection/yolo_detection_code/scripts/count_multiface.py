#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from tqdm import tqdm


DATASET_DIR = "/home/ylj20/FacialRecognitionTest/yolo_detection/yolo_detection_data"
DEFAULT_MODEL = "/home/ylj20/FacialRecognitionTest/yolo_detection/yolo_detection_code/models/macaque_face_detector5/weights/best.pt"


def iter_images(images_dir: Path):
    for name in os.listdir(images_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            yield images_dir / name


def count_multiface(images_dir: Path, model: YOLO, conf: float, device: str):
    multiface = []
    total = 0
    for img_path in tqdm(list(iter_images(images_dir)), desc=f"Scanning {images_dir.name}"):
        total += 1
        results = model.predict(str(img_path), conf=conf, device=device, verbose=False)
        if not results:
            continue
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0
        if count >= 2:
            multiface.append((img_path.name, count))
    return total, multiface


def main():
    parser = argparse.ArgumentParser(description="Count images with multiple face detections.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to scan")
    parser.add_argument("--output", default=None, help="Optional path to save multiface list (tsv)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    all_multiface = []
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        images_dir = Path(DATASET_DIR) / "images" / split
        if not images_dir.exists():
            print(f"Skipping missing split: {images_dir}")
            continue
        total, multiface = count_multiface(images_dir, model, args.conf, args.device)
        print(f"{split}: {len(multiface)} multiface out of {total} images")
        all_multiface.extend([(split, name, count) for name, count in multiface])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("split\timage\tcount\n")
            for split, name, count in all_multiface:
                f.write(f"{split}\t{name}\t{count}\n")
        print(f"Wrote multiface list to: {out_path}")


if __name__ == "__main__":
    main()
