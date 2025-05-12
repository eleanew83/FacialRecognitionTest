# Macaque Face Annotation Guide

## Current Status

The annotations for your YOLOv8 macaque face detection model have been automatically generated using MediaPipe's face detection. These annotations should work for initial training, but they might not be perfectly accurate for all images. Here are options for reviewing and improving the annotations if needed.

## Option 1: Using Your Trained YOLOv5 Model

If you have a trained YOLOv5 model (`Copy of best_yolo_v5.pt`), you can use it to improve annotations:

```bash
cd yolo/scripts
python3 use_custom_model.py --model /path/to/your/Copy\ of\ best_yolo_v5.pt --visualize
```

## Option 2: Manual Annotation with LabelImg

For the most accurate annotations, manual annotation is recommended:

1. Install LabelImg:

```bash
cd yolo/scripts
python3 setup_annotation_tools.py
# Then select option 1 to install LabelImg
```

2. Use LabelImg to annotate macaque faces:

```bash
labelImg ../../yolo_detection/images/train
```

3. In LabelImg:
   - Set "Save Dir" to `../../yolo_detection/labels/train` 
   - Set Format to YOLO
   - Draw bounding boxes around macaque faces
   - Use class name "macaque_face" (or just "0")
   - Save with Ctrl+S or press 'w' to save and move to next image

## Option 3: Using Sample Image Annotation

If you have many images, you might want to manually annotate a smaller subset first:

```bash
cd yolo/scripts
python3 setup_annotation_tools.py
# Then select option 2 to create a sample of 100 images
```

## Option 4: Reviewing Annotations

To check the quality of your annotations:

```bash
cd yolo/scripts
python3 visualize_annotations.py --dataset ../../yolo_detection
```

This will create visualization images in `../../yolo_detection/visualization_train` and `../../yolo_detection/visualization_val` so you can review the annotations.

## Tips for Good Annotations

1. **Be consistent**: Draw bounding boxes in a consistent way across images
2. **Include the whole face**: Include the whole macaque face, from forehead to chin
3. **Add margin**: Include a small margin around the face (10-20% extra space)
4. **Handle occlusion**: For partially obscured faces, annotate the visible part
5. **Use multiple images per individual**: Ensure each macaque has several annotated images from different angles

## Training with Improved Annotations

After improving annotations, you can train your model again:

```bash
cd yolo/scripts
python3 train_yolo_detection.py --mode train --epochs 50 --batch 16 --device cpu
```

## Getting Cropped Faces

Once your model is trained, you can extract face crops:

```bash
cd yolo/scripts
python3 train_yolo_detection.py --mode crop
```

The cropped faces will be saved to `yolo/output/macaque_crops/`, organized by individual macaque IDs. 