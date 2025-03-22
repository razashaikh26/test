# Model Files

This directory should contain the YOLOv8 model files.

## Required Files

1. `yolov8n.pt` - YOLOv8 Nano model (or any YOLOv8 variant)
2. `coco.names` - Class names file

## How to Get the Model Files

### YOLOv8 Model

You can download YOLOv8 model from:
- Ultralytics GitHub: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Or use a Google Drive link and provide it as an environment variable on Render.

### COCO Names File

The coco.names file is already included in this directory. If missing, create a text file with the COCO class names:

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
...
```

## Environment Variables

You can customize the model paths with these environment variables:
- `YOLO_MODEL_PATH`: Path to the YOLOv8 model file
- `COCO_NAMES`: Path to the class names file 