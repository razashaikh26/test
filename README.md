# AI Traffic Monitoring

A Streamlit application for vehicle detection using YOLOv7.

## Features

- Upload images to detect vehicles
- Count and classify vehicles in images
- Download processed images with bounding boxes

## Setup Instructions

### Local Development

1. Clone this repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv7 model files:
   - Download `yolov7.weights` and `yolov7.cfg` from the official YOLOv7 repository
   - Download `coco.names` file
   - Place all model files in the `models` directory

4. Run the application:
   ```
   streamlit run main.py
   ```

### Deployment on Render

1. Create a new Web Service on Render
2. Link to your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt && python download_models.py`
   - Start Command: `streamlit run main.py`
4. Add environment variables:
   - PORT: 8501
   - PYTHONUNBUFFERED: true
   - YOLO_WEIGHTS_URL: URL to your YOLOv7 weights file
   - YOLO_CFG_URL: URL to your YOLOv7 configuration file
   - COCO_NAMES_URL: URL to your COCO class names file

## Providing Model Files

Since Render Shell is a paid feature, you can use one of these methods to provide your model files:

### Option 1: Cloud Storage URLs

1. Upload your model files (yolov7.weights, yolov7.cfg, coco.names) to a cloud storage service that allows direct downloads (Google Drive, Dropbox, AWS S3, etc.)
2. Set the download URLs as environment variables in Render:
   - YOLO_WEIGHTS_URL
   - YOLO_CFG_URL
   - COCO_NAMES_URL

### Option 2: Git LFS

1. Install Git LFS: `git lfs install`
2. Track your large model files:
   ```
   git lfs track "*.weights"
   git lfs track "*.cfg"
   ```
3. Add, commit, and push your model files
4. Render will automatically fetch these files during deployment

## License

This project is licensed under the MIT License. 