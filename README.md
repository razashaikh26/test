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
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run main.py`
4. Add environment variables:
   - PORT: 8501
   - PYTHONUNBUFFERED: true

## Model Files

You need to upload the model files to Render. After deploying:

1. Create a `models` directory in the root of your application
2. Upload the following files:
   - `yolov7.weights`
   - `yolov7.cfg`
   - `coco.names`

## License

This project is licensed under the MIT License. 