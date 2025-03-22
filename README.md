# AI Traffic Monitoring with YOLOv8

An advanced vehicle detection system using YOLOv8 and Streamlit.

## Features

- Multiple input sources: image upload, webcam, IP camera, and video files
- Real-time vehicle detection and counting
- Adjustable confidence threshold 
- Download processed images

## Setup Instructions

### Local Development

1. Clone this repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 model:
   ```
   python download_model.py
   ```
4. Run the application:
   ```
   streamlit run traffic22.py
   ```

### Deployment on Render

1. Create a new Web Service on Render
2. Link to your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt && python download_model.py`
   - Start Command: `streamlit run traffic22.py`
4. Add environment variables:
   - PORT: 8501
   - PYTHONUNBUFFERED: true
   - YOLO_MODEL_URL (optional): URL to your custom YOLOv8 model

## Usage

1. Choose an input source:
   - **Image Upload**: Process a single image
   - **Webcam**: Live detection from a connected camera (local use only)
   - **IP Camera**: Connect to an IP camera stream
   - **Video File**: Process an uploaded video file

2. Adjust the confidence threshold slider to control detection sensitivity

3. For video sources:
   - Click "Start Detection" to begin processing
   - Click "Stop Detection" to end processing

## Technical Details

- Uses YOLOv8 from Ultralytics
- Detects vehicles including cars, motorcycles, buses, and trucks
- Real-time FPS calculation for video streams

## Limitations on Render

- Webcam access is not available on the Render platform
- For heavy video processing, consider using a higher-tier Render plan

## License

This project is licensed under the MIT License. 