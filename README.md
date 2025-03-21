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
   - Place all model files in the `models` directory

4. Run the application:
   ```
   streamlit run main.py
   ```

### Deployment on Render

1. Create a new Web Service on Render
2. Link to your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt && python download_from_drive.py`
   - Start Command: `streamlit run main.py`
4. Add environment variables:
   - PORT: 8501
   - PYTHONUNBUFFERED: true
   - YOLO_WEIGHTS_URL: Google Drive link to your yolov7.weights file
   - YOLO_CFG_URL: Google Drive link to your yolov7.cfg file

## Using Google Drive Links

For Google Drive links, you can use the regular sharing links:

1. Upload your model files to Google Drive
2. Right-click on each file → "Share" → "Anyone with the link" → "Copy link"
3. Use these links in your Render environment variables

The download script will automatically convert the sharing links to direct download links.

Example:
- Original link: `https://drive.google.com/file/d/1abcdefg123456/view?usp=sharing`
- Used as: `YOLO_WEIGHTS_URL=https://drive.google.com/file/d/1abcdefg123456/view?usp=sharing`

## Troubleshooting

If you encounter errors downloading the model files:

1. Make sure your Google Drive links are publicly accessible (set to "Anyone with the link can view")
2. Check the build logs in Render for any error messages
3. For large files (>100MB), try splitting them or using a different hosting service

## License

This project is licensed under the MIT License. 