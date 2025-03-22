import os
import sys
import urllib.request
import requests
import shutil

def download_file(url, output_path):
    """Download a file from a URL to a specified path"""
    print(f"Downloading from {url} to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Create a request with a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Download with requests (better for larger files)
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        print(f"Successfully downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

def main():
    """Main function to download YOLOv8 model"""
    print("YOLOv8 Model Downloader")
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Set default model URL and path
    yolo_model_url = os.environ.get(
        'YOLO_MODEL_URL', 
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
    )
    yolo_model_path = os.path.join(models_dir, 'yolov8n.pt')
    
    print(f"Using model URL: {yolo_model_url}")
    
    # Download the model
    success = download_file(yolo_model_url, yolo_model_path)
    
    if success:
        print(f"YOLOv8 model downloaded successfully to {yolo_model_path}")
        print(f"File size: {os.path.getsize(yolo_model_path) / (1024 * 1024):.2f} MB")
        return 0
    else:
        print("Failed to download YOLOv8 model")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 