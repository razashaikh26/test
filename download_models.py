import os
import urllib.request
import sys

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # URLs to your model files (replace with your actual URLs)
    yolo_weights_url = os.environ.get('YOLO_WEIGHTS_URL', '')
    yolo_cfg_url = os.environ.get('YOLO_CFG_URL', '')
    coco_names_url = os.environ.get('COCO_NAMES_URL', '')
    
    if not yolo_weights_url or not yolo_cfg_url or not coco_names_url:
        print("Error: Model URLs not provided in environment variables.")
        print("Please set YOLO_WEIGHTS_URL, YOLO_CFG_URL, and COCO_NAMES_URL")
        sys.exit(1)
    
    # Download files
    success = True
    success &= download_file(yolo_weights_url, os.path.join(models_dir, 'yolov7.weights'))
    success &= download_file(yolo_cfg_url, os.path.join(models_dir, 'yolov7.cfg'))
    success &= download_file(coco_names_url, os.path.join(models_dir, 'coco.names'))
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 