import os
import urllib.request
import sys
import ssl

def download_file(url, output_path):
    print(f"Attempting to download from {url} to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Handle SSL certificate issues if any
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # Create a request with a user agent to avoid being blocked
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
        )
        
        # Download the file
        with urllib.request.urlopen(req, context=ctx) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        file_size = os.path.getsize(output_path)
        print(f"Successfully downloaded {output_path} ({file_size} bytes)")
        
        if file_size < 1000:  # If file is too small (possibly an error page)
            with open(output_path, 'r', errors='ignore') as f:
                content = f.read(200)  # Read first 200 chars to check content
                print(f"Warning: File may be invalid. First 200 chars: {content}")
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if the models directory is writable
    try:
        test_file = os.path.join(models_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Directory {models_dir} is writable")
    except Exception as e:
        print(f"Warning: Directory {models_dir} may not be writable: {str(e)}")
    
    # URLs to your model files (replace with your actual URLs)
    yolo_weights_url = os.environ.get('YOLO_WEIGHTS_URL', '')
    yolo_cfg_url = os.environ.get('YOLO_CFG_URL', '')
    coco_names_url = os.environ.get('COCO_NAMES_URL', '')
    
    # Print URLs for debugging
    print(f"YOLO_WEIGHTS_URL: {yolo_weights_url}")
    print(f"YOLO_CFG_URL: {yolo_cfg_url}")
    print(f"COCO_NAMES_URL: {coco_names_url}")
    
    if not yolo_weights_url or not yolo_cfg_url or not coco_names_url:
        print("Error: Model URLs not provided in environment variables.")
        print("Please set YOLO_WEIGHTS_URL, YOLO_CFG_URL, and COCO_NAMES_URL")
        sys.exit(1)
    
    # Download files
    success = True
    yolo_weights_path = os.path.join(models_dir, 'yolov7.weights')
    yolo_cfg_path = os.path.join(models_dir, 'yolov7.cfg')
    coco_names_path = os.path.join(models_dir, 'coco.names')
    
    success &= download_file(yolo_weights_url, yolo_weights_path)
    success &= download_file(yolo_cfg_url, yolo_cfg_path)
    success &= download_file(coco_names_url, coco_names_path)
    
    # Check if files exist and have reasonable sizes
    if os.path.exists(yolo_weights_path):
        size_mb = os.path.getsize(yolo_weights_path) / (1024 * 1024)
        print(f"yolov7.weights exists, size: {size_mb:.2f} MB")
        if size_mb < 10:  # YOLOv7 weights should be at least tens of MB
            print("Warning: yolov7.weights is suspiciously small")
    
    if os.path.exists(yolo_cfg_path):
        size_kb = os.path.getsize(yolo_cfg_path) / 1024
        print(f"yolov7.cfg exists, size: {size_kb:.2f} KB")
    
    if os.path.exists(coco_names_path):
        with open(coco_names_path, 'r', errors='ignore') as f:
            lines = f.readlines()
            print(f"coco.names exists, contains {len(lines)} lines")
            print(f"First few classes: {', '.join(line.strip() for line in lines[:5])}")
    
    # List contents of models directory
    print(f"Contents of {models_dir}:")
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 