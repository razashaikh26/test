import os
import re
import sys
import requests
import ssl
import urllib.request

def is_google_drive_link(url):
    """Check if URL is a Google Drive link"""
    return 'drive.google.com' in url

def extract_file_id(url):
    """Extract file ID from Google Drive URL"""
    # Pattern for file/d/FILE_ID/view
    file_id_pattern = r'file/d/([a-zA-Z0-9_-]+)'
    match = re.search(file_id_pattern, url)
    
    if match:
        return match.group(1)
    
    # Alternative pattern for id=FILE_ID
    alt_pattern = r'id=([a-zA-Z0-9_-]+)'
    match = re.search(alt_pattern, url)
    
    if match:
        return match.group(1)
    
    return None

def get_direct_download_link(file_id):
    """Generate direct download link from file ID"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file(url, output_path):
    """Download file from a URL to the output path"""
    print(f"Downloading from {url} to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if it's a Google Drive link and convert if needed
    if is_google_drive_link(url):
        file_id = extract_file_id(url)
        if file_id:
            url = get_direct_download_link(file_id)
            print(f"Converted to direct download link: {url}")
        else:
            print(f"Warning: Couldn't extract file ID from Google Drive URL: {url}")
    
    try:
        # Create a session to handle redirects and cookies
        session = requests.Session()
        
        # Google Drive sometimes asks for confirmation for large files
        response = session.get(url, stream=True)
        
        # Check if we got a confirmation page
        if 'drive.google.com' in url and 'confirm=' in response.text:
            # Extract the confirmation token
            confirm_match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                url = f"{url}&confirm={confirm_token}"
                print(f"Added confirmation token to URL: {url}")
                response = session.get(url, stream=True)
        
        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024*1024):.2f} MB")
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(output_path)
        print(f"Successfully downloaded to {output_path} ({file_size} bytes)")
        
        # Verify the file isn't an HTML error page
        if file_size < 10000 and file_size > 0:  # If file is suspiciously small
            with open(output_path, 'r', errors='ignore') as f:
                content = f.read(1000)
                if '<html' in content.lower() or '<!doctype html' in content.lower():
                    print(f"Warning: Downloaded file appears to be HTML, not the expected model file")
                    print(f"First 1000 chars: {content[:1000]}")
        
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

def main():
    print("Google Drive Download Helper")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Get URLs from environment variables
    yolo_weights_url = os.environ.get('YOLO_WEIGHTS_URL', '')
    yolo_cfg_url = os.environ.get('YOLO_CFG_URL', '')
    coco_names_url = os.environ.get('COCO_NAMES_URL', '')
    
    # Print URLs for debugging
    print(f"YOLO_WEIGHTS_URL: {yolo_weights_url}")
    print(f"YOLO_CFG_URL: {yolo_cfg_url}")
    print(f"COCO_NAMES_URL: {coco_names_url}")
    
    # Define file paths
    yolo_weights_path = os.path.join(models_dir, 'yolov7.weights')
    yolo_cfg_path = os.path.join(models_dir, 'yolov7.cfg')
    coco_names_path = os.path.join(models_dir, 'coco.names')
    
    # Download files
    success = True
    
    if yolo_weights_url:
        success &= download_file(yolo_weights_url, yolo_weights_path)
    else:
        print("Warning: YOLO_WEIGHTS_URL not provided")
    
    if yolo_cfg_url:
        success &= download_file(yolo_cfg_url, yolo_cfg_path)
    else:
        print("Warning: YOLO_CFG_URL not provided")
    
    # We already have coco.names in the repository, so only download if explicitly provided
    if coco_names_url:
        success &= download_file(coco_names_url, coco_names_path)
    
    # Check files
    print("\nChecking downloaded files:")
    
    if os.path.exists(yolo_weights_path):
        size_mb = os.path.getsize(yolo_weights_path) / (1024 * 1024)
        print(f"yolov7.weights exists, size: {size_mb:.2f} MB")
        if size_mb < 10:  # YOLOv7 weights should be at least tens of MB
            print("Warning: yolov7.weights is suspiciously small")
    else:
        print("Error: yolov7.weights does not exist")
        success = False
    
    if os.path.exists(yolo_cfg_path):
        size_kb = os.path.getsize(yolo_cfg_path) / 1024
        print(f"yolov7.cfg exists, size: {size_kb:.2f} KB")
        if size_kb < 10:  # Config file should be at least several KB
            print("Warning: yolov7.cfg is suspiciously small")
    else:
        print("Error: yolov7.cfg does not exist")
        success = False
    
    if os.path.exists(coco_names_path):
        print(f"coco.names exists")
    else:
        print("Error: coco.names does not exist")
        success = False
    
    # Display content of models directory
    print("\nContents of models directory:")
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")
    
    if success:
        print("\nSuccess! All files downloaded")
        return 0
    else:
        print("\nError: Some files couldn't be downloaded")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 