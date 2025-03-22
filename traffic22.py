import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Set page config first - must be the first Streamlit command
st.set_page_config(page_title="Vehicle Detection", layout="wide")

# Define paths based on environment variables or defaults
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(SCRIPT_DIR, "models"))
YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', os.path.join(MODELS_DIR, "yolov8n.pt"))
COCO_NAMES = os.environ.get('COCO_NAMES', os.path.join(MODELS_DIR, "coco.names"))
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', os.path.join(SCRIPT_DIR, "output"))
TEMP_FOLDER = os.environ.get('TEMP_FOLDER', os.path.join(SCRIPT_DIR, "temp"))

# Create necessary directories
for directory in [MODELS_DIR, OUTPUT_FOLDER, TEMP_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Load YOLO model
@st.cache_resource
def load_yolo():
    # Check if YOLO file exists
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"YOLO model file not found at: {YOLO_MODEL_PATH}")
        st.info("Please ensure the YOLOv8 model file is uploaded to the models directory.")
        return None
    
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

# Load COCO class names
@st.cache_resource
def load_coco_classes():
    if not os.path.exists(COCO_NAMES):
        st.error(f"COCO names file not found at: {COCO_NAMES}")
        # Default vehicle classes
        return ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    
    with open(COCO_NAMES, "r") as f:
        return f.read().strip().split("\n")

# Initialize session state variables
if 'total_vehicles' not in st.session_state:
    st.session_state.total_vehicles = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Vehicle Detection Function
def detect_vehicles(frame, model, coco_classes, conf_threshold=0.25):
    if frame is None or model is None:
        return frame, 0
    
    # Define vehicle classes for YOLOv8
    vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
    
    # Run inference
    results = model(frame, conf=conf_threshold)
    
    # Process results
    vehicle_count = 0
    
    # Extract detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            if cls in vehicle_classes:
                vehicle_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0].item())
                label = coco_classes[cls] if cls < len(coco_classes) else f"Class {cls}"
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, vehicle_count

# Calculate FPS
def update_fps():
    current_time = time.time()
    elapsed_time = current_time - st.session_state.last_time
    st.session_state.frame_count += 1
    
    if elapsed_time > 1:
        st.session_state.fps = st.session_state.frame_count / elapsed_time
        st.session_state.frame_count = 0
        st.session_state.last_time = current_time

# Detect on single image
def process_uploaded_image(image_bytes, model, coco_classes, conf_threshold=0.25):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_frame, vehicle_count = detect_vehicles(frame, model, coco_classes, conf_threshold)
    
    # Save the processed image
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{int(time.time())}.jpg")
    cv2.imwrite(output_path, processed_frame)
    
    return processed_frame, vehicle_count, output_path

# Streamlit UI
def main():    
    st.sidebar.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=80)
    st.sidebar.title("AI Traffic Monitoring")
    st.sidebar.markdown("---")
    
    # Main content
    st.title("🚗 Advanced Vehicle Detection System")
    st.write("This application uses YOLOv8 to detect and count vehicles in images and videos.")
    
    # Check if required model file exists
    if not os.path.exists(YOLO_MODEL_PATH):
        st.warning(f"YOLO model file not found at: {YOLO_MODEL_PATH}")
        st.info("Please upload the YOLOv8 model file to the models directory.")
    
    # Load model
    model = load_yolo()
    coco_classes = load_coco_classes()
    
    # Input source selection
    input_source = st.sidebar.radio("Select Input Source", ["Image Upload", "Webcam", "IP Camera", "Video File"])
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    
    # Check if running on Render
    is_render = 'RENDER' in os.environ
    
    if input_source == "Image Upload":
        st.sidebar.write("Upload an image to detect vehicles")
        uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process image
            image_bytes = uploaded_file.getvalue()
            processed_frame, vehicle_count, output_path = process_uploaded_image(image_bytes, model, coco_classes, conf_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)
            with col2:
                st.subheader(f"Processed Image ({vehicle_count} Vehicles)")
                st.image(processed_frame, channels="BGR", use_column_width=True)
            
            st.success(f"✅ Successfully detected {vehicle_count} vehicles in the image")
            st.download_button("📥 Download Processed Image", 
                                data=open(output_path, "rb").read(), 
                                file_name="detected_vehicles.jpg",
                                mime="image/jpeg")
    
    else:  # Video sources (Webcam, IP Camera, Video File)
        # UI placeholders
        stframe = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            vehicle_count_placeholder = st.empty()
        with metrics_col2:
            total_vehicle_placeholder = st.empty()
        with metrics_col3:
            fps_placeholder = st.empty()
        
        # Check if webcam is available on Render
        if input_source == "Webcam" and is_render:
            st.warning("⚠️ Webcam access may not be available on the Render platform.")
            st.info("For webcam access, please run this application locally.")
        
        # Camera/video setup
        cam_id = None
        
        if input_source == "Webcam":
            cap_options = {
                "Default Webcam": 0,
                "External Camera": 1
            }
            selected_cam = st.sidebar.selectbox("Select Camera", list(cap_options.keys()))
            cam_id = cap_options[selected_cam]
            
        elif input_source == "IP Camera":
            ip_camera_url = st.sidebar.text_input("Enter IP Camera URL")
            if ip_camera_url:
                cam_id = ip_camera_url
            else:
                st.sidebar.warning("Please enter a valid IP camera URL.")
                
        elif input_source == "Video File":
            video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            if video_file is not None:
                # Save the uploaded file to a temporary location
                temp_path = os.path.join(TEMP_FOLDER, "temp_video.mp4")
                with open(temp_path, "wb") as f:
                    f.write(video_file.read())
                cam_id = temp_path
            else:
                st.sidebar.warning("Please upload a video file.")
        
        # Control buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_button = st.button("Start Detection")
        with col2:
            stop_button = st.button("Stop Detection")
            
        if stop_button:
            st.session_state.processing = False
            st.sidebar.success("Detection stopped")
        
        # Process video when start button is clicked
        if start_button and cam_id is not None and not st.session_state.processing:
            st.session_state.processing = True
            
            if model is None:
                st.error("YOLO model could not be loaded. Please check the model file exists at the specified path.")
                st.session_state.processing = False
                return
                
            # Open video capture
            cap = cv2.VideoCapture(cam_id)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                st.error(f"Error: Could not open video source. Please check if the source is valid and accessible.")
                st.session_state.processing = False
                return
            
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Reset counters
            st.session_state.total_vehicles = 0
            st.session_state.frame_count = 0
            st.session_state.last_time = time.time()
            st.session_state.fps = 0
            
            # Display initial empty metrics
            vehicle_count_placeholder.metric("Current Vehicles", 0)
            total_vehicle_placeholder.metric("Total Vehicles", 0)
            fps_placeholder.metric("FPS", "0.0")
            
            # Main processing loop
            while st.session_state.processing:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if input_source == "Video File":
                        st.info("Video processing complete")
                    else:
                        st.error("Error: Could not read frame from video source.")
                    break
                
                # Process frame
                processed_frame, vehicle_count = detect_vehicles(frame, model, coco_classes, conf_threshold)
                
                # Update metrics
                st.session_state.total_vehicles += vehicle_count
                update_fps()
                
                # Display frame and metrics
                stframe.image(processed_frame, channels="BGR", use_column_width=True)
                vehicle_count_placeholder.metric("Current Vehicles", vehicle_count)
                total_vehicle_placeholder.metric("Total Vehicles", st.session_state.total_vehicles)
                fps_placeholder.metric("FPS", f"{st.session_state.fps:.1f}")
                
                # Add a slight delay to control frame rate
                time.sleep(0.01)
                
                # Check if stop button was pressed
                if not st.session_state.processing:
                    break
            
            # Release camera
            cap.release()
            st.session_state.processing = False
            st.sidebar.success("Detection stopped")
    
    # Instructions when no input is provided
    if input_source == "Image Upload" and "uploaded_file" not in locals():
        st.info("👈 Upload an image using the sidebar to detect vehicles")
        st.markdown("### How it works")
        st.write("""
        1. Upload an image containing traffic scenes
        2. The AI model detects vehicles such as cars, trucks, buses, and motorcycles
        3. View the detected vehicles with bounding boxes
        4. Download the processed image
        """)
    
    # Information about different detection modes
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Modes")
    st.sidebar.markdown("""
    - **Image Upload**: Process a single image
    - **Webcam**: Live detection from a connected camera
    - **IP Camera**: Connect to an IP camera stream
    - **Video File**: Process an uploaded video file
    """)

if __name__ == "__main__":
    main()
