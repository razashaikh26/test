import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Set page config first - must be the first Streamlit command
st.set_page_config(page_title="Vehicle Detection", layout="wide")

# Define paths based on your setup
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
IMAGE_FOLDER = os.environ.get('IMAGE_FOLDER', os.path.join(os.path.dirname(__file__), 'test_images'))
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', os.path.join(os.path.dirname(__file__), 'output'))

# Model paths
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', os.path.join(MODELS_DIR, 'yolov7.weights'))
YOLO_CONFIG = os.environ.get('YOLO_CONFIG', os.path.join(MODELS_DIR, 'yolov7.cfg'))
COCO_NAMES = os.environ.get('COCO_NAMES', os.path.join(MODELS_DIR, 'coco.names'))

# Ensure directories exist
for directory in [MODELS_DIR, IMAGE_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Load YOLO model
@st.cache_resource
def load_yolo():
    try:
        net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error("Please ensure the model files (weights and config) are uploaded to the models directory.")
        return None, None

# Load COCO class names
@st.cache_resource
def load_coco_classes():
    try:
        with open(COCO_NAMES, "r") as f:
            return f.read().strip().split("\n")
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        # Return a default list of vehicle classes
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

# Vehicle Detection Function
def detect_vehicles(image_path, net, output_layers, coco_classes, conf_threshold=0.3, nms_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Error: Could not load image {image_path}")
        return None, None
    
    height, width = image.shape[:2]
    
    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    vehicle_classes = {2, 3, 5, 7}  # Car, Motorcycle, Bus, Truck

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and class_id in vehicle_classes:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    vehicle_count = 0
    
    vehicle_colors = {
        2: (0, 255, 0),    # Car - Green
        3: (0, 165, 255),  # Motorcycle - Orange
        5: (255, 0, 0),    # Bus - Blue
        7: (128, 0, 128)   # Truck - Purple
    }

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = coco_classes[class_ids[i]]
            color = vehicle_colors.get(class_ids[i], (0, 255, 0))
            confidence_score = confidences[i]
            
            # Draw rectangle and label
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence_score:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            vehicle_count += 1

    # Save detected image
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path).replace(".jpg", "_detected.jpg"))
    cv2.imwrite(output_path, image)

    return vehicle_count, output_path

# Streamlit UI
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=80)
    st.sidebar.title("AI Traffic Monitoring")
    st.sidebar.markdown("---")
    st.sidebar.write("Upload an image to detect vehicles using YOLOv7.")
    
    # Main content
    st.title("🚗 Vehicle Detection System")
    st.write("This application uses YOLOv7 to detect and count vehicles in images.")

    # Load Model
    net, output_layers = load_yolo()
    coco_classes = load_coco_classes()
    
    # Check if model is loaded
    if net is None:
        st.error("⚠️ Model not loaded. Please check the model files.")
        st.info("You need to upload the YOLOv7 weights and configuration files to the models directory.")
        return

    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Adjust how confident the model needs to be to detect a vehicle"
    )

    if uploaded_file is not None:
        # Convert uploaded image to OpenCV format
        image_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the image
        vehicle_count, output_path = detect_vehicles(image_path, net, output_layers, coco_classes, conf_threshold=conf_threshold)

        # Display results
        if vehicle_count is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_column_width=True)
            if output_path:
                with col2:
                    st.subheader(f"Processed Image ({vehicle_count} Vehicles)")
                    st.image(output_path, use_column_width=True)

            st.success(f"✅ Successfully detected {vehicle_count} vehicles in the image")
            st.download_button(
                "📥 Download Processed Image", 
                data=open(output_path, "rb").read(), 
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )
    else:
        # Display sample images when no file is uploaded
        st.info("👈 Upload an image using the sidebar to detect vehicles")
        st.markdown("### How it works")
        st.write("""
        1. Upload an image containing traffic scenes
        2. The AI model detects vehicles such as cars, trucks, buses, and motorcycles
        3. View the detected vehicles with bounding boxes
        4. Download the processed image
        """)

if __name__ == "__main__":
    main()