import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Define paths based on your setup
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
IMAGE_FOLDER = os.environ.get('IMAGE_FOLDER', os.path.join(os.path.dirname(__file__), 'test_images'))
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', os.path.join(os.path.dirname(__file__), 'output'))

# Ensure directories exist
for directory in [MODELS_DIR, IMAGE_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Model paths
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', os.path.join(MODELS_DIR, 'yolov7.weights'))
YOLO_CONFIG = os.environ.get('YOLO_CONFIG', os.path.join(MODELS_DIR, 'yolov7.cfg'))
COCO_NAMES = os.environ.get('COCO_NAMES', os.path.join(MODELS_DIR, 'coco.names'))

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
def detect_vehicles(image_path, net, output_layers, coco_classes, conf_threshold=0.15, nms_threshold=0.5):
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

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = coco_classes[class_ids[i]]  
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {vehicle_count+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            vehicle_count += 1

    # Save detected image
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path).replace(".jpg", "_detected.jpg"))
    cv2.imwrite(output_path, image)

    return vehicle_count, output_path

# Streamlit UI
def main():
    st.set_page_config(page_title="Vehicle Detection", layout="wide")

    st.sidebar.header("🚗 AI Traffic Monitoring")
    st.sidebar.write("Upload an image to detect vehicles.")

    # Load Model
    net, output_layers = load_yolo()
    coco_classes = load_coco_classes()
    
    # Check if model is loaded
    if net is None:
        st.error("⚠️ Model not loaded. Please check the model files.")
        st.info("You need to upload the YOLOv7 weights and configuration files to the models directory.")
        return

    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Convert uploaded image to OpenCV format
        image_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the image
        vehicle_count, output_path = detect_vehicles(image_path, net, output_layers, coco_classes)

        # Display results
        if vehicle_count is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if output_path:
                with col2:
                    st.image(output_path, caption=f"Processed Image - {vehicle_count} Vehicles", use_column_width=True)

            st.success(f"✅ Vehicles Detected: {vehicle_count}")
            st.download_button("📥 Download Processed Image", data=open(output_path, "rb").read(), file_name="detected_image.jpg")

if __name__ == "__main__":
    main()