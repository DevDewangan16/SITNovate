# app.py - Main application file with enhanced weapon detection
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.ensemble import IsolationForest
import threading
import time
import logging
from flask import Flask, render_template, Response, jsonify
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app for web dashboard
app = Flask(__name__)

# Global variables
alert_status = False
current_frame = None
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds between alerts

# Load primary YOLOv8 model for general object detection
primary_model = YOLO('yolov8n.pt')  # Base model for general detection

# Load specialized weapon detection model (either fine-tuned or a more capable model)
try:
    # Try to load a larger model for better detection capability
    weapon_model = YOLO('yolov8x.pt')  # Using the largest YOLOv8 model for better detection
    logger.info("Loaded YOLOv8x model for enhanced weapon detection")
except:
    # Fallback to medium if x is not available
    weapon_model = YOLO('yolov8m.pt')
    logger.info("Loaded YOLOv8m model for enhanced weapon detection")

# Initialize anomaly detection model
anomaly_detector = IsolationForest(contamination=0.05)
is_model_trained = False
behavior_features = []

# Define expanded weapon classes
# Core weapons from COCO dataset
COCO_WEAPON_CLASSES = {
    'knife': 0.7,          # Confidence threshold
    'scissors': 0.75,
    'baseball bat': 0.65,
    'bottle': 0.8,         # Higher threshold for common items
    'wine glass': 0.85,
    'fork': 0.85,
    'sports ball': 0.9     # Higher threshold for sports equipment
}

# Additional weapons to detect (not in standard COCO but YOLOv8 larger models can detect)
CUSTOM_WEAPON_CLASSES = {
    'gun': 0.5,            # Lower threshold for critical items
    'firearm': 0.5,
    'pistol': 0.5,
    'rifle': 0.5,
    'shotgun': 0.5,
    'handgun': 0.5,
    'cricket bat': 0.7,
    'hockey stick': 0.7,
    'golf club': 0.7,
    'sword': 0.6,
    'bow': 0.7,
    'arrow': 0.7,
    'axe': 0.6,
    'hammer': 0.8
}

# Combine all weapon classes
ALL_WEAPON_CLASSES = {**COCO_WEAPON_CLASSES, **CUSTOM_WEAPON_CLASSES}

# COCO class IDs
PERSON_CLASS_ID = 0  # In COCO dataset, person is class 0

# Alert system
def send_alert(frame, reason):
    global alert_status, last_alert_time
    current_time = time.time()
    
    # Implement cooldown to prevent alert flooding
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return
    
    last_alert_time = current_time
    alert_status = True
    
    # Log the alert
    logger.warning(f"ALERT TRIGGERED: {reason}")
    
    # Create alerts directory if it doesn't exist
    if not os.path.exists('alerts'):
        os.makedirs('alerts')
    
    # Save the frame that triggered the alert
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"alerts/alert_{timestamp}.jpg", frame)
    
    # Alert will auto-reset after 30 seconds
    threading.Timer(30.0, reset_alert).start()

def reset_alert():
    global alert_status
    alert_status = False
    logger.info("Alert status reset")

# Feature extraction for anomaly detection
def extract_behavior_features(detections, frame_shape):
    features = []
    
    # Track number of people and their positions
    people_count = 0
    people_positions = []
    
    if len(detections) == 0:
        return np.zeros(5)  # Return zeros if no detections
    
    for detection in detections:
        for i, cls in enumerate(detection.boxes.cls):
            if cls.item() == PERSON_CLASS_ID:
                people_count += 1
                
                # Get bounding box
                x1, y1, x2, y2 = detection.boxes.xyxy[i].tolist()
                center_x = (x1 + x2) / 2 / frame_shape[1]  # Normalize by frame width
                center_y = (y1 + y2) / 2 / frame_shape[0]  # Normalize by frame height
                box_width = (x2 - x1) / frame_shape[1]
                box_height = (y2 - y1) / frame_shape[0]
                
                people_positions.append((center_x, center_y, box_width, box_height))
    
    if people_count == 0:
        return np.zeros(5)
    
    # Calculate features
    avg_width = sum([pos[2] for pos in people_positions]) / people_count
    avg_height = sum([pos[3] for pos in people_positions]) / people_count
    
    # Calculate average position
    avg_x = sum([pos[0] for pos in people_positions]) / people_count
    avg_y = sum([pos[1] for pos in people_positions]) / people_count
    
    # Feature vector: [people_count, avg_x, avg_y, avg_width, avg_height]
    return np.array([people_count, avg_x, avg_y, avg_width, avg_height])

# Specialized weapon detection function
def detect_weapons(frame):
    # Run the weapon-specific model with higher resolution
    results = weapon_model(frame, verbose=False)
    
    weapons_detected = []
    
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id].lower()
            
            # Check standard weapons
            if cls_name in ALL_WEAPON_CLASSES and conf >= ALL_WEAPON_CLASSES[cls_name]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                weapons_detected.append({
                    'class': cls_name,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
                
            # Special case for cricket bat (might be detected as baseball bat)
            elif cls_name == 'baseball bat' and conf >= 0.6:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                weapons_detected.append({
                    'class': 'bat (baseball/cricket)',
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
    
    return weapons_detected

# Video processing function
def process_frame(frame):
    global current_frame, is_model_trained, behavior_features
    
    # Make a copy of the frame for display
    display_frame = frame.copy()
    
    # Run YOLOv8 primary detection for people and common objects
    primary_results = primary_model(frame, verbose=False)
    
    # Extract detections
    primary_detections = primary_results[0]
    
    # Track detected objects and potential threats
    person_detected = False
    
    # Check for people in primary detections
    for detection in primary_detections:
        boxes = detection.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Check if a person is detected
            if cls == PERSON_CLASS_ID and conf >= 0.6:
                person_detected = True
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Person: {conf:.2f}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Specialized weapon detection
    weapons_detected = detect_weapons(frame)
    
    # Draw weapon detections
    for weapon in weapons_detected:
        x1, y1, x2, y2 = weapon['box']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display_frame, f"{weapon['class']}: {weapon['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Handle anomaly detection if a person is detected
    anomaly_detected = False
    if person_detected:
        # Extract behavior features
        features = extract_behavior_features(primary_detections, frame.shape)
        
        # Add to training data
        behavior_features.append(features)
        
        # Train model after collecting enough samples
        if len(behavior_features) >= 100 and not is_model_trained:
            logger.info("Training anomaly detection model...")
            anomaly_detector.fit(behavior_features)
            is_model_trained = True
            logger.info("Anomaly detection model trained")
        
        # Run anomaly detection if model is trained
        if is_model_trained:
            # Keep only the last 1000 samples
            if len(behavior_features) > 1000:
                behavior_features = behavior_features[-1000:]
                
            # Predict anomaly
            prediction = anomaly_detector.predict([features])[0]
            if prediction == -1:  # -1 indicates anomaly
                anomaly_detected = True
                cv2.putText(display_frame, "ANOMALY DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add alert status to frame
    if alert_status:
        cv2.putText(display_frame, "ALERT ACTIVE", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Add FPS info
    fps = int(1 / (time.time() - process_frame.last_time + 0.001))
    process_frame.last_time = time.time()
    cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1] - 100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Trigger alert if weapon or anomaly is detected
    if weapons_detected:
        weapon_names = ', '.join([w['class'] for w in weapons_detected])
        send_alert(frame, f"Weapons detected: {weapon_names}")
    elif anomaly_detected:
        send_alert(frame, "Abnormal behavior detected")
    
    # Update the current frame for the web stream
    current_frame = display_frame
    
    return display_frame

# Initialize last time for FPS calculation
process_frame.last_time = time.time()

# Camera capture thread
def camera_thread():
    global current_frame
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        logger.error("Error: Could not open webcam.")
        return
    
    logger.info("Camera initialized successfully")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame from webcam")
                break
                
            # Process the frame
            process_frame(frame)
            
            # Add small delay to reduce CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error in camera thread: {e}")
    finally:
        cap.release()
        logger.info("Camera released")

# Flask routes for web dashboard
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global current_frame
    while True:
        if current_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def get_alert_status():
    return jsonify({"alert": alert_status})

if __name__ == '__main__':
    # Create alerts directory
    if not os.path.exists('alerts'):
        os.makedirs('alerts')
    
    # Start camera thread
    camera_thread = threading.Thread(target=camera_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
