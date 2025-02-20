import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
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

# Load lightweight YOLOv8 model
model = YOLO('yolov8n.pt')
logger.info("Loaded YOLOv8n model for object detection and weapon recognition")

# Define focused weapon classes (only sharp objects and guns)
WEAPON_CLASSES = {
    'knife': 0.65,      # Lower threshold for better detection
    'scissors': 0.7,
    'gun': 0.5,         # Lower threshold for guns
    'firearm': 0.5,
    'pistol': 0.5,
    'rifle': 0.5,
    'shotgun': 0.5,
    'handgun': 0.5,
    'sword': 0.6
}

# Common object classes to track (modify as needed)
COMMON_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'bottle', 'cup', 'bowl', 'chair',
    'laptop', 'cell phone', 'book', 'clock', 'tv', 'keyboard'
]

# COCO class IDs
PERSON_CLASS_ID = 0  # In COCO dataset, person is class 0

# Simplified anomaly detection
class EnhancedAnomalyDetector:
    def __init__(self, 
                 window_size=120,
                 velocity_threshold=2.5,         # Increased from 1.8
                 count_threshold=2.5,            # Increased from 2.0
                 min_data_points=30,
                 position_weight=0.5,            # Adjusted from 0.7
                 count_weight=0.2,               # Adjusted from 0.3
                 acceleration_weight=0.3):        # New weight for sudden movements
        self.positions = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.area_coverage = deque(maxlen=window_size)
        self.person_distances = deque(maxlen=window_size)  # New: track distances between people
        self.velocity_threshold = velocity_threshold
        self.count_threshold = count_threshold
        self.min_data_points = min_data_points
        self.position_weight = position_weight
        self.count_weight = count_weight
        self.acceleration_weight = acceleration_weight
        self.is_ready = False
        self.baseline_established = False
        self.baseline_velocity_mean = None
        self.baseline_velocity_std = None
        self.baseline_count_mean = None 
        self.baseline_count_std = None
        self.baseline_distance_mean = None
        self.baseline_distance_std = None
        
    def calculate_person_distances(self, people_positions):
        """Calculate minimum distances between all pairs of people"""
        if len(people_positions) < 2:
            return None
        
        distances = []
        for i in range(len(people_positions)):
            for j in range(i + 1, len(people_positions)):
                dist = np.linalg.norm(np.array(people_positions[i]) - np.array(people_positions[j]))
                distances.append(dist)
        return np.min(distances)  # Return minimum distance between any pair
        
    def update(self, detections, frame_shape):
        # Extract person positions and calculate coverage
        people_positions = []
        people_count = 0
        total_area = 0
        frame_area = frame_shape[0] * frame_shape[1]
        
        for detection in detections:
            boxes = detection.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0].item())
                if cls == PERSON_CLASS_ID:
                    people_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_area = (x2 - x1) * (y2 - y1)
                    total_area += box_area
                    
                    center_x = (x1 + x2) / 2 / frame_shape[1]
                    center_y = (y1 + y2) / 2 / frame_shape[0]
                    people_positions.append((center_x, center_y))
        
        # Store count and area coverage
        self.counts.append(people_count)
        self.area_coverage.append(total_area / frame_area if frame_area > 0 else 0)
        
        # Calculate and store minimum distances between people
        min_distance = self.calculate_person_distances(people_positions)
        self.person_distances.append(min_distance)
        
        if not people_positions:
            self.positions.append(None)
            self.velocities.append(None)
            return False
        
        # Calculate average position and velocity
        avg_pos = np.mean(people_positions, axis=0)
        self.positions.append(avg_pos)
        
        if len(self.positions) < 2 or self.positions[-2] is None:
            self.velocities.append(None)
        else:
            velocity = np.linalg.norm(np.array(avg_pos) - np.array(self.positions[-2]))
            self.velocities.append(velocity)
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.positions) >= self.min_data_points:
            self.establish_baseline()
            self.baseline_established = True
            self.is_ready = True
        
        return self.detect_anomaly() if self.is_ready else False
    
    def establish_baseline(self):
        """Establish baseline statistics from initial observations"""
        valid_velocities = [v for v in self.velocities if v is not None]
        valid_counts = list(self.counts)
        valid_distances = [d for d in self.person_distances if d is not None]
        
        if len(valid_velocities) < self.min_data_points // 2:
            return
        
        # Calculate baseline statistics with robust statistics
        self.baseline_velocity_mean = np.median(valid_velocities)  # Use median instead of mean
        self.baseline_velocity_std = np.percentile(valid_velocities, 75) - np.percentile(valid_velocities, 25)
        
        self.baseline_count_mean = np.median(valid_counts)
        self.baseline_count_std = max(np.std(valid_counts), 1.0)
        
        if valid_distances:
            self.baseline_distance_mean = np.median(valid_distances)
            self.baseline_distance_std = max(np.std(valid_distances), 0.05)
    
    def detect_anomaly(self):
        """Enhanced anomaly detection focusing on sudden movements and fight-like scenarios"""
        if not self.is_ready:
            return False
        
        # Calculate velocity score
        if self.velocities[-1] is None:
            velocity_score = 0
        else:
            # Use exponential scaling for sudden movements
            velocity_diff = abs(self.velocities[-1] - self.baseline_velocity_mean)
            velocity_score = np.exp(velocity_diff / self.baseline_velocity_std) - 1
        
        # Calculate count score
        count_diff = abs(self.counts[-1] - self.baseline_count_mean)
        count_score = count_diff / self.baseline_count_std
        
        # Calculate acceleration (sudden changes in velocity)
        recent_velocities = [v for v in list(self.velocities)[-5:] if v is not None]
        if len(recent_velocities) >= 3:
            acceleration = np.diff(recent_velocities)
            acceleration_score = np.max(np.abs(acceleration)) * 2
        else:
            acceleration_score = 0
        
        # Calculate distance score (for detecting close interactions/fights)
        if self.person_distances[-1] is not None and self.baseline_distance_mean is not None:
            distance_score = max(0, (self.baseline_distance_mean - self.person_distances[-1]) / self.baseline_distance_std)
        else:
            distance_score = 0
        
        # Combined anomaly score with weighted factors
        anomaly_score = (
            velocity_score * self.position_weight +
            count_score * self.count_weight +
            acceleration_score * self.acceleration_weight +
            distance_score * 0.2  # Weight for distance score
        )
        
        # Dynamic threshold based on recent history
        dynamic_threshold = max(
            self.velocity_threshold,
            self.count_threshold
        ) * (1 + 0.5 * acceleration_score)  # Increase threshold for sustained activity
        
        return anomaly_score > dynamic_threshold
    
    def get_debug_info(self):
        """Return current detection metrics for debugging"""
        if not self.is_ready:
            return "Baseline not yet established"
            
        latest_velocity = self.velocities[-1] if self.velocities[-1] is not None else 0
        latest_distance = self.person_distances[-1] if self.person_distances[-1] is not None else None
        
        return {
            "velocity": latest_velocity,
            "velocity_zscore": abs(latest_velocity - self.baseline_velocity_mean) / self.baseline_velocity_std if self.baseline_velocity_std else 0,
            "person_count": self.counts[-1],
            "count_zscore": abs(self.counts[-1] - self.baseline_count_mean) / self.baseline_count_std if self.baseline_count_std else 0,
            "min_person_distance": latest_distance,
            "baseline_samples": len(self.positions)
        }

# Initialize the enhanced anomaly detector
anomaly_detector = EnhancedAnomalyDetector()

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
    threading.Timer(10, reset_alert).start()

def reset_alert():
    global alert_status
    alert_status = False
    logger.info("Alert status reset")

# Object detection function
def detect_objects_and_weapons(frame, results):
    weapons_detected = []
    objects_detected = []
    
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id].lower()
            
            # Check if the detected object is a weapon
            if cls_name in WEAPON_CLASSES and conf >= WEAPON_CLASSES[cls_name]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                weapons_detected.append({
                    'class': cls_name,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
            
            # Check if it's a common object and has reasonable confidence
            elif cls_name in COMMON_OBJECTS and conf >= 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                objects_detected.append({
                    'class': cls_name,
                    'confidence': conf,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
    
    return weapons_detected, objects_detected

# Video processing function (optimized)
def process_frame(frame):
    global current_frame
    
    # Skip frames to improve performance
    if hasattr(process_frame, 'frame_count'):
        process_frame.frame_count += 1
        if process_frame.frame_count % 2 != 0:  # Process every other frame
            return frame
    else:
        process_frame.frame_count = 0
    
    # Make a copy of the frame for display
    display_frame = frame.copy()
    
    # Run lightweight YOLOv8 detection
    results = model(frame, imgsz=416, verbose=False)  # Reduced image size for faster processing
    
    # Update anomaly detection
    anomaly_detected = anomaly_detector.update(results, frame.shape)
    if anomaly_detected:
        cv2.putText(display_frame, "ANOMALY DETECTED", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Detect weapons and objects using the same model results
    weapons_detected, objects_detected = detect_objects_and_weapons(frame, results)
    
    # Draw weapon detections (red)
    for weapon in weapons_detected:
        x1, y1, x2, y2 = weapon['box']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display_frame, f"{weapon['class']}: {weapon['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw object detections (blue)
    for obj in objects_detected:
        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(display_frame, f"{obj['class']}: {obj['confidence']:.2f}", 
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Add alert status to frame
    if alert_status:
        cv2.putText(display_frame, "ALERT ACTIVE", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Add object count info
    cv2.putText(display_frame, f"Objects: {len(objects_detected)}", (10, display_frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
    
    # Initialize webcam with lower resolution for better performance
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Lower resolution
    
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
            time.sleep(0.03)  # Longer delay to reduce CPU load
            
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
        time.sleep(0.05)  # 20 FPS stream (reduced for performance)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def get_alert_status():
    return jsonify({"alert": alert_status})

# API endpoint to get detected objects
@app.route('/objects')
def get_objects():
    # This endpoint could be enhanced to track objects over time
    if current_frame is not None:
        results = model(current_frame, imgsz=416, verbose=False)
        weapons, objects = detect_objects_and_weapons(current_frame, results)
        
        return jsonify({
            "weapons": [{"class": w["class"], "confidence": w["confidence"]} for w in weapons],
            "objects": [{"class": o["class"], "confidence": o["confidence"]} for o in objects]
        })
    else:
        return jsonify({"weapons": [], "objects": []})

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

    