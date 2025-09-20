#!/usr/bin/env python3
"""
motion_triggered_ncnn_yolo.py
- OpenCV capture + frame-diff motion detection
- Run Ultralytics YOLO on motion frames (NCNN-exported model)
- Print simplified detections to terminal
"""

import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO  # ultralytics handles exported formats including NCNN
import sys
import os
import threading
from collections import deque

# Add the camserve module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'camserve'))
from camserve import SimpleCameraServer

# Add the services module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'services'))
from event_logger import get_logger

# --- Configuration ---
# Update this path to the correct absolute or relative path of your NCNN model directory.
NCNN_MODEL_PATH = "model_ncnn_model"
CAMERA_INDEX = 0                         # or path to RTSP/HTTP stream
IMG_SIZE = 320                           # model expects 320x320 per metadata
CONF_THRESH = 0.50                       # min confidence to consider detection
MOTION_AREA_THRESH = 50                  # min contour area to count as motion (tuned for sensitivity)
DETECTION_COOLDOWN = 1.0                 # seconds between running detector when motion persists
USE_HALF = False                         # NCNN inference may not support float16; follow your export
MAX_DETECTIONS_PER_FRAME = 10
DISPLAY = True                           # set True for an OpenCV display window

# Video recording configuration
VIDEO_DURATION = 10.0                    # seconds to record when motion is detected
FPS = 30                                 # frames per second for video recording
SAVED_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'saved')  # folder to save videos and images

# mapping from model class ids -> names
CLASS_NAMES = {0: "Fire", 1: "Smoke"}

# Initialize event logger
event_logger = get_logger()

# --- Video Recording and Image Saving Classes ---
class VideoRecorder:
    """Handles video recording when motion is detected"""
    
    def __init__(self, fps=30, duration=10.0, saved_folder="saved"):
        self.fps = fps
        self.duration = duration
        self.saved_folder = saved_folder
        self.is_recording = False
        self.video_writer = None
        self.frame_buffer = deque(maxlen=int(fps * duration))
        self.recording_thread = None
        
        # Ensure saved folder exists
        os.makedirs(saved_folder, exist_ok=True)
    
    def start_recording(self, frame):
        """Start recording a video from the current frame"""
        if self.is_recording:
            return None  # Already recording
        
        self.is_recording = True
        self.frame_buffer.clear()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_video_{timestamp}.mp4"
        filepath = os.path.join(self.saved_folder, filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        
        # Add current frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_frames, daemon=True)
        self.recording_thread.start()
        
        print(f"🎥 Started recording: {filename}")
        return filepath
    
    def add_frame(self, frame):
        """Add a frame to the current recording"""
        if self.is_recording:
            self.frame_buffer.append(frame.copy())
    
    def _record_frames(self):
        """Record frames in a separate thread"""
        start_time = time.time()
        
        while self.is_recording and (time.time() - start_time) < self.duration:
            if self.frame_buffer:
                frame = self.frame_buffer.popleft()
                self.video_writer.write(frame)
            time.sleep(1.0 / self.fps)
        
        self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and save the video"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Write remaining frames
        while self.frame_buffer:
            frame = self.frame_buffer.popleft()
            self.video_writer.write(frame)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        print("🎥 Video recording completed")


class ImageSaver:
    """Handles saving images with bounding boxes for fire/smoke detections"""
    
    def __init__(self, saved_folder="saved"):
        self.saved_folder = saved_folder
        
        # Ensure saved folder exists
        os.makedirs(saved_folder, exist_ok=True)
    
    def save_detection_image(self, frame, detections, class_names):
        """Save an image with bounding boxes drawn for detections"""
        if not detections:
            return None
        
        # Create a copy of the frame to draw on
        frame_with_boxes = frame.copy()
        
        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2 = detection['coordinates']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame_with_boxes, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            
            # Draw label text
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(self.saved_folder, filename)
        
        # Save the image
        cv2.imwrite(filepath, frame_with_boxes)
        print(f"📸 Saved detection image: {filename}")
        return filepath


# Initialize video recorder and image saver
video_recorder = VideoRecorder(fps=FPS, duration=VIDEO_DURATION, saved_folder=SAVED_FOLDER)
image_saver = ImageSaver(saved_folder=SAVED_FOLDER)

# --- Load model ---
try:
    model = YOLO(NCNN_MODEL_PATH)
    print("Model loaded successfully:", NCNN_MODEL_PATH)
except Exception as e:
    raise SystemExit(f"Error loading model: {e}")

# --- Video capture setup ---
camera_server = SimpleCameraServer(camera_id=CAMERA_INDEX)
if not camera_server.start():
    raise SystemExit(f"Cannot open camera/source: {CAMERA_INDEX}")

# Wait for first frame to be available
print("Waiting for camera frames...")
while True:
    frame = camera_server.get_frame()
    if frame is not None:
        break
    time.sleep(0.1)

# convert to grayscale + blur for motion detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
background = gray.astype("float")

last_detection_time = 0.0

print("Starting main loop. Press 'q' in the display window or Ctrl+C to quit.")

try:
    while True:
        frame = camera_server.get_frame()
        if frame is None:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        # --- Motion detection (frame differencing with running average) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        cv2.accumulateWeighted(gray, background, 0.5)
        background_uint8 = cv2.convertScaleAbs(background)
        diff = cv2.absdiff(gray, background_uint8)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False
        motion_area_total = 0
        for c in contours:
            area = cv2.contourArea(c)
            motion_area_total += area
            if area >= MOTION_AREA_THRESH:
                motion_found = True
                if DISPLAY:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw total motion area on display
        if DISPLAY:
            cv2.putText(frame, f"Total Motion Area: {int(motion_area_total)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame to video recording if currently recording
        if video_recorder.is_recording:
            video_recorder.add_frame(frame)

        now = time.time()
        if motion_found and (now - last_detection_time) >= DETECTION_COOLDOWN:
            last_detection_time = now
            
            # Start video recording if not already recording
            video_file_path = None
            if not video_recorder.is_recording:
                video_file_path = video_recorder.start_recording(frame)
            else:
                # Add frame to current recording
                video_recorder.add_frame(frame)
            
            # Log motion detection event with file location
            event_logger.log_motion_detection(video_file_path)

            # Use the 320x320 smframe for YOLO detection
            smframe = camera_server.get_smframe()
            if smframe is not None:
                results = model(smframe, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
            else:
                print("Warning: No smframe available for detection")
                continue

            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    # Collect all detections for image saving
                    detections = []
                    
                    for i in range(min(len(boxes), MAX_DETECTIONS_PER_FRAME)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())

                        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        
                        # Scale coordinates from 320x320 back to original frame size
                        frame_h, frame_w = frame.shape[:2]
                        scale_x = frame_w / 320.0
                        scale_y = frame_h / 320.0
                        
                        x1 = int(xyxy[0] * scale_x)
                        y1 = int(xyxy[1] * scale_y)
                        x2 = int(xyxy[2] * scale_x)
                        y2 = int(xyxy[3] * scale_y)
                        
                        
                        # Add to detections list for image saving
                        detections.append({
                            'coordinates': (x1, y1, x2, y2),
                            'class_name': class_name,
                            'confidence': conf
                        })

                        if DISPLAY:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Save image with bounding boxes if there are fire or smoke detections
                    if detections:
                        image_file_path = image_saver.save_detection_image(frame, detections, CLASS_NAMES)
                        
                        # Log individual detections with file location
                        for detection in detections:
                            class_name = detection['class_name']
                            confidence = detection['confidence']
                            
                            if class_name == "Fire":
                                event_logger.log_fire_detection(confidence, image_file_path)
                            elif class_name == "Smoke":
                                event_logger.log_smoke_detection(confidence, image_file_path)

        if DISPLAY:
            cv2.imshow("feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted by user, exiting...")

finally:
    # Stop video recording if active
    if video_recorder.is_recording:
        video_recorder.stop_recording()
    
    camera_server.stop()
    if DISPLAY:
        cv2.destroyAllWindows()