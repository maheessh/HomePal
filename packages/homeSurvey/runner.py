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

# mapping from model class ids -> names
CLASS_NAMES = {0: "Fire", 1: "Smoke"}

# Initialize event logger
event_logger = get_logger()

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

        now = time.time()
        if motion_found and (now - last_detection_time) >= DETECTION_COOLDOWN:
            last_detection_time = now
            
            # Log motion detection event
            frame_info = {
                "frame_size": frame.shape[:2],
                "motion_area_total": motion_area_total
            }
            event_logger.log_motion_detection(motion_area_total, frame_info=frame_info)

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
                    for i in range(min(len(boxes), MAX_DETECTIONS_PER_FRAME)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())

                        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        
                        # Log detection event with coordinates and confidence
                        coordinates = {
                            "x1": int(xyxy[0]),
                            "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]),
                            "y2": int(xyxy[3])
                        }
                        
                        frame_info = {
                            "detection_size": smframe.shape[:2],
                            "model_size": IMG_SIZE,
                            "confidence_threshold": CONF_THRESH
                        }
                        
                        # Log based on detection type
                        if class_name == "Fire":
                            event_logger.log_fire_detection(conf, coordinates, frame_info)
                        elif class_name == "Smoke":
                            event_logger.log_smoke_detection(conf, coordinates, frame_info)

                        if DISPLAY:
                            # Scale coordinates from 320x320 back to original frame size
                            frame_h, frame_w = frame.shape[:2]
                            scale_x = frame_w / 320.0
                            scale_y = frame_h / 320.0
                            
                            x1 = int(xyxy[0] * scale_x)
                            y1 = int(xyxy[1] * scale_y)
                            x2 = int(xyxy[2] * scale_x)
                            y2 = int(xyxy[3] * scale_y)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if DISPLAY:
            cv2.imshow("feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted by user, exiting...")

finally:
    camera_server.stop()
    if DISPLAY:
        cv2.destroyAllWindows()