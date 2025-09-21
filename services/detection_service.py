#!/usr/bin/env python3
"""
Detection Service for Home Surveillance
Handles fire detection, smoke detection, and motion detection using AI models
"""

import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
import threading

# Add packages to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'packages'))

# Import the centralized event logger
try:
    from .event_logger import event_logger
except ImportError:
    from event_logger import event_logger

class DetectionService:
    """Service for home surveillance detection tasks"""
    
    def __init__(self, model_path="packages/inferno_ncnn_model"):
        self.model_path = model_path
        self.inferno_model = None
        self.events_lock = threading.Lock()
        self.events_file = "events.json"
        self.captures_dir = "captured_images"
        self._initialize_model()
        self._ensure_directories()
        
        # Smoke detection parameters
        self.smoke_history = []
        self.smoke_threshold = 0.3  # Threshold for smoke detection
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
    
    def _initialize_model(self):
        """Initialize the NCNN model"""
        try:
            # Import the model class
            import importlib.util
            model_file = os.path.join(self.model_path, "model_ncnn.py")
            spec = importlib.util.spec_from_file_location("model_ncnn", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            self.inferno_model = model_module.InfernoNCNNModel(self.model_path)
            print(f"✅ Inferno NCNN model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load Inferno NCNN model: {e}")
            self.inferno_model = None
    
    def detect_fire(self, frame):
        """
        Detect fire in the frame using Inferno NCNN model
        Returns: (processed_frame, fire_detected, confidence)
        """
        if self.inferno_model is None:
            return frame, False, 0.0
        
        try:
            processed_frame = frame.copy()
            
            # Use the Inferno model to detect fire
            fire_detected, fire_confidence = self.inferno_model.is_fire_detected(frame)
            
            # Draw fire detection result
            if fire_detected:
                cv2.rectangle(processed_frame, (10, 10), (300, 60), (0, 0, 255), -1)
                cv2.putText(processed_frame, f"FIRE DETECTED! {fire_confidence:.2f}", 
                          (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(processed_frame, f"Fire Confidence: {fire_confidence:.2f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return processed_frame, fire_detected, fire_confidence
                
        except Exception as e:
            print(f"❌ Fire detection error: {e}")
            return frame, False, 0.0
    
    def detect_smoke(self, frame):
        """
        Detect smoke in the frame using computer vision techniques
        Returns: (processed_frame, smoke_detected, confidence)
        """
        try:
            processed_frame = frame.copy()
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
            
            # Define smoke color range (grayish colors)
            # Smoke typically appears as white/gray with some transparency
            lower_smoke = np.array([0, 0, 100])    # Lower bound for smoke
            upper_smoke = np.array([180, 30, 255]) # Upper bound for smoke
            
            # Create mask for smoke colors
            smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            smoke_detected = False
            smoke_confidence = 0.0
            smoke_areas = []
            
            # Analyze contours for smoke-like characteristics
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Calculate contour properties
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Smoke typically has irregular shapes and varying density
                    # Calculate density variation in the region
                    roi = processed_frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        density_std = np.std(gray_roi)
                        
                        # Smoke characteristics: irregular shape, varying density
                        if 0.3 < aspect_ratio < 3.0 and density_std > 20:
                            smoke_areas.append((x, y, w, h))
                            
                            # Draw smoke detection rectangle
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                            
                            # Calculate confidence based on area and characteristics
                            confidence = min(1.0, area / 10000.0)  # Normalize by area
                            smoke_confidence = max(smoke_confidence, confidence)
            
            # Smoke is detected if we have significant areas with high confidence
            if len(smoke_areas) > 0 and smoke_confidence > self.smoke_threshold:
                smoke_detected = True
                
                # Update smoke history for temporal consistency
                self.smoke_history.append(smoke_confidence)
                if len(self.smoke_history) > 10:
                    self.smoke_history.pop(0)
                
                # Use average confidence over recent frames for stability
                avg_confidence = np.mean(self.smoke_history)
                if avg_confidence > self.smoke_threshold:
                    cv2.rectangle(processed_frame, (10, 70), (350, 110), (0, 165, 255), -1)
                    cv2.putText(processed_frame, f"SMOKE DETECTED! {avg_confidence:.2f}", 
                              (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    smoke_detected = False
            else:
                # Clear smoke history if no smoke detected
                if len(self.smoke_history) > 0:
                    self.smoke_history.pop(0)
            
            # Add smoke status text
            status_color = (0, 165, 255) if smoke_detected else (255, 255, 255)
            status_text = f"Smoke: {'DETECTED' if smoke_detected else 'CLEAR'}"
            cv2.putText(processed_frame, status_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            return processed_frame, smoke_detected, smoke_confidence
            
        except Exception as e:
            print(f"❌ Smoke detection error: {e}")
            return frame, False, 0.0
    
    def detect_motion(self, frame, background_subtractor):
        """
        Detect motion in the frame using background subtraction
        Returns: (processed_frame, motion_detected)
        """
        try:
            processed_frame = frame.copy()
            
            # Apply background subtractor
            fg_mask = background_subtractor.apply(processed_frame)
            
            # Clean up the mask
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            motion_areas = []
            
            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue
                
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))
                
                # Draw motion rectangle
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add motion status text
            status_color = (0, 255, 0) if motion_detected else (255, 255, 255)
            status_text = f"Motion: {'DETECTED' if motion_detected else 'CLEAR'}"
            cv2.putText(processed_frame, status_text, (10, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            return processed_frame, motion_detected, motion_areas
            
        except Exception as e:
            print(f"❌ Motion detection error: {e}")
            return frame, False, []
    
    def log_event(self, event_type, confidence=None, motion_areas=None, image_path=None):
        """Log detection event using centralized EventLogger"""
        try:
            if event_type.lower().startswith("fire"):
                return event_logger.log_fire_event(confidence or 0.0, image_path)
            elif event_type.lower().startswith("smoke"):
                return event_logger.log_smoke_event(confidence or 0.0, image_path)
            elif event_type.lower().startswith("motion"):
                return event_logger.log_motion_event(motion_areas or [], confidence, image_path)
            elif event_type.lower().startswith("person"):
                return event_logger.log_person_event(confidence or 0.0, image_path)
            else:
                # Generic event logging
                return event_logger.log_event(
                    event_type=event_type,
                    module="surveillance",
                    confidence=confidence,
                    image_path=image_path,
                    metadata={"motion_areas": motion_areas} if motion_areas else None
                )
        except Exception as e:
            print(f"❌ Failed to log event: {e}")
            return ""
    
    def save_frame(self, frame, event_type):
        """Save frame as image for event logging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type}_{timestamp}.jpg"
            filepath = os.path.join(self.captures_dir, filename)
            
            cv2.imwrite(filepath, frame)
            return filename
        except Exception as e:
            print(f"❌ Failed to save frame: {e}")
            return None
