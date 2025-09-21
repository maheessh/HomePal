#!/usr/bin/env python3
"""
Pose Service for Personal Monitoring
Handles pose estimation using YOLOv8 Pose NCNN model for activity recognition
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

class PoseService:
    """Service for pose estimation and activity recognition"""
    
    def __init__(self, model_path="packages/yolo8spose_ncnn_model"):
        self.model_path = model_path
        self.pose_model = None
        self.events_lock = threading.Lock()
        self.events_file = "events.json"
        self.captures_dir = "captured_images"
        self._initialize_model()
        self._ensure_directories()
        
        # COCO pose keypoint indices
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
    
    def _initialize_model(self):
        """Initialize the YOLOv8 Pose NCNN model"""
        try:
            # Import the model class
            import importlib.util
            model_file = os.path.join(self.model_path, "model_ncnn.py")
            spec = importlib.util.spec_from_file_location("model_ncnn", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            self.pose_model = model_module.YOLOv8PoseNCNNModel(self.model_path)
            print(f"✅ YOLOv8 Pose NCNN model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 Pose NCNN model: {e}")
            self.pose_model = None
    
    def detect_poses(self, frame):
        """
        Detect poses in the frame using YOLOv8 Pose NCNN model
        Returns: (processed_frame, poses, activities)
        """
        if self.pose_model is None:
            return frame, [], []
        
        try:
            processed_frame = frame.copy()
            
            # Use the pose model to detect poses
            poses = self.pose_model.predict(frame)
            
            activities = []
            
            # Classify activity for each pose and draw on frame
            for pose in poses:
                keypoints = pose['keypoints']
                activity = self._classify_activity(keypoints)
                activities.append(activity)
                
                # Draw pose on frame
                self._draw_pose(processed_frame, pose['bbox'], keypoints, activity)
            
            return processed_frame, poses, activities
            
        except Exception as e:
            print(f"❌ Pose detection error: {e}")
            return frame, [], []
    
    def _classify_activity(self, keypoints):
        """
        Classify activity from pose keypoints
        Returns: activity string (Standing, Sitting, Walking, Falling)
        """
        if not keypoints or len(keypoints) < 17:
            return "Unknown"
        
        try:
            # Get relevant keypoints (convert to numpy for easier manipulation)
            kps = np.array(keypoints)
            
            # Extract key joint positions
            nose = kps[0][:2] if kps[0][2] > 0.5 else None
            left_shoulder = kps[5][:2] if kps[5][2] > 0.5 else None
            right_shoulder = kps[6][:2] if kps[6][2] > 0.5 else None
            left_hip = kps[11][:2] if kps[11][2] > 0.5 else None
            right_hip = kps[12][:2] if kps[12][2] > 0.5 else None
            left_knee = kps[13][:2] if kps[13][2] > 0.5 else None
            right_knee = kps[14][:2] if kps[14][2] > 0.5 else None
            left_ankle = kps[15][:2] if kps[15][2] > 0.5 else None
            right_ankle = kps[16][:2] if kps[16][2] > 0.5 else None
            
            # Calculate body dimensions
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
                hip_center = np.mean([left_hip, right_hip], axis=0)
                
                # Calculate body height and width
                body_height = abs(hip_center[1] - shoulder_center[1])
                body_width = abs(left_shoulder[0] - right_shoulder[0])
                
                # Calculate leg angles if knees are visible
                left_leg_angle = 180
                right_leg_angle = 180
                
                if left_hip and left_knee and left_ankle:
                    left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
                
                if right_hip and right_knee and right_ankle:
                    right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
                
                # Activity classification logic
                avg_leg_angle = (left_leg_angle + right_leg_angle) / 2
                
                # Falling: body is horizontal (width >> height)
                if body_width > body_height * 1.5:
                    return "Falling"
                
                # Sitting: legs are bent (knee angles are acute)
                elif avg_leg_angle < 120:
                    return "Sitting"
                
                # Standing: legs are straight and body is vertical
                elif avg_leg_angle > 150 and body_height > body_width * 1.2:
                    return "Standing"
                
                # Walking/Moving: intermediate state
                else:
                    return "Walking"
            
            return "Unknown"
            
        except Exception as e:
            print(f"❌ Activity classification error: {e}")
            return "Unknown"
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        except:
            return 180
    
    def _draw_pose(self, frame, bbox, keypoints, activity):
        """Draw pose keypoints and activity on frame"""
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:  # Only draw visible keypoints
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw activity label
        activity_color = (0, 255, 0) if activity != "Falling" else (0, 0, 255)
        cv2.putText(frame, activity, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, activity_color, 2)
    
    def log_event(self, activity, confidence=None, pose_data=None, image_path=None):
        """Log pose/activity event using centralized EventLogger"""
        try:
            return event_logger.log_activity_event(
                activity=activity,
                confidence=confidence,
                image_path=image_path
            )
        except Exception as e:
            print(f"❌ Failed to log event: {e}")
            return ""
    
    def save_frame(self, frame, activity):
        """Save frame as image for event logging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{activity}_{timestamp}.jpg"
            filepath = os.path.join(self.captures_dir, filename)
            
            cv2.imwrite(filepath, frame)
            return filename
        except Exception as e:
            print(f"❌ Failed to save frame: {e}")
            return None
