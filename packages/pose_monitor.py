# pose_service.py

import cv2
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Define connections between keypoints to form the skeleton
# This represents the connections for a 17-point COCO model
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
    (5, 11), (6, 12), (11, 12),  # Hips
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

class PoseService:
    """
    Handles pose detection, tracking, and activity analysis.
    """
    def __init__(self, model_path="yolo8npose_ncnn_model"):
        print("🛠️  Initializing Pose Service...")
        print(f"⌛ Loading model '{model_path}'. This might take a moment...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")

        # State tracking for each person (keyed by track_id)
        self.person_tracker = {}
        
        # Fall detection parameters
        self.FALL_VELOCITY_THRESHOLD = 15  # Vertical velocity threshold (pixels/second)
        self.FALL_TIME_THRESHOLD = 1.5      # Seconds to confirm fall after impact

    def detect_poses(self, frame):
        """
        Detects poses, analyzes activities, and draws skeletons.
        Returns the processed frame, a list of poses, and a list of activities.
        """
        # Use YOLOv8's tracking feature for more robust, stateful analysis
        results = self.model.track(frame, persist=True, verbose=False)
        
        poses = []
        activities = []
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_list = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            boxes = results[0].boxes.xywh.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                kpts = keypoints_list[i]
                confidence_scores = confs[i]
                box = boxes[i]

                # Draw the skeleton on the frame for visualization
                self._draw_skeleton(frame, kpts, confidence_scores)

                # Analyze activity for this person
                activity, pose_data = self._analyze_activity(track_id, kpts, box)
                activities.append(activity)
                poses.append(pose_data)
        
        self._cleanup_trackers()
        return frame, poses, activities

    def _draw_skeleton(self, frame, keypoints, confidences, conf_threshold=0.5):
        """Draws the skeleton and keypoints on the frame."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if confidences[start_idx] > conf_threshold and confidences[end_idx] > conf_threshold:
                start_point = tuple(np.round(keypoints[start_idx]).astype(int))
                end_point = tuple(np.round(keypoints[end_idx]).astype(int))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for i, point in enumerate(keypoints):
            if confidences[i] > conf_threshold:
                x, y = np.round(point).astype(int)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def _analyze_activity(self, track_id, keypoints, box):
        """
        Analyzes the pose of a single person to determine their activity
        using a stateful approach to detect falls accurately.
        """
        current_time = time.time()
        
        # Calculate torso center (midpoint between shoulders)
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        torso_y = (left_shoulder[1] + right_shoulder[1]) / 2 if left_shoulder[0] > 0 and right_shoulder[0] > 0 else 0
        
        w, h = box[2], box[3]
        aspect_ratio = w / h if h > 0 else 0

        # Initialize or update tracker for this person
        if track_id not in self.person_tracker:
            self.person_tracker[track_id] = {
                'history': [], 'status': 'Stable', 'fall_event_time': None
            }
        
        tracker = self.person_tracker[track_id]
        tracker['history'].append((current_time, torso_y))
        tracker['history'] = tracker['history'][-10:] # Keep last 10 records
        tracker['last_update'] = current_time

        # Calculate vertical velocity
        velocity = 0
        if len(tracker['history']) > 2:
            dt = tracker['history'][-1][0] - tracker['history'][-2][0]
            dy = tracker['history'][-1][1] - tracker['history'][-2][1] # Positive dy is downward
            if dt > 0:
                velocity = dy / dt

        # --- Fall Detection State Machine ---
        activity = "Stable"
        
        # 1. Check for a potential fall (high downward velocity)
        if velocity > self.FALL_VELOCITY_THRESHOLD and tracker['status'] != 'Falling':
            tracker['status'] = 'Potential Fall'
            tracker['fall_event_time'] = current_time

        # 2. Confirm the fall if person is horizontal on the ground
        if tracker['status'] == 'Potential Fall':
            if aspect_ratio > 1.2: # Person is horizontal
                if (current_time - tracker['fall_event_time']) >= self.FALL_TIME_THRESHOLD:
                    activity = "Falling"
                    tracker['status'] = 'Falling'
                else:
                    activity = "Potential Fall" # Waiting for confirmation
            else: # Person stood up or wasn't horizontal, reset
                tracker['status'] = 'Stable'
                tracker['fall_event_time'] = None

        elif tracker['status'] == 'Falling':
            # Remain in 'Falling' state until they are no longer horizontal
            if aspect_ratio < 1.0:
                tracker['status'] = 'Stable'
            else:
                activity = "Falling"

        # 3. Basic activity classification if not in a fall state
        if tracker['status'] == 'Stable':
            if aspect_ratio > 1.2:
                activity = "Lying Down"
            else:
                activity = "Standing/Sitting"

        pose_data = {'keypoints': keypoints.tolist(), 'box': box.tolist(), 'velocity': velocity}
        return activity, pose_data

    def _cleanup_trackers(self):
        """Remove trackers for people who are no longer in the frame."""
        current_time = time.time()
        stale_ids = [
            track_id for track_id, data in self.person_tracker.items()
            if (current_time - data.get('last_update', 0)) > 5.0 # 5-second timeout
        ]
        for track_id in stale_ids:
            del self.person_tracker[track_id]

    def log_event(self, activity, confidence, pose_data, image_path):
        """Logs a detected event to a JSON file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "activity": activity,
            "confidence": float(confidence),
            "image_path": image_path,
            "pose_data": pose_data
        }
        try:
            with open("events.json", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Error logging event: {e}")

    def save_frame(self, frame, activity_name):
        """Saves a frame to the 'events' directory."""
        if not os.path.exists("events"):
            os.makedirs("events")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join("events", f"{activity_name}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        return filename
