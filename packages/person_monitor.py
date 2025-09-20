#!/usr/bin/env python3
"""
Extended Person Monitor:
1. Detects persons using a YOLOv8 model.
2. Recognizes known persons (face recognition via DeepFace).
3. Analyzes detected persons' activity (emotion recognition).
4. Classifies physical activities (Standing, Sitting, Walking, Falling) using YOLOv8 pose estimation.
"""

import cv2
import os
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np


# --- Primary Detector Class (Finds People) ---
class ObjectDetector:
    """A class to handle object detection using a YOLO model."""

    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame, target_class='person'):
        """
        Detects target objects in the frame.
        Returns: (processed_frame, detected_boxes)
        """
        processed_frame = frame.copy()
        detected_boxes = []
        results = self.model(processed_frame, verbose=False)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                if class_name == target_class:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    detected_boxes.append([x1, y1, x2, y2])

                    # Draw the bounding box for the person
                    conf = box.conf[0]
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return processed_frame, detected_boxes


# --- Person Recognition (DeepFace Face Recognition) ---
class PersonRecognizer:
    """Recognizes known people using DeepFace."""

    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        self.known_faces = self.load_faces()

    def load_faces(self):
        """Load all face images from a directory."""
        if not os.path.exists(self.faces_dir):
            print(f"⚠️ Faces directory '{self.faces_dir}' not found.")
            return []

        faces = []
        for filename in os.listdir(self.faces_dir):
            if filename.endswith((".jpg", ".png")):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.faces_dir, filename)
                faces.append((name, path))
        print(f"✅ Loaded {len(faces)} known faces.")
        return faces

    def recognize(self, frame, person_box):
        """Identify the person inside a given bounding box using DeepFace verify."""
        x1, y1, x2, y2 = person_box
        person_roi = frame[y1:y2, x1:x2]

        name = "Unknown"
        try:
            for known_name, known_path in self.known_faces:
                result = DeepFace.verify(person_roi, known_path, enforce_detection=False)
                if result["verified"]:
                    name = known_name
                    break
        except Exception as e:
            print(f"[FaceRec ERROR] {e}")

        cv2.putText(frame, name, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame, name


# --- Emotion Recognition (DeepFace Emotion Analysis) ---
class EmotionRecognizer:
    """Recognizes basic emotions using DeepFace."""

    def analyze(self, frame, person_box):
        """Run emotion analysis on a person ROI."""
        x1, y1, x2, y2 = person_box
        person_roi = frame[y1:y2, x1:x2]

        activity_label = "Analyzing..."
        try:
            result = DeepFace.analyze(person_roi,
                                      actions=['emotion'],
                                      enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            activity_label = result["dominant_emotion"]
        except Exception as e:
            print(f"[Emotion ERROR] {e}")

        cv2.putText(frame, activity_label, (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, activity_label


# --- Physical Activity Recognition (Pose-based) ---
class ActivityRecognizer:
    """Recognizes activities (Standing, Sitting, Walking, Falling) using YOLOv8 pose model."""

    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def classify_activity(self, keypoints):
        """
        Classify activity from pose keypoints.
        keypoints: np.array of shape (17,2) for COCO keypoints
        """
        activity = "Unknown"
        if keypoints is None or len(keypoints) < 17:
            return activity

        # Get key joints
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_knee, r_knee = keypoints[13], keypoints[14]

        # Height vs width
        y_coords = keypoints[:, 1]
        x_coords = keypoints[:, 0]
        height = max(y_coords) - min(y_coords)
        width = max(x_coords) - min(x_coords)

        # Simple rules
        if width > height * 1.4:
            activity = "Falling"
        elif abs(l_hip[1] - l_knee[1]) < 40 and abs(r_hip[1] - r_knee[1]) < 40:
            activity = "Sitting"
        elif height > width * 1.2:
            activity = "Standing"
        else:
            activity = "Walking/Moving"

        return activity

    def analyze(self, frame):
        """
        Run pose-based activity recognition on the entire frame.
        Returns: frame with annotations.
        """
        results = self.model(frame, verbose=False)
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                activity = self.classify_activity(keypoints)

                # Draw activity label
                x_min, y_min = int(min(keypoints[:, 0])), int(min(keypoints[:, 1]))
                cv2.putText(frame, activity, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame


# --- Main Execution Block ---
if __name__ == '__main__':
    VIDEO_SOURCE = 0
    person_detector = ObjectDetector(model_path='yolov8n.pt')
    person_recognizer = PersonRecognizer(faces_dir="faces")  # put known faces in /faces
    emotion_recognizer = EmotionRecognizer()
    activity_recognizer = ActivityRecognizer()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{VIDEO_SOURCE}'.")
        exit()

    print("🎥 Starting Person + Activity Monitor. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect persons
        processed_frame, person_boxes = person_detector.detect(frame, target_class='person')

        # Step 2: For each person, run recognition + emotion
        if person_boxes:
            for box in person_boxes:
                processed_frame, name = person_recognizer.recognize(processed_frame, box)
                processed_frame, emotion = emotion_recognizer.analyze(processed_frame, box)

        # Step 3: Run activity classification (pose-based)
        processed_frame = activity_recognizer.analyze(processed_frame)

        # Show frame
        cv2.imshow('Person + Activity Monitor', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
