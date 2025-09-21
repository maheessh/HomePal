# config.py
import os

class Config:
    # Camera
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    INFERENCE_SIZE = 320

    # Motion detection
    MOTION_THRESHOLD = 5000
    EVENT_COOLDOWN = 5  # seconds

    # Paths (you can adjust if needed)
    OBJECT_DETECTION_MODEL = os.path.join("models", "yolov8n.pt")
    FIRE_DETECTION_MODEL = os.path.join("models", "fire_ncnn_model.onnx")
    POSE_MODEL = os.path.join("models", "yolov8n-pose.pt")
    FACE_DB_PATH = os.path.join("face_db")

    # Classes we care about in surveillance
    SURVEILLANCE_CLASSES = ["person", "fire", "smoke"]
