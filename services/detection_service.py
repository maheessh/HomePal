# services/detection_service.py
import os
import time
from collections import deque
import numpy as np
import cv2

# Try to import YOLO (ultralytics). If present, we use it as a model-based detector.
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

class DetectionService:
    """
    DetectionService provides:
      - detect_fire(frame) -> (mask, found_bool, confidence)
      - detect_smoke(frame) -> (mask, found_bool, confidence)

    It will try to load a model path if provided, otherwise fallback to color+motion heuristics.
    It keeps short-term history to reduce false positives.
    """
    def __init__(self, fire_model_path=None, smoke_model_path=None,
                 smoothing_window=5, min_fire_area=500, min_smoke_area=800):
        self.fire_model_path = fire_model_path
        self.smoke_model_path = smoke_model_path
        self.smoothing_window = smoothing_window
        self.min_fire_area = min_fire_area
        self.min_smoke_area = min_smoke_area

        # Temporal smoothing queues (store booleans)
        self._fire_history = deque(maxlen=smoothing_window)
        self._smoke_history = deque(maxlen=smoothing_window)

        # Try to load model(s) if path provided and ultralytics is installed
        self._fire_model = None
        self._smoke_model = None
        if _YOLO_AVAILABLE:
            try:
                if fire_model_path and os.path.exists(fire_model_path):
                    self._fire_model = YOLO(fire_model_path)
            except Exception:
                self._fire_model = None
            try:
                if smoke_model_path and os.path.exists(smoke_model_path):
                    self._smoke_model = YOLO(smoke_model_path)
            except Exception:
                self._smoke_model = None

    # -----------------------
    # Utility helpers
    # -----------------------
    def _temporal_decision(self, history, current_bool, required=3):
        """Push current_bool into history and decide if event should be considered real."""
        history.append(bool(current_bool))
        return sum(history) >= required  # e.g., >=3 true in last N frames

    # -----------------------
    # Color-based fire detection (fast fallback)
    # -----------------------
    def _color_fire_detect(self, frame):
        """Return mask, found_bool, confidence_estimate"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Flame colors: H roughly 0-50 (red->yellow), S high, V high
        lower = np.array([0, 100, 150], dtype=np.uint8)
        upper = np.array([50, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Filter by contour area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= self.min_fire_area]
        found = len(large_areas) > 0
        conf = 0.0
        if found:
            # confidence ~ normalized area of largest contour over frame area (0..1)
            max_area = max(large_areas)
            conf = min(1.0, max_area / (frame.shape[0] * frame.shape[1]) * 5.0)  # scale factor
        return mask, found, float(conf)

    # -----------------------
    # Smoke detection heuristic (fallback)
    # -----------------------
    def _color_smoke_detect(self, frame):
        """
        Smoke tends to be low-contrast, desaturated, grayish blobs.
        We'll detect regions with low saturation but significant texture / edge activity.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        # Candidate pixels: low saturation, not too dark or too bright
        sat_thresh = 60
        v_low, v_high = 50, 220
        candidate = ((s < sat_thresh) & (v > v_low) & (v < v_high)).astype('uint8') * 255

        # Enhance edges inside candidate region (smoke has soft edges; use Laplacian)
        edges = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_8U)
        _, edges_bin = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(candidate, edges_bin)

        # Clean up and check area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= self.min_smoke_area]
        found = len(large_areas) > 0
        conf = 0.0
        if found:
            conf = min(1.0, max(large_areas) / (frame.shape[0] * frame.shape[1]) * 3.0)
        return mask, found, float(conf)

    # -----------------------
    # Model-based wrapper (if YOLO/NCNN model available)
    # -----------------------
    def _model_detect(self, model, frame, class_name='fire', conf_threshold=0.3):
        """
        Run YOLO model on frame (resized as needed) and return mask + found + avg_conf.
        Assumes model's res.names contains class names including 'fire'/'smoke' as applicable.
        """
        if model is None:
            return None, False, 0.0
        try:
            # Run inference (model will handle resizing)
            results = model(frame, verbose=False)
            # Aggregate boxes where class matches
            confs = []
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for res in results:
                if getattr(res, 'boxes', None) is None:
                    continue
                for box in res.boxes:
                    cls_idx = int(box.cls[0])
                    name = res.names.get(cls_idx, str(cls_idx))
                    if class_name.lower() in name.lower():
                        conf = float(box.conf[0].cpu().item()) if hasattr(box.conf, 'cpu') else float(box.conf[0])
                        confs.append(conf)
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            found = len(confs) > 0
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return mask, found, avg_conf
        except Exception:
            return None, False, 0.0

    # -----------------------
    # Public API
    # -----------------------
    def detect_fire(self, frame):
        """
        Returns: (mask, found_bool, confidence)
        - mask: binary mask (H x W) or None
        - found_bool: bool
        - confidence: float 0..1 estimate
        """
        # 1) Prefer model if available
        if self._fire_model is not None:
            mask, found, conf = self._model_detect(self._fire_model, frame, class_name='fire', conf_threshold=0.25)
            if mask is None:
                # fallback to color
                mask, found, conf = self._color_fire_detect(frame)
        else:
            mask, found, conf = self._color_fire_detect(frame)

        # Temporal smoothing: only return True if detection persists across several frames
        final_decision = self._temporal_decision(self._fire_history, found, required=max(1, self.smoothing_window//2))
        # Amplify confidence slightly if temporal consensus is met
        if final_decision and conf < 0.9:
            conf = min(1.0, conf + 0.2)
        return mask, bool(final_decision), float(conf)

    def detect_smoke(self, frame):
        if self._smoke_model is not None:
            mask, found, conf = self._model_detect(self._smoke_model, frame, class_name='smoke', conf_threshold=0.25)
            if mask is None:
                mask, found, conf = self._color_smoke_detect(frame)
        else:
            mask, found, conf = self._color_smoke_detect(frame)

        final_decision = self._temporal_decision(self._smoke_history, found, required=max(1, self.smoothing_window//2))
        if final_decision and conf < 0.9:
            conf = min(1.0, conf + 0.15)
        return mask, bool(final_decision), float(conf)
