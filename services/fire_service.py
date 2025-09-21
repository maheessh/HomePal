#!/usr/bin/env python3
"""
Fire Service for HomePal
Attempts fire detection using NCNN model with a robust color-based fallback.
Designed to mirror the PoseService pattern (separate service module), so the
surveillance (motion) code stays clean and fire logic lives here.
"""

from __future__ import annotations

import os
import importlib.util
from typing import Optional, Tuple

import cv2
import numpy as np

# Reuse existing detection heuristics to avoid duplication
try:
    from .detection_service import DetectionService
except ImportError:
    # Fallback import style if relative fails
    from services.detection_service import DetectionService


class FireService:
    """Service wrapper for fire detection.

    Usage:
        fire_service = FireService(model_dir="packages/inferno_ncnn_model")
        vis_frame, detected, confidence = fire_service.detect_fire(frame)
    """

    def __init__(self, model_dir: str = os.path.join("packages", "inferno_ncnn_model")):
        self.model_dir = model_dir
        self._ncnn_available = False
        self._ncnn_model = None  # lazily prepared net/extractor if available

        # Try to load NCNN python bindings and prepare a net
        try:
            import ncnn  # noqa: F401
            self._ncnn_available = True
        except Exception:
            self._ncnn_available = False

        # Try to import any helper from model_ncnn.py (if it defines a class)
        self._helper = None
        try:
            helper_path = os.path.join(self.model_dir, "model_ncnn.py")
            if os.path.exists(helper_path):
                spec = importlib.util.spec_from_file_location("inferno_model_ncnn", helper_path)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                # If a helper class exists (e.g., InfernoNCNNModel), keep a reference
                self._helper = getattr(mod, "InfernoNCNNModel", None)
        except Exception:
            self._helper = None

    # -----------------------
    # Public API
    # -----------------------
    def detect_fire(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """Detect fire in a frame.

        Returns: (vis_frame, detected_bool, confidence_float)
        """
        if frame is None or frame.size == 0:
            return frame, False, 0.0

        # Try NCNN helper class if available
        if self._helper is not None:
            try:
                model = self._helper(self.model_dir)
                det = model.predict(frame)
                # Expect det to be list of boxes or a confidence map depending on implementation
                # Fallback parse: treat any non-empty output as detection
                detected = bool(det)
                confidence = 0.8 if detected else 0.0
                vis = frame.copy()
                # If boxes are present, draw them
                if isinstance(det, (list, tuple)) and det and isinstance(det[0], (list, tuple, np.ndarray)):
                    for box in det:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in box[:4]]
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 80, 255), 2)
                        except Exception:
                            continue
                if detected:
                    self._draw_label(vis, "FIRE", (10, 30))
                return vis, detected, confidence
            except Exception:
                # Continue to fallback below
                pass

        # Fallback: reuse DetectionService fire heuristic/model if available
        mask, detected, confidence = self._detect_fire_fallback(frame)
        vis = self._overlay_mask(frame, mask, color=(0, 80, 255))
        if detected:
            self._draw_label(vis, f"FIRE {confidence:.2f}", (10, 30))
        return vis, detected, confidence

    # -----------------------
    # Fallback detection via DetectionService (includes color heuristic)
    # -----------------------
    def _detect_fire_fallback(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        try:
            # Initialize lazily and reuse
            if not hasattr(self, "_detector") or self._detector is None:
                self._detector = DetectionService(fire_model_path=None)
            mask, detected, confidence = self._detector.detect_fire(frame)
            # Ensure mask shape for overlay
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return mask, bool(detected), float(confidence)
        except Exception:
            # Absolute last resort: simple HSV threshold
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 100, 150], dtype=np.uint8)
            upper = np.array([50, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected = any(cv2.contourArea(c) > 500 for c in contours)
            confidence = 0.6 if detected else 0.0
            return mask, detected, confidence

    # -----------------------
    # Visualization helpers
    # -----------------------
    def _overlay_mask(self, frame: np.ndarray, mask: np.ndarray, color=(0, 80, 255)) -> np.ndarray:
        vis = frame.copy()
        if mask is None:
            return vis
        if mask.ndim == 2:
            mask_color = np.zeros_like(vis)
            mask_color[:, :] = color
            alpha = (mask > 0).astype(np.float32) * 0.25
            vis = (vis.astype(np.float32) * (1 - alpha)[..., None] + mask_color.astype(np.float32) * alpha[..., None]).astype(np.uint8)
        return vis

    def _draw_label(self, frame: np.ndarray, text: str, org: Tuple[int, int]) -> None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)


__all__ = ["FireService"]


