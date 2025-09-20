#!/usr/bin/env python3
"""
A dedicated module for performing object detection using a YOLO model.
"""
from ultralytics import YOLO
import cv2

class ObjectDetector:
    """A class to handle object detection using a YOLO model."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the detector with a YOLO model."""
        self.model = YOLO(model_path)

    def detect(self, frame, target_classes=None):
        """
        Detects objects and returns the frame with drawn boxes and a list of detections.
        
        Args:
            frame: The video frame to process.
            target_classes (list, optional): A list of class names to detect. 
                                            If None, detects all objects.
        Returns:
            A tuple containing:
            - processed_frame (numpy.ndarray): The frame with boxes drawn.
            - detections (list): A list of detected class names (e.g., ['person', 'car']).
        """
        processed_frame = frame.copy()
        detections = []
        
        # Run inference
        results = self.model(processed_frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Check if the detected object is one of our targets
                if target_classes is None or class_name in target_classes:
                    detections.append(class_name)
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    label = f'{class_name} {box.conf[0]:.2f}'
                    
                    # Draw a blue box for general motion, green for a person
                    color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
        return processed_frame, detections