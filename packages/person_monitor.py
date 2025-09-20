#!/usr/bin/env python3
"""
Person Monitor: Detect people and infer basic activity (sitting/standing)
using YOLOv3 with OpenCV DNN.
"""

import cv2
import numpy as np


class PersonActivityDetector:
    """Detect people and infer sitting/standing using YOLOv3."""

    def __init__(self, model_path="yolo_model/"):
        print("🔍 Loading Person Activity Detector (YOLOv3)...")
        try:
            weights_path = f"{model_path}yolov3.weights"
            config_path = f"{model_path}yolov3.cfg"
            names_path = f"{model_path}coco.names"

            # Load YOLOv3 network
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # Load class labels
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            # Output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [
                layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()
            ]

            print("✅ YOLOv3 model loaded successfully.")
        except Exception as e:
            print(
                f"❌ Failed to load YOLO model. Make sure cfg/weights/names exist. Error: {e}"
            )
            self.net = None

    def detect(self, frame):
        """
        Run person detection + activity inference.
        Returns:
            processed_frame (np.ndarray): frame with bounding boxes & labels
            activities (list): list of strings (e.g., ["Person: Standing"])
        """
        if not self.net:
            return frame, []

        height, width, _ = frame.shape
        processed_frame = frame.copy()
        activities = []

        # Preprocess
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if self.classes[class_id] == "person" and confidence > 0.3:
                    center_x, center_y = (
                        int(detection[0] * width),
                        int(detection[1] * height),
                    )
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]

                # Aspect ratio heuristic
                aspect_ratio = h / w if w > 0 else 0
                activity = "Standing" if aspect_ratio >= 1.5 else "Sitting"

                label = f"Person: {activity}"
                activities.append(label)

                # Draw
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    processed_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        return processed_frame, activities


# -----------------------------------------------------------
# Standalone testing (webcam stream)
# -----------------------------------------------------------
if __name__ == "__main__":
    detector = PersonActivityDetector("yolo_model/")

    cap = cv2.VideoCapture(0)  # change to video file or RTSP if needed

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        exit(1)

    print("🎥 Starting person monitoring... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, acts = detector.detect(frame)

        if acts:
            print("Detected:", acts)

        cv2.imshow("Person Monitor", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
