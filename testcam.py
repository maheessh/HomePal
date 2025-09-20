# camserv.py
#!/usr/bin/env python3
"""
Camera Server that detects motion (Surveillance) and human presence (Monitor Me),
saving events with captured images to a JSON file.
Now extended with YOLOv8 + DeepFace from person_monitor.py.
"""
import cv2
import threading
import time
import datetime
import json
import os
import uuid
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
from deepface import DeepFace

EVENTS_FILE = 'events.json'
CAPTURES_DIR = 'captured_images'

# ------------------- Person Monitor Classes (No changes needed) -------------------
class ObjectDetector:
    """YOLO person detector"""
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame, target_class='person'):
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
                    conf = box.conf[0]
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return processed_frame, detected_boxes

class PersonRecognizer:
    """DeepFace-based person recognition"""
    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        self.known_faces = self.load_faces()

    def load_faces(self):
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
        x1, y1, x2, y2 = person_box
        person_roi = frame[y1:y2, x1:x2]
        name = "Unknown"
        try:
            for known_name, known_path in self.known_faces:
                result = DeepFace.verify(person_roi, known_path, enforce_detection=False, model_name='VGG-Face', distance_metric='cosine')
                if result["verified"]:
                    name = known_name
                    break
        except Exception as e:
            # DeepFace can throw errors if no face is found in ROI, which is expected
            pass
        cv2.putText(frame, name, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame, name

class EmotionRecognizer:
    """DeepFace-based emotion analysis"""
    def analyze(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        person_roi = frame[y1:y2, x1:x2]
        emotion = "N/A"
        try:
            result = DeepFace.analyze(person_roi,
                                      actions=['emotion'],
                                      enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            emotion = result["dominant_emotion"]
        except Exception as e:
            # Expected if no face is clearly visible
            pass
        cv2.putText(frame, emotion, (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, emotion

class ActivityRecognizer:
    """Pose-based activity recognition"""
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def classify_activity(self, keypoints):
        activity = "Unknown"
        if keypoints is None or len(keypoints) < 17:
            return activity
        
        # Check if keypoints have confidence scores (3rd dimension)
        has_confidence = keypoints.shape[1] == 3
        
        # Define keypoint indices
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        
        # Basic check if keypoints are detected with minimum confidence
        if has_confidence and (keypoints[L_HIP][2] < 0.5 or keypoints[R_HIP][2] < 0.5):
            return "Standing" # Default if hips are not clear
        
        l_hip_pt, r_hip_pt = keypoints[L_HIP], keypoints[R_HIP]
        l_knee_pt, r_knee_pt = keypoints[L_KNEE], keypoints[R_KNEE]

        y_coords = keypoints[:, 1]
        x_coords = keypoints[:, 0]
        height = max(y_coords) - min(y_coords)
        width = max(x_coords) - min(x_coords)

        if height < 10 or width < 10: return "Unknown" # Avoid division by zero for tiny skeletons

        if width > height * 1.4:
            activity = "Falling"
        elif abs(l_hip_pt[1] - l_knee_pt[1]) < 40 and abs(r_hip_pt[1] - r_knee_pt[1]) < 40:
            activity = "Sitting"
        elif height > width * 1.2:
            activity = "Standing"
        else:
            activity = "Walking/Moving"
        return activity

    def analyze(self, frame):
        results = self.model(frame, verbose=False)
        activity = "Unknown"
        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                keypoints_tensor = result.keypoints.xy[0]
                if keypoints_tensor.numel() > 0: # Check if tensor is not empty
                    keypoints = keypoints_tensor.cpu().numpy()
                    activity = self.classify_activity(keypoints)
                    x_min, y_min = int(min(keypoints[:, 0])), int(min(keypoints[:, 1]))
                    cv2.putText(frame, activity, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame, activity

# ------------------- Camera Server -------------------
class SimpleCameraServer:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None # This will hold the latest processed frame for the stream
        self.raw_frame = None # This will hold the latest raw frame for captures
        self._running = False
        self._frame_lock = threading.Lock()
        self.active_modules = {"surveillance": False, "monitor": False}

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Person monitor placeholders (lazy loaded)
        self.person_detector = None
        self.person_recognizer = None
        self.emotion_recognizer = None
        self.activity_recognizer = None

        self._file_lock = threading.Lock()
        self._last_event_time = 0
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/stream')
        def mjpeg_stream():
            def generate_frames():
                while self._running and self.cap:
                    try:
                        with self._frame_lock:
                            if self.frame is not None:
                                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.033)
                    except Exception as e:
                        print(f"❌ Error in MJPEG stream: {e}")
                        break
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.get_camera_info())

        @self.app.route('/api/detection/config', methods=['POST'])
        def configure_detection():
            config = request.get_json()
            if 'surveillance' in config:
                self.active_modules['surveillance'] = config['surveillance']
            if 'monitor' in config:
                self.active_modules['monitor'] = config['monitor']
            print(f"✅ Detection config updated: {self.active_modules}")
            return jsonify({"success": True, "message": "Config updated"})

    def start(self) -> bool:
        if self._running:
            return True
        try:
            # Try to use DSHOW backend for Windows for better performance
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    print("❌ All camera backend attempts failed.")
                    return False
            print("✅ Camera opened successfully.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._running = True
            
            # Ensure the captures directory exists
            if not os.path.exists(CAPTURES_DIR):
                os.makedirs(CAPTURES_DIR)

            self.start_capture_thread()
            return True
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False

    def stop(self):
        self._running = False
        time.sleep(0.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("✅ Camera capture stopped.")

    def _create_and_save_event(self, event_data):
        """Captures image and writes event to file."""
        with self._file_lock:
            # Capture the image
            filename = f"event_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            filepath = os.path.join(CAPTURES_DIR, filename)
            
            local_raw_frame = None
            with self._frame_lock:
                 if self.raw_frame is not None:
                    local_raw_frame = self.raw_frame.copy()

            if local_raw_frame is not None:
                cv2.imwrite(filepath, local_raw_frame)
                event_data["image_path"] = filename # Add image path to event
            else:
                event_data["image_path"] = None # No image available
                print("⚠️ Could not capture image; raw_frame is None.")

            # Append to events.json
            try:
                events_list = []
                if os.path.exists(EVENTS_FILE) and os.path.getsize(EVENTS_FILE) > 0:
                    with open(EVENTS_FILE, 'r') as f:
                        events_list = json.load(f)
                
                events_list.insert(0, event_data)
                
                with open(EVENTS_FILE, 'w') as f:
                    json.dump(events_list, f, indent=4)
            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ Error writing to {EVENTS_FILE}: {e}")

    def capture_frames(self):
        while self._running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with self._frame_lock:
                self.raw_frame = frame.copy() # Store raw frame for saving

            processed_frame = frame.copy()
            current_time = time.time()

            # ----- Surveillance: Motion Detection -----
            if self.active_modules['surveillance']:
                fg_mask = self.background_subtractor.apply(processed_frame)
                fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
                fg_mask = cv2.dilate(fg_mask, None, iterations=2)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = False
                for c in contours:
                    if cv2.contourArea(c) > 1000:
                        motion_detected = True
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if motion_detected and (current_time - self._last_event_time > 5):
                    self._last_event_time = current_time
                    event = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "module": "surveillance",
                        "class_name": "Motion",
                        "description": ""
                    }
                    self._create_and_save_event(event)

            # ----- Monitor Me: Lazy-load YOLO + DeepFace -----
            if self.active_modules['monitor']:
                if self.person_detector is None:
                    print("⏳ Loading YOLO + DeepFace models...")
                    self.person_detector = ObjectDetector()
                    self.person_recognizer = PersonRecognizer(faces_dir="faces")
                    self.emotion_recognizer = EmotionRecognizer()
                    self.activity_recognizer = ActivityRecognizer()
                    print("✅ Models loaded.")

                processed_frame, person_boxes = self.person_detector.detect(processed_frame, target_class='person')
                processed_frame, activity = self.activity_recognizer.analyze(processed_frame)

                if person_boxes and (current_time - self._last_event_time > 5):
                    self._last_event_time = current_time
                    # Process the first person found for the event log
                    box = person_boxes[0]
                    _, name = self.person_recognizer.recognize(processed_frame, box)
                    _, emotion = self.emotion_recognizer.analyze(processed_frame, box)
                    
                    event_details = f"Person: {name}, Emotion: {emotion}, Activity: {activity}"
                    
                    event = {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "module": "monitor",
                        "class_name": event_details,
                        "description": ""
                    }
                    self._create_and_save_event(event)
                
                # Still draw boxes for all people in the live feed
                for box in person_boxes:
                     self.person_recognizer.recognize(processed_frame, box)
                     self.emotion_recognizer.analyze(processed_frame, box)

            with self._frame_lock:
                self.frame = processed_frame

    def start_capture_thread(self):
        threading.Thread(target=self.capture_frames, daemon=True).start()

    def get_camera_info(self):
        return {"is_running": self._running}

    def run_server(self, host='0.0.0.0', port=5001, debug=False):
        self.app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == "__main__":
    camera = SimpleCameraServer(camera_id=0)
    if camera.start():
        try:
            camera.run_server()
        finally:
            camera.stop()