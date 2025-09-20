# camserv.py
#!/usr/bin/env python3
"""
Camera Server - OPTIMIZED VERSION
- Advanced Surveillance: YOLO-based object detection with automatic video recording.
- Enhanced Monitor Me: More robust pose-based fall detection logic.
- Performance Optimizations: Frame resizing, model consolidation, and skipped-frame processing.
"""
import cv2
import threading
import time
import datetime
import json
import os
import uuid
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
from deepface import DeepFace
from collections import deque

# --- OPTIMIZATION: Central Configuration ---
class Config:
    # Paths and Files
    EVENTS_FILE = 'events.json'
    CAPTURES_DIR = 'captured_images'
    RECORDINGS_DIR = 'captured_videos'
    FACES_DIR = 'faces'
    
    # Camera Settings
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    
    # Performance Settings
    INFERENCE_SIZE = 320  # Resize frames to this size for AI models (big speedup)
    PROCESS_INTERVAL = 5  # Run heavy DeepFace analysis every 5 frames
    
    # Model Paths
    # NOTE: Using one model for person/object detection
    OBJECT_DETECTION_MODEL = 'yolov8n.pt' 
    POSE_ESTIMATION_MODEL = 'yolov8n-pose.pt'
    
    # Detection Settings
    SURVEILLANCE_CLASSES = ['person', 'car'] # <-- UPDATE with 'fire', etc.
    EVENT_COOLDOWN = 15 # Seconds

# (VideoRecorder class remains the same - no changes needed)
class VideoRecorder:
    """Handles video recording when a critical event is detected."""
    def __init__(self, fps=20.0, duration=10.0, saved_folder=Config.RECORDINGS_DIR):
        self.fps = fps
        self.duration = duration
        self.saved_folder = saved_folder
        self.is_recording = False
        self.video_writer = None
        self.start_time = 0
        self.recording_thread = None
        self.recording_lock = threading.Lock()
        os.makedirs(saved_folder, exist_ok=True)

    def start_recording(self, frame):
        """Starts a new video recording in a separate thread."""
        with self.recording_lock:
            if self.is_recording:
                return  # Already recording
            self.is_recording = True
            self.start_time = time.time()
            
            height, width = frame.shape[:2]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_video_{timestamp}.mp4"
            filepath = os.path.join(self.saved_folder, filename)
            
            self.video_writer = cv2.VideoWriter(
                filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height)
            )
            
            self.recording_thread = threading.Thread(target=self._record, args=(filepath,))
            self.recording_thread.start()
            print(f"🎥 Started recording to {filepath}")
            return filename

    def _record(self, filepath):
        """Private method to manage the recording duration."""
        while time.time() - self.start_time < self.duration:
            time.sleep(0.1)
        self.stop_recording()
        print(f"✅ Finished recording: {filepath}")

    def add_frame(self, frame):
        """Adds a frame to the current video if recording."""
        with self.recording_lock:
            if self.is_recording and self.video_writer and self.video_writer.isOpened():
                self.video_writer.write(frame)

    def stop_recording(self):
        """Stops the recording and releases the writer."""
        with self.recording_lock:
            if self.is_recording:
                self.is_recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

# (PersonRecognizer and EmotionRecognizer are unchanged)
class PersonRecognizer:
    """DeepFace-based person recognition"""
    def __init__(self, faces_dir=Config.FACES_DIR):
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
        except Exception: pass
        return name

class EmotionRecognizer:
    """DeepFace-based emotion analysis"""
    def analyze(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        person_roi = frame[y1:y2, x1:x2]
        emotion = "N/A"
        try:
            result = DeepFace.analyze(person_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list): result = result[0]
            emotion = result["dominant_emotion"]
        except Exception: pass
        return emotion


# ------------------- Camera Server -------------------
class SimpleCameraServer:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None 
        self.raw_frame = None
        self._running = False
        self._frame_lock = threading.Lock()
        self.active_modules = {"surveillance": False, "monitor": False}

        self._last_event_time = 0
        self.video_recorder = VideoRecorder()
        
        # --- OPTIMIZATION: Model Consolidation ---
        self.object_detector = None
        self.pose_estimator = None
        self.person_recognizer = None
        self.emotion_recognizer = None

        # --- OPTIMIZATION: Frame Skipping ---
        self.frame_counter = 0

        self._file_lock = threading.Lock()
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/stream')
        def mjpeg_stream():
            def generate_frames():
                while self._running:
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
        def api_status(): return jsonify(self.get_camera_info())

        @self.app.route('/api/detection/config', methods=['POST'])
        def configure_detection():
            config = request.get_json()
            if 'surveillance' in config: self.active_modules['surveillance'] = config['surveillance']
            if 'monitor' in config: self.active_modules['monitor'] = config['monitor']
            print(f"✅ Detection config updated: {self.active_modules}")
            return jsonify({"success": True, "message": "Config updated"})

    def start(self) -> bool:
        if self._running: return True
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    print("❌ All camera backend attempts failed."); return False
            print("✅ Camera opened successfully.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
            self._running = True
            os.makedirs(Config.CAPTURES_DIR, exist_ok=True)
            os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)
            self.start_capture_thread()
            return True
        except Exception as e:
            print(f"❌ Camera start error: {e}"); return False

    def stop(self):
        self._running = False
        self.video_recorder.stop_recording()
        time.sleep(0.5)
        if self.cap: self.cap.release(); self.cap = None
        print("✅ Camera capture stopped.")

    def _create_and_save_event(self, event_data):
        # This function can be offloaded to a queue/thread for more performance
        with self._file_lock:
            # Capture image
            filename = f"event_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            filepath = os.path.join(Config.CAPTURES_DIR, filename)
            local_raw_frame = self.raw_frame.copy() if self.raw_frame is not None else None
            if local_raw_frame is not None:
                cv2.imwrite(filepath, local_raw_frame)
                event_data["image_path"] = filename
            
            # Save to JSON
            try:
                events_list = []
                if os.path.exists(Config.EVENTS_FILE) and os.path.getsize(Config.EVENTS_FILE) > 0:
                    with open(Config.EVENTS_FILE, 'r') as f: events_list = json.load(f)
                events_list.insert(0, event_data)
                with open(Config.EVENTS_FILE, 'w') as f: json.dump(events_list, f, indent=4)
            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ Error writing to {Config.EVENTS_FILE}: {e}")

    def capture_frames(self):
        while self._running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1); continue

            self.frame_counter += 1
            with self._frame_lock: self.raw_frame = frame.copy()
            
            # --- OPTIMIZATION: Resize frame ONCE for all models ---
            inference_frame = cv2.resize(frame, (Config.INFERENCE_SIZE, Config.INFERENCE_SIZE))
            
            processed_frame = frame.copy()
            self.video_recorder.add_frame(self.raw_frame)
            
            current_time = time.time()
            on_cooldown = (current_time - self._last_event_time) < Config.EVENT_COOLDOWN

            # --- LAZY LOADING MODELS ---
            if self.active_modules['surveillance'] and self.object_detector is None:
                print("⏳ Loading Object Detection model...")
                self.object_detector = YOLO(Config.OBJECT_DETECTION_MODEL)
                print("✅ Object Detection model loaded.")
            if self.active_modules['monitor'] and self.pose_estimator is None:
                print("⏳ Loading Monitor Me models (Pose, Face)...")
                self.pose_estimator = YOLO(Config.POSE_ESTIMATION_MODEL)
                self.person_recognizer = PersonRecognizer()
                self.emotion_recognizer = EmotionRecognizer()
                print("✅ Monitor Me models loaded.")

            # ----- SURVEILLANCE MODULE -----
            if self.active_modules['surveillance'] and self.object_detector:
                results = self.object_detector(inference_frame, verbose=False)
                detections_found = []
                for res in results:
                    for box in res.boxes:
                        class_name = self.object_detector.names[int(box.cls[0])]
                        if class_name in Config.SURVEILLANCE_CLASSES:
                            # OPTIMIZATION: Scale bounding box back to original frame size
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1 = int(x1 * Config.CAM_WIDTH / Config.INFERENCE_SIZE), int(y1 * Config.CAM_HEIGHT / Config.INFERENCE_SIZE)
                            x2, y2 = int(x2 * Config.CAM_WIDTH / Config.INFERENCE_SIZE), int(y2 * Config.CAM_HEIGHT / Config.INFERENCE_SIZE)
                            
                            label = f'{class_name} {box.conf[0]:.2f}'
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            detections_found.append(class_name)
                
                if detections_found and not on_cooldown:
                    self._last_event_time = current_time
                    video_path = self.video_recorder.start_recording(self.raw_frame)
                    event = {"id": str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat(), "module": "surveillance",
                             "class_name": f"Object Detected: {', '.join(set(detections_found))}", "video_path": video_path}
                    self._create_and_save_event(event)

            # ----- MONITOR ME MODULE -----
            if self.active_modules['monitor'] and self.pose_estimator:
                results = self.pose_estimator(inference_frame, verbose=False)
                is_falling = False
                person_boxes = []

                for res in results:
                    if res.boxes and len(res.boxes) > 0:
                        for box in res.boxes:
                            w, h = box.xywh[0][2:].cpu().numpy()
                            if w > 0 and h > 0 and (w / h) > 1.4:
                                is_falling = True; color = (0, 0, 255)
                            else: color = (0, 255, 0)
                            
                            # Scale box for drawing and face analysis
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            x1_orig, y1_orig = int(x1 * Config.CAM_WIDTH / Config.INFERENCE_SIZE), int(y1 * Config.CAM_HEIGHT / Config.INFERENCE_SIZE)
                            x2_orig, y2_orig = int(x2 * Config.CAM_WIDTH / Config.INFERENCE_SIZE), int(y2 * Config.CAM_HEIGHT / Config.INFERENCE_SIZE)
                            person_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])

                            activity = "Falling" if is_falling else "Stable"
                            cv2.rectangle(processed_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                            cv2.putText(processed_frame, activity, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # --- OPTIMIZATION: Process expensive tasks every N frames ---
                if person_boxes and (self.frame_counter % Config.PROCESS_INTERVAL == 0):
                    for box in person_boxes:
                        name = self.person_recognizer.recognize(processed_frame, box)
                        emotion = self.emotion_recognizer.analyze(processed_frame, box)
                        cv2.putText(processed_frame, f"{name} ({emotion})", (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if is_falling and not on_cooldown:
                    self._last_event_time = current_time
                    video_path = self.video_recorder.start_recording(self.raw_frame)
                    event = {"id": str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat(), "module": "monitor",
                             "class_name": "Fall Detected", "video_path": video_path}
                    self._create_and_save_event(event)

            with self._frame_lock:
                self.frame = processed_frame

    def start_capture_thread(self):
        threading.Thread(target=self.capture_frames, daemon=True).start()

    def get_camera_info(self): return {"is_running": self._running}

    def run_server(self, host='0.0.0.0', port=5001, debug=False):
        self.app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == "__main__":
    camera = SimpleCameraServer(camera_id=0)
    if camera.start():
        try: camera.run_server()
        finally: camera.stop()