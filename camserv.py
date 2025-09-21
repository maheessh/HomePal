#!/usr/bin/env python3
"""
Camera Server - FULLY INTEGRATED VERSION
This version performs real-time event detection, saves image captures for events,
logs events with structured metadata, and includes an SMS alert system.
"""
import os, sys, time, uuid, json, logging, datetime, threading
from typing import Optional
import cv2
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
from dotenv import load_dotenv
from twilio.rest import Client
from services.fire_service import FireService

# --- Setup ---
CWD=os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log=logging.getLogger("camserv")

# --- MODIFIED: Load .env from the project root for consistency ---
load_dotenv(os.path.join(CWD, '.env'))

# --- Configuration ---
class C:
    """Defines the core operational parameters for the server."""
    EVENTS_FILE = "events.json"
    # --- NEW: Image capture directory, relative to project root ---
    CAPTURES_DIR = "captured_images"

    CAM_WIDTH, CAM_HEIGHT = 640, 480
    INFERENCE_SIZE = 320
    EVENT_COOLDOWN = 0  # No cooldown - log events immediately

    # Motion Detection Parameters
    MOTION_TRIGGER_SCAN_DURATION = 10
    MOTION_THRESHOLD = 30
    MOTION_MIN_AREA = 500

    # AI Model Files
    OBJECT_DETECTION_MODEL = "yolov8n.pt"
    POSE_ESTIMATION_MODEL = "yolov8n-pose.pt"

    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    RECIPIENT_NUMBER = os.getenv("RECIPIENT_NUMBER")

# --- SMS Sending Function ---
def send_emergency_message(message: str):
    """Sends an SMS in a separate thread to avoid blocking the main loop."""
    def sender():
        if not all([C.TWILIO_ACCOUNT_SID, C.TWILIO_AUTH_TOKEN, C.TWILIO_PHONE_NUMBER, C.RECIPIENT_NUMBER]):
            log.warning("Twilio credentials not set in .env file. Cannot send SMS.")
            return
        try:
            client = Client(C.TWILIO_ACCOUNT_SID, C.TWILIO_AUTH_TOKEN)
            msg = client.messages.create(
                body=message,
                from_=C.TWILIO_PHONE_NUMBER,
                to=C.RECIPIENT_NUMBER
            )
            log.info(f"✅ SMS alert sent successfully: {msg.sid}")
        except Exception as e:
            log.error(f"❌ Failed to send SMS: {e}")

    threading.Thread(target=sender, daemon=True).start()

# --- Main Server Class ---
class S:
    """The primary class that manages camera capture, processing, and the API."""
    def __init__(self, cid=0):
        self.cid = cid
        self.cap = None
        self.running = False
        self._frame_lock = threading.Lock()
        self.frame = None

        self.mods = {"surveillance": False, "monitor": False, "fire": False}
        self._last_event = 0
        self.fall_detected_time = None
        self.fall_alert_triggered = False
        self.motion_end_time = 0

        self.od_model = None
        self.pose_model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
        self.fire_service = FireService(model_dir=os.path.join(CWD, "packages", "inferno_ncnn_model"))

        # --- FIX: Ensure captures directory exists ---
        self.captures_path = os.path.join(CWD, C.CAPTURES_DIR)
        os.makedirs(self.captures_path, exist_ok=True)
        log.info(f"Capture directory set to: {self.captures_path}")

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Initializes the API endpoints for the server."""
        @self.app.route("/stream")
        def stream_generator():
            def frame_generator():
                while self.running:
                    with self._frame_lock:
                        if self.frame is None:
                            time.sleep(0.05)
                            continue
                        ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok: continue
                        data = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    time.sleep(0.033)
            return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.route("/api/status")
        def get_status():
            return jsonify(self.get_server_info())

        @self.app.route("/api/detection/config", methods=["POST"])
        def configure_detection():
            config_data = request.get_json(silent=True) or {}
            self.mods["surveillance"] = bool(config_data.get("surveillance", self.mods["surveillance"]))
            self.mods["monitor"] = bool(config_data.get("monitor", self.mods["monitor"]))
            # Fire can be controlled independently; app may couple it to surveillance
            self.mods["fire"] = bool(config_data.get("fire", self.mods["fire"]))
            log.info(f"Detection modules updated: {self.mods}")
            return jsonify({"success": True, "active_modules": self.mods})

    def start_camera(self):
        if self.running: return True
        self.cap = cv2.VideoCapture(self.cid, cv2.CAP_DSHOW if sys.platform == 'win32' else self.cid)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(self.cid)
        if not self.cap.isOpened():
            log.error("Failed to open camera.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_HEIGHT)
        self.running = True
        threading.Thread(target=self._main_loop, daemon=True).start()
        log.info("Camera capture thread started.")
        return True

    def stop_camera(self):
        self.running = False
        time.sleep(0.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        log.info("Camera stopped.")

    def _ensure_model_loaded(self, model_type):
        if model_type == 'od' and self.od_model is None:
            try:
                self.od_model = YOLO(C.OBJECT_DETECTION_MODEL)
                log.info("Object detection model loaded.")
            except Exception as e: log.error(f"Failed to load object detection model: {e}")
        elif model_type == 'pose' and self.pose_model is None:
            try:
                self.pose_model = YOLO(C.POSE_ESTIMATION_MODEL)
                log.info("Pose estimation model loaded.")
            except Exception as e: log.error(f"Failed to load pose estimation model: {e}")
    
    # --- NEW: Function to save an image for an event ---
    def _save_event_image(self, frame, event_id: str) -> Optional[str]:
        """Saves an image capture for an event and returns the filename."""
        try:
            filename = f"capture_{event_id}.jpg"
            filepath = os.path.join(self.captures_path, filename)
            cv2.imwrite(filepath, frame)
            log.info(f"📸 Image captured for event: {filename}")
            return filename
        except Exception as e:
            log.error(f"Failed to save event image: {e}")
            return None

    def _save_event_log(self, event_data: dict):
        try:
            events = {"events": []}
            if os.path.exists(C.EVENTS_FILE) and os.path.getsize(C.EVENTS_FILE) > 0:
                with open(C.EVENTS_FILE, "r") as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, dict) and "events" in content:
                            events = content
                    except json.JSONDecodeError:
                        log.warning("events.json is corrupted; starting fresh.")

            events["events"].insert(0, event_data)
            with open(C.EVENTS_FILE, "w") as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            log.error(f"Failed to write to event log: {e}")

    def _main_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            processed_frame = frame.copy()
            on_cooldown = (time.time() - self._last_event) < C.EVENT_COOLDOWN

            if self.mods["monitor"]:
                processed_frame = self._monitor_module(processed_frame, frame, on_cooldown)
            if self.mods["surveillance"]:
                processed_frame = self._surveillance_module(processed_frame, frame, on_cooldown)
            if self.mods.get("fire"):
                processed_frame = self._fire_module(processed_frame, frame, on_cooldown)

            with self._frame_lock:
                self.frame = processed_frame
    
    # --- MODIFIED: surveillance_module now saves an image ---
    def _surveillance_module(self, processed_frame, original_frame, on_cooldown):
        fg_mask = self.bg_subtractor.apply(processed_frame)
        _, fg_mask = cv2.threshold(fg_mask, C.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = any(cv2.contourArea(c) > C.MOTION_MIN_AREA for c in contours)
        is_scanning = time.time() < self.motion_end_time

        if motion_detected and not is_scanning and not on_cooldown:
            self.motion_end_time = time.time() + C.MOTION_TRIGGER_SCAN_DURATION
            self._last_event = time.time()
            event_id = str(uuid.uuid4())
            
            # Save image and get filename
            image_filename = self._save_event_image(original_frame, event_id)

            self._save_event_log({
                "id": event_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "module": "surveillance",
                # FIX: Use class_name for consistency with frontend
                "class_name": "Motion", 
                "notification": True,
                "image_path": image_filename,
                "description": None, # Placeholder for AI worker
                # NEW: Metadata for frontend filtering
                "metadata": {"category": "motion", "alert_level": "low"}
            })
            is_scanning = True

        if is_scanning:
            cv2.putText(processed_frame, "AI SCANNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return processed_frame
    
    # --- MODIFIED: monitor_module now saves an image ---
    def _monitor_module(self, processed_frame, original_frame, on_cooldown):
        self._ensure_model_loaded('pose')
        if not self.pose_model: return processed_frame

        try:
            results = self.pose_model(cv2.resize(processed_frame, (C.INFERENCE_SIZE, C.INFERENCE_SIZE)), verbose=False)
        except Exception as e:
            log.error(f"Pose model inference failed: {e}")
            return processed_frame

        is_currently_falling = False
        if results and results[0].boxes:
            for b in results[0].boxes:
                xywh = b.xywh[0].cpu().numpy()
                w, h = float(xywh[2]), float(xywh[3])
                is_fallen_pose = w > h * 1.4

                xy = b.xyxy[0].cpu().numpy()
                scale_w, scale_h = C.CAM_WIDTH / C.INFERENCE_SIZE, C.CAM_HEIGHT / C.INFERENCE_SIZE
                x1, y1 = int(xy[0] * scale_w), int(xy[1] * scale_h)
                x2, y2 = int(xy[2] * scale_w), int(xy[3] * scale_h)

                color = (0, 0, 255) if is_fallen_pose else (0, 255, 0)
                activity = "Falling" if is_fallen_pose else "Stable"
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if is_fallen_pose:
                    is_currently_falling = True

        if is_currently_falling:
            if self.fall_detected_time is None:
                self.fall_detected_time = time.time()
            
            elapsed = time.time() - self.fall_detected_time
            if elapsed >= 5 and not self.fall_alert_triggered and not on_cooldown: # Reduced time for testing
                log.warning("EMERGENCY: Fall confirmed for over 5 seconds!")
                self._last_event = time.time()
                self.fall_alert_triggered = True
                
                timestamp = datetime.datetime.now()
                event_id = str(uuid.uuid4())
                image_filename = self._save_event_image(original_frame, event_id)

                self._save_event_log({
                    "id": event_id,
                    "timestamp": timestamp.isoformat(),
                    "module": "monitor",
                    "class_name": "Fall",
                    "notification": True,
                    "image_path": image_filename,
                    "description": None,
                    "metadata": {"category": "health", "alert_level": "critical"}
                })
                send_emergency_message(f"HomePal Alert: FALL detected at {timestamp.strftime('%H:%M:%S')}.")
        else:
            self.fall_detected_time = None
            self.fall_alert_triggered = False
            
        return processed_frame

    def _fire_module(self, processed_frame, original_frame, on_cooldown):
        vis, detected, confidence = self.fire_service.detect_fire(processed_frame)
        if detected and not on_cooldown:
            self._last_event = time.time()
            event_id = str(uuid.uuid4())
            image_filename = self._save_event_image(original_frame, event_id)

            self._save_event_log({
                "id": event_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "module": "surveillance",
                "class_name": "Fire",
                "notification": True,
                "image_path": image_filename,
                "description": None,
                "metadata": {"category": "fire", "alert_level": "critical", "confidence": float(confidence)}
            })

        return vis

    def get_server_info(self):
        return {
            "running": self.running,
            "modules": self.mods,
            "models": {
                "object_detection": self.od_model is not None,
                "pose_estimation": self.pose_model is not None
            }
        }

    def run_server(self, host="0.0.0.0", port=5001, debug=False):
        log.info(f"Flask server starting on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# --- Main Entry Point ---
if __name__ == "__main__":
    log.info("Initializing Camera Server...")
    server = S(0)
    
    if server.start_camera():
        try:
            # Set initial state for testing if desired
            # server.mods["monitor"] = True
            # server.mods["surveillance"] = True
            server.run_server()
        except KeyboardInterrupt:
            log.info("Shutdown signal received.")
    else:
        log.error("Could not start camera. Shutting down.")
        
    server.stop_camera()
    log.info("Camera Server has shut down.")