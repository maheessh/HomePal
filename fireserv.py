#!/usr/bin/env python3
"""
Fire Server - Runs Inferno NCNN-based fire detection and logs events.
 - Streams processed frames at /stream
 - Status at /api/status
 - Accepts config at /api/detection/config (enable/disable)
Saves captures in captured_images/ and logs to events.json
"""
import os, sys, time, uuid, json, logging, datetime, threading
from typing import Optional
import cv2
from flask import Flask, Response, jsonify, request
import requests
from dotenv import load_dotenv

# Local imports
from services.fire_service import FireService
from services.telegram_service import send_fire_alert


CWD = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("fireserv")

# Load environment variables
load_dotenv()


class Cfg:
    CAPTURES_DIR = "captured_images"
    EVENTS_FILE = "events.json"
    CAM_WIDTH, CAM_HEIGHT = 640, 480
    EVENT_COOLDOWN = 5.0  # 5 second cooldown between fire events to reduce spam
    SAVE_SIZE = 320
    CONF_THRESHOLD = 0.70  # Higher threshold to reduce false positives
    TELEGRAM_COOLDOWN = 30.0  # 30 second cooldown for Telegram notifications
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


class FireServer:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self._frame_lock = threading.Lock()
        self.frame = None
        self.enabled = True

        self.fire_service = FireService(model_dir=os.path.join(CWD, "packages", "inferno_ncnn_model"))

        # IO
        self.captures_path = os.path.join(CWD, Cfg.CAPTURES_DIR)
        os.makedirs(self.captures_path, exist_ok=True)

        self.events_file = os.path.join(CWD, Cfg.EVENTS_FILE)
        self._last_event_time = 0.0
        self._last_telegram_time = 0.0

        # Web
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/stream")
        def stream():
            def gen():
                while self.running:
                    with self._frame_lock:
                        if self.frame is None:
                            time.sleep(0.05)
                            continue
                        ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ok:
                            continue
                        data = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    time.sleep(0.033)
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.route("/api/status")
        def status():
            return jsonify({
                "running": self.running,
                "enabled": self.enabled
            })

        @self.app.route("/api/detection/config", methods=["POST"])
        def config():
            data = request.get_json(silent=True) or {}
            self.enabled = bool(data.get("fire", self.enabled))
            return jsonify({"success": True, "enabled": self.enabled})

    def _save_event_image(self, frame, event_id: str) -> Optional[str]:
        try:
            filename = f"capture_{event_id}.jpg"
            filepath = os.path.join(self.captures_path, filename)
            resized = cv2.resize(frame, (Cfg.SAVE_SIZE, Cfg.SAVE_SIZE))
            cv2.imwrite(filepath, resized)
            log.info(f"📸 Saved capture {filename}")
            return filename
        except Exception as e:
            log.error(f"Failed to save image: {e}")
            return None

    def _append_event(self, event_data: dict):
        try:
            events = {"events": []}
            if os.path.exists(self.events_file) and os.path.getsize(self.events_file) > 0:
                with open(self.events_file, "r") as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, dict) and "events" in content:
                            events = content
                    except json.JSONDecodeError:
                        log.warning("events.json corrupted; resetting structure.")
            events["events"].insert(0, event_data)
            with open(self.events_file, "w") as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            log.error(f"Failed to write event: {e}")

    def _send_telegram_alert(self, message: str):
        """Sends a message using the Telegram Bot API."""
        bot_token = Cfg.TELEGRAM_BOT_TOKEN
        chat_id = Cfg.TELEGRAM_CHAT_ID

        if not all([bot_token, chat_id]):
            log.warning("Telegram credentials not set in .env file. Cannot send Telegram alert.")
            return

        # Construct the API URL
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            log.info(f"✅ Telegram alert sent successfully")
        except requests.exceptions.RequestException as e:
            log.error(f"❌ Failed to send Telegram alert: {e}")

    def start_camera(self) -> bool:
        if self.running:
            return True
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == 'win32' else self.camera_id)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            log.error("Could not open camera")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Cfg.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.CAM_HEIGHT)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log.info("Camera started for FireServer")
        return True

    def stop_camera(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
            self.cap = None
        log.info("Camera stopped")

    def _loop(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            # Use clean frame for streaming (no visual overlays)
            clean_frame = frame.copy()
            if self.enabled:
                # Fire detection without visual overlays
                _, detected, confidence = self.fire_service.detect_fire(frame)
                
                # Debug logging every 60 frames (2 seconds at 30fps) to reduce log spam
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 0
                
                if self._frame_count % 60 == 0:
                    log.info(f"Fire detection: detected={detected}, confidence={confidence:.3f}, threshold={Cfg.CONF_THRESHOLD}")
                
                # Only log events with proper cooldown and higher confidence threshold
                if detected and confidence >= Cfg.CONF_THRESHOLD and (time.time() - self._last_event_time) >= Cfg.EVENT_COOLDOWN:
                    self._last_event_time = time.time()
                    event_id = str(uuid.uuid4())
                    image_filename = self._save_event_image(frame, event_id)
                    self._append_event({
                        "id": event_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "module": "surveillance",
                        "class_name": "Fire",
                        "notification": True,
                        "image_path": image_filename,
                        "description": None,
                        "metadata": {"category": "fire", "alert_level": "critical", "confidence": float(confidence)}
                    })
                    
                    # Send Telegram notification using the dedicated service
                    threading.Thread(target=send_fire_alert, args=(confidence, image_filename), daemon=True).start()
                    log.warning(f"🔥 FIRE DETECTED! Telegram notification queued. Confidence: {confidence:.3f}")

            with self._frame_lock:
                self.frame = clean_frame

    def run(self, host="0.0.0.0", port=5002, debug=False):
        log.info(f"Fire server starting at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    srv = FireServer(0)
    if srv.start_camera():
        try:
            srv.run()
        except KeyboardInterrupt:
            pass
    srv.stop_camera()
    log.info("Fire server stopped")


