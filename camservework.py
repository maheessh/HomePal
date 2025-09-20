#!/usr/bin/env python3
"""
Camera Server that detects motion (Surveillance) and human presence (Monitor Me),
saving events to a JSON file based on which modules are active.
"""

import cv2
import threading
import time
from flask import Flask, Response, jsonify, request
import json
import os

# Define the path for our event log file, which app.py will also use
EVENTS_FILE = 'events.json'

class SimpleCameraServer:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self._running = False
        self._frame_lock = threading.Lock()
        
        # Controlled externally via app.py
        self.active_modules = {"surveillance": False, "monitor": False}

        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Haar cascade for human presence detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._file_lock = threading.Lock()
        self._last_event_time = 0  # shared cooldown for events

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
                        time.sleep(0.033)  # ~30 FPS
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
    
    def write_event_to_file(self, event):
        """Safely append a single event to events.json."""
        with self._file_lock:
            try:
                events_list = []
                if os.path.exists(EVENTS_FILE) and os.path.getsize(EVENTS_FILE) > 0:
                    with open(EVENTS_FILE, 'r') as f:
                        events_list = json.load(f)
                
                events_list.insert(0, event)
                with open(EVENTS_FILE, 'w') as f:
                    json.dump(events_list, f, indent=4)
            except (IOError, json.JSONDecodeError) as e:
                print(f"❌ Error writing to {EVENTS_FILE}: {e}")

    def capture_frames(self):
        """Frame capture loop. Detects motion (surveillance) and human presence (monitor)."""
        while self._running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            processed_frame = frame.copy()
            current_time = time.time()
            event_logged = False

            # --- Surveillance: Motion Detection ---
            if self.active_modules['surveillance']:
                fg_mask = self.background_subtractor.apply(processed_frame)
                fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
                fg_mask = cv2.dilate(fg_mask, None, iterations=2)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = any(cv2.contourArea(c) > 1000 for c in contours)
                if motion_detected:
                    for c in contours:
                        if cv2.contourArea(c) > 1000:
                            (x, y, w, h) = cv2.boundingRect(c)
                            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if current_time - self._last_event_time > 5:
                        self._last_event_time = current_time
                        self.write_event_to_file({"timestamp": int(current_time), "module": "surveillance", "class_name": "Motion"})
                        event_logged = True

            # --- Monitor Me: Human Presence Detection ---
            if self.active_modules['monitor']:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                if len(faces) > 0 and (current_time - self._last_event_time > 5):
                    self._last_event_time = current_time
                    self.write_event_to_file({"timestamp": int(current_time), "module": "monitor", "class_name": "Person"})
                    event_logged = True

            with self._frame_lock:
                self.frame = processed_frame
    
    def start_capture_thread(self):
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
    
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
