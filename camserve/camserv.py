#!/usr/bin/env python3
"""
Camera Server - RESTRUCTURED VERSION
Integrates with the new structured packages:
- home_surveillance.py for fire detection and motion
- pose_monitor.py for activity recognition
- Uses NCNN models for better performance
"""

import cv2
import threading
import time
import datetime
import json
import os
import uuid
import numpy as np
import sys
from flask import Flask, Response, jsonify, request
from collections import deque

# Add packages and services to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
packages_path = os.path.join(parent_dir, 'packages')
services_path = os.path.join(parent_dir, 'services')

sys.path.append(packages_path)
sys.path.append(services_path)

# Import our new services
sys.path.insert(0, services_path)  # Add services to beginning of path
from detection_service import DetectionService
from pose_service import PoseService

class Config:
    """Central configuration for the camera server"""
    # Paths and Files
    EVENTS_FILE = 'events.json'
    CAPTURES_DIR = 'captured_images'
    RECORDINGS_DIR = 'captured_videos'
    
    # Camera Settings
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    CAM_FPS = 30
    
    # Performance Settings
    INFERENCE_INTERVAL = 2  # Run AI inference every N frames
    EVENT_COOLDOWN = 5  # Seconds between same type events
    
    # Model Paths
    INFERNO_MODEL_PATH = 'packages/inferno_ncnn_model'
    POSE_MODEL_PATH = 'packages/yolo8spose_ncnn_model'

class VideoRecorder:
    """Handles video recording when critical events are detected"""
    
    def __init__(self, fps=20.0, duration=10.0, saved_folder=Config.RECORDINGS_DIR):
        self.fps = fps
        self.duration = duration
        self.saved_folder = saved_folder
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Ensure recordings directory exists
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
    
    def start_recording(self, frame):
        """Start recording video from the current frame"""
        if self.is_recording:
            return None
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"event_recording_{timestamp}.mp4"
        video_path = os.path.join(self.saved_folder, video_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, self.fps, 
            (Config.CAM_WIDTH, Config.CAM_HEIGHT)
        )
        
        self.is_recording = True
        self.recording_start_time = time.time()
        
        print(f"[INFO] Started recording: {video_filename}")
        return video_filename
    
    def add_frame(self, frame):
        """Add a frame to the current recording"""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
            
            # Check if recording duration is complete
            if time.time() - self.recording_start_time >= self.duration:
                self.stop_recording()
    
    def stop_recording(self):
        """Stop the current recording"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("[INFO] Recording stopped")

class CameraServer:
    """Main camera server that handles both surveillance and monitoring"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self._running = False
        self._frame_lock = threading.Lock()
        self.frame = None
        self.raw_frame = None
        
        # Module states
        self.active_modules = {'surveillance': False, 'monitor': False}
        
        # Services
        self.detection_service = None
        self.pose_service = None
        
        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Video recorder
        self.video_recorder = VideoRecorder()
        
        # Event tracking
        self.last_event_times = {}
        self.frame_counter = 0
        
        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        print("[INFO] Camera Server initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/stream')
        def stream():
            """Video streaming endpoint"""
            def generate_frames():
                while self._running:
                    with self._frame_lock:
                        if self.frame is not None:
                            ret, buffer = cv2.imencode('.jpg', self.frame, 
                                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + 
                                      frame_bytes + b'\r\n')
                    time.sleep(1/30)  # 30 FPS
            
            return Response(generate_frames(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def status():
            """API status endpoint"""
            return jsonify({
                'status': 'running' if self._running else 'stopped',
                'modules': self.active_modules,
                'camera_info': self.get_camera_info()
            })
        
        @self.app.route('/api/detection/config', methods=['POST'])
        def update_detection_config():
            """Update detection configuration"""
            try:
                data = request.get_json()
                if data:
                    self.active_modules.update(data)
                    print(f"[INFO] Module states updated: {self.active_modules}")
                    
                    # Initialize services based on active modules
                    self._initialize_services()
                    
                return jsonify({'success': True, 'modules': self.active_modules})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 400
    
    def _initialize_services(self):
        """Initialize AI services based on active modules"""
        try:
            # Initialize detection service for surveillance
            if self.active_modules['surveillance'] and self.detection_service is None:
                print("[INFO] Initializing detection service...")
                self.detection_service = DetectionService(Config.INFERNO_MODEL_PATH)
            
            # Initialize pose service for monitoring
            if self.active_modules['monitor'] and self.pose_service is None:
                print("[INFO] Initializing pose service...")
                self.pose_service = PoseService(Config.POSE_MODEL_PATH)
                
        except Exception as e:
            print(f"❌ Error initializing services: {e}")
    
    def _create_and_save_event(self, event):
        """Create and save event to events.json"""
        try:
            events = []
            if os.path.exists(Config.EVENTS_FILE):
                try:
                    with open(Config.EVENTS_FILE, 'r') as f:
                        events = json.load(f)
                except (json.JSONDecodeError, IOError):
                    events = []
            
            # Add new event at the beginning
            events.insert(0, event)
            
            # Keep only last 1000 events
            if len(events) > 1000:
                events = events[:1000]
            
            # Save to file
            with open(Config.EVENTS_FILE, 'w') as f:
                json.dump(events, f, indent=4)
            
            print(f"[INFO] Event logged: {event['class_name']} ({event['module']})")
            
        except Exception as e:
            print(f"❌ Error saving event: {e}")
    
    def _is_on_cooldown(self, event_type):
        """Check if an event type is on cooldown"""
        current_time = time.time()
        last_time = self.last_event_times.get(event_type, 0)
        return (current_time - last_time) < Config.EVENT_COOLDOWN
    
    def _update_event_time(self, event_type):
        """Update the last event time for a specific type"""
        self.last_event_times[event_type] = time.time()
    
    def capture_frames(self):
        """Main frame capture and processing loop"""
        print("[INFO] Starting frame capture thread...")
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Store raw frame
            self.raw_frame = frame.copy()
            
            # Resize frame for processing
            processed_frame = cv2.resize(frame, (Config.CAM_WIDTH, Config.CAM_HEIGHT))
            
            # Add video recorder frame if recording
            if self.video_recorder.is_recording:
                self.video_recorder.add_frame(processed_frame)
            
            # Process frames based on active modules
            if any(self.active_modules.values()):
                processed_frame = self._process_frame(processed_frame)
            
            # Add system info overlay
            processed_frame = self._add_info_overlay(processed_frame)
            
            # Store processed frame
            with self._frame_lock:
                self.frame = processed_frame
            
            self.frame_counter += 1
            time.sleep(1/Config.CAM_FPS)
    
    def _process_frame(self, frame):
        """Process frame with active AI modules"""
        current_time = time.time()
        
        # ----- SURVEILLANCE MODULE -----
        if self.active_modules['surveillance'] and self.detection_service:
            # Run inference every N frames for performance
            if self.frame_counter % Config.INFERENCE_INTERVAL == 0:
                # Fire detection
                frame, fire_detected, fire_confidence = self.detection_service.detect_fire(frame)
                
                if fire_detected and not self._is_on_cooldown('fire'):
                    self._update_event_time('fire')
                    image_path = self.detection_service.save_frame(frame, "fire_detection")
                    self.detection_service.log_event(
                        "Fire Detected",
                        confidence=fire_confidence,
                        image_path=image_path
                    )
                    # Start recording for fire events
                    video_path = self.video_recorder.start_recording(self.raw_frame)
                    print(f"[ALERT] FIRE DETECTED! Confidence: {fire_confidence:.2f}")
                
                # Smoke detection
                frame, smoke_detected, smoke_confidence = self.detection_service.detect_smoke(frame)
                
                if smoke_detected and not self._is_on_cooldown('smoke'):
                    self._update_event_time('smoke')
                    image_path = self.detection_service.save_frame(frame, "smoke_detection")
                    self.detection_service.log_event(
                        "Smoke Detected",
                        confidence=smoke_confidence,
                        image_path=image_path
                    )
                    # Start recording for smoke events
                    video_path = self.video_recorder.start_recording(self.raw_frame)
                    print(f"[ALERT] SMOKE DETECTED! Confidence: {smoke_confidence:.2f}")
                
                # Motion detection
                frame, motion_detected, motion_areas = self.detection_service.detect_motion(
                    frame, self.background_subtractor
                )
                
                if motion_detected and len(motion_areas) > 0 and not self._is_on_cooldown('motion'):
                    # Check for significant motion
                    significant_motion = any((area[2] * area[3]) > 5000 for area in motion_areas)
                    
                    if significant_motion:
                        self._update_event_time('motion')
                        image_path = self.detection_service.save_frame(frame, "motion_detection")
                        self.detection_service.log_event(
                            "Major Motion Detected",
                            confidence=len(motion_areas) / 10.0,
                            motion_areas=motion_areas,
                            image_path=image_path
                        )
                        # Start recording for significant motion
                        video_path = self.video_recorder.start_recording(self.raw_frame)
                        print(f"[INFO] Major motion detected: {len(motion_areas)} areas")
        
        # ----- MONITOR MODULE -----
        if self.active_modules['monitor'] and self.pose_service:
            # Run inference every N frames for performance
            if self.frame_counter % Config.INFERENCE_INTERVAL == 0:
                frame, poses, activities = self.pose_service.detect_poses(frame)
                
                # Log activity events
                for i, activity in enumerate(activities):
                    if not self._is_on_cooldown(f'activity_{activity}'):
                        confidence = poses[i]['confidence'] if i < len(poses) else 0.0
                        image_path = self.pose_service.save_frame(frame, activity.lower())
                        
                        self.pose_service.log_event(
                            activity,
                            confidence=confidence,
                            pose_data=poses[i] if i < len(poses) else None,
                            image_path=image_path
                        )
                        
                        self._update_event_time(f'activity_{activity}')
                        
                        # Special handling for falling
                        if activity == "Falling":
                            video_path = self.video_recorder.start_recording(self.raw_frame)
                            print(f"[ALERT] FALLING DETECTED! Confidence: {confidence:.2f}")
                        else:
                            print(f"[INFO] {activity} detected - Confidence: {confidence:.2f}")
        
        return frame
    
    def _add_info_overlay(self, frame):
        """Add system information overlay to frame"""
        # System status
        active_count = sum(self.active_modules.values())
        status_text = f"ACTIVE ({active_count})" if active_count > 0 else "STANDBY"
        status_color = (0, 255, 0) if active_count > 0 else (255, 255, 0)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Module status
        modules_text = ""
        if self.active_modules['surveillance']:
            modules_text += "SURV "
        if self.active_modules['monitor']:
            modules_text += "MON "
        
        # Recording status
        recording_text = "REC" if self.video_recorder.is_recording else ""
        
        # Add overlays
        cv2.putText(frame, f"HomePal - {status_text}", 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Time: {timestamp}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Modules: {modules_text.strip()}", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_counter} {recording_text}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start(self):
        """Start the camera server"""
        try:
            print(f"[INFO] Starting camera {self.camera_id}...")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, Config.CAM_FPS)
            
            self._running = True
            
            # Start capture thread
            threading.Thread(target=self.capture_frames, daemon=True).start()
            
            print("[SUCCESS] Camera server started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start camera server: {e}")
            return False
    
    def stop(self):
        """Stop the camera server"""
        print("[INFO] Stopping camera server...")
        self._running = False
        
        if self.cap:
            self.cap.release()
        
        if self.video_recorder:
            self.video_recorder.stop_recording()
        
        print("[SUCCESS] Camera server stopped")
    
    def get_camera_info(self):
        """Get camera information"""
        if self.cap:
            return {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "is_running": self._running
            }
        return {"is_running": self._running}
    
    def run_server(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask server"""
        print(f"[INFO] Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    print("[INFO] Starting HomePal Camera Server")
    
    camera = CameraServer(camera_id=0)
    if camera.start():
        try:
            camera.run_server()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped by user")
        finally:
            camera.stop()
    else:
        print("❌ Failed to start camera server")

if __name__ == "__main__":
    main()