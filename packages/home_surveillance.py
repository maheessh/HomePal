#!/usr/bin/env python3
"""
Home Surveillance System
Detects fire using Inferno NCNN model and motion detection
Logs events to events.json and streams to surveillance frontend
"""

import cv2
import sys
import os
import time
import argparse
import subprocess
from datetime import datetime

# Add services to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
services_path = os.path.join(parent_dir, 'services')
sys.path.append(services_path)

from detection_service import DetectionService

class HomeSurveillance:
    """Main home surveillance system"""
    
    def __init__(self, video_source=0, model_path="inferno_ncnn_model"):
        self.video_source = video_source
        self.detection_service = DetectionService(model_path)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.is_running = False
        self.last_fire_log = 0
        self.last_motion_log = 0
        self.log_cooldown = 5.0  # 5 second cooldown to reduce excessive logging
        
    def start(self):
        """Start the surveillance system"""
        print("🏠 Starting Home Surveillance System...")
        print("📹 Fire detection: Inferno NCNN Model")
        print("🏃 Motion detection: Background subtraction")
        print("📝 Events logged to: events.json")
        print("🌐 Stream available at: /stream")
        print("Press 'q' to quit")
        # Start fire server
        self._start_fire_server()
        
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source '{self.video_source}'")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Process frame for fire detection
                processed_frame, fire_detected, fire_confidence = self.detection_service.detect_fire(frame)
                
                # Process frame for motion detection
                processed_frame, motion_detected, motion_areas = self.detection_service.detect_motion(
                    processed_frame, self.background_subtractor
                )
                
                # Log fire detection events (with cooldown)
                if fire_detected and (current_time - self.last_fire_log) > self.log_cooldown:
                    image_path = self.detection_service.save_frame(processed_frame, "fire_detection")
                    self.detection_service.log_event(
                        "Fire Detected", 
                        confidence=fire_confidence,
                        image_path=image_path
                    )
                    self.last_fire_log = current_time
                    print(f"🔥 FIRE DETECTED! Confidence: {fire_confidence:.2f}")
                
                # Log significant motion events (with cooldown)
                if motion_detected and len(motion_areas) > 0 and (current_time - self.last_motion_log) > self.log_cooldown:
                    # Check if motion is significant (large enough areas)
                    significant_motion = any(
                        (area[2] * area[3]) > 5000 for area in motion_areas  # area > 5000 pixels
                    )
                    
                    if significant_motion:
                        image_path = self.detection_service.save_frame(processed_frame, "motion_detection")
                        self.detection_service.log_event(
                            "Major Motion Detected",
                            confidence=len(motion_areas) / 10.0,  # Simple confidence based on number of areas
                            motion_areas=motion_areas,
                            image_path=image_path
                        )
                        self.last_motion_log = current_time
                        print(f"🏃 Major motion detected: {len(motion_areas)} areas")
                
                # Display clean frame without overlays (for local testing)
                cv2.imshow('Home Surveillance', processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset background model
                    self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=50, detectShadows=True
                    )
                    print("🔄 Background model reset")
                
        except KeyboardInterrupt:
            print("\n⏹️ Surveillance stopped by user")
        except Exception as e:
            print(f"❌ Surveillance error: {e}")
        finally:
            self.stop(cap)
        
        return True
    
    def _add_info_overlay(self, frame, frame_count, fire_confidence):
        """Add system information overlay to frame"""
        # System status
        status_text = "ACTIVE"
        status_color = (0, 255, 0)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add overlays
        cv2.putText(frame, f"Home Surveillance - {status_text}", 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Time: {timestamp}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Fire Confidence: {fire_confidence:.2f}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def stop(self, cap=None):
        """Stop the surveillance system"""
        self.is_running = False
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("🛑 Home Surveillance System stopped")
        self._stop_fire_server()

    def _start_fire_server(self):
        """Start the dedicated fire detection server in background."""
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            fireserv_path = os.path.join(project_root, 'fireserv.py')
            if not os.path.exists(fireserv_path):
                print("⚠️ Fire server script not found; skipping start.")
                self._fire_proc = None
                return
            python_exec = os.path.join(project_root, 'venv', 'Scripts', 'python.exe') if sys.platform == 'win32' else os.path.join(project_root, 'venv', 'bin', 'python')
            if not os.path.exists(python_exec):
                python_exec = sys.executable
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self._fire_proc = subprocess.Popen([python_exec, fireserv_path], cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creationflags)
            print("🔥 Fire server started in background.")
        except Exception as e:
            print(f"❌ Failed to start fire server: {e}")
            self._fire_proc = None

    def _stop_fire_server(self):
        """Stop the fire detection server if running."""
        try:
            proc = getattr(self, '_fire_proc', None)
            if not proc:
                return
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            self._fire_proc = None
            print("🔥 Fire server stopped.")
        except Exception as e:
            print(f"⚠️ Failed to stop fire server: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Home Surveillance System")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (default: 0)")
    parser.add_argument("--model", type=str, default="inferno_ncnn_model", 
                       help="Path to Inferno NCNN model (default: inferno_ncnn_model)")
    
    args = parser.parse_args()
    
    # Create and start surveillance system
    surveillance = HomeSurveillance(
        video_source=args.camera,
        model_path=args.model
    )
    
    surveillance.start()

if __name__ == "__main__":
    main()
