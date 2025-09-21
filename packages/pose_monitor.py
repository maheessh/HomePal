#!/usr/bin/env python3
"""
Pose Monitor System
Detects human poses and activities (falling, standing, sitting, walking)
Uses YOLOv8 Pose NCNN model and streams to monitor frontend
"""

import cv2
import sys
import os
import time
import argparse
from datetime import datetime

# Add services to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
services_path = os.path.join(parent_dir, 'services')
sys.path.append(services_path)

from pose_service import PoseService

class PoseMonitor:
    """Main pose monitoring system"""
    
    def __init__(self, video_source=0, model_path="yolo8spose_ncnn_model"):
        self.video_source = video_source
        self.pose_service = PoseService(model_path)
        self.is_running = False
        self.last_activity_log = {}
        self.log_cooldown = 3  # seconds between same activity logs
        
    def start(self):
        """Start the pose monitoring system"""
        print("👤 Starting Pose Monitor System...")
        print("🤸 Activity detection: YOLOv8 Pose NCNN Model")
        print("📝 Events logged to: events.json")
        print("🌐 Stream available at: /stream")
        print("Press 'q' to quit")
        
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
                
                # Process frame for pose detection
                processed_frame, poses, activities = self.pose_service.detect_poses(frame)
                
                # Log activity events (with cooldown per activity type)
                if activities:
                    for i, activity in enumerate(activities):
                        # Check cooldown for this specific activity
                        if (current_time - self.last_activity_log.get(activity, 0)) > self.log_cooldown:
                            confidence = poses[i]['confidence'] if i < len(poses) else 0.0
                            image_path = self.pose_service.save_frame(processed_frame, activity.lower())
                            
                            self.pose_service.log_event(
                                activity,
                                confidence=confidence,
                                pose_data=poses[i] if i < len(poses) else None,
                                image_path=image_path
                            )
                            
                            self.last_activity_log[activity] = current_time
                            
                            # Special alert for falling
                            if activity == "Falling":
                                print(f"🚨 FALLING DETECTED! Confidence: {confidence:.2f}")
                            else:
                                print(f"👤 {activity} detected - Confidence: {confidence:.2f}")
                
                # Add system info overlay
                self._add_info_overlay(processed_frame, frame_count, len(poses), activities)
                
                # Display frame (for local testing)
                cv2.imshow('Pose Monitor', processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset activity logs
                    self.last_activity_log = {}
                    print("🔄 Activity logs reset")
                
        except KeyboardInterrupt:
            print("\n⏹️ Pose monitoring stopped by user")
        except Exception as e:
            print(f"❌ Pose monitoring error: {e}")
        finally:
            self.stop(cap)
        
        return True
    
    def _add_info_overlay(self, frame, frame_count, num_poses, activities):
        """Add system information overlay to frame"""
        # System status
        status_text = "ACTIVE"
        status_color = (0, 255, 0)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Current activities
        activity_text = ", ".join(set(activities)) if activities else "No Activity"
        if "Falling" in activities:
            status_color = (0, 0, 255)  # Red for falling
            status_text = "ALERT - FALLING"
        
        # Add overlays
        cv2.putText(frame, f"Pose Monitor - {status_text}", 
                   (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Time: {timestamp}", 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Poses: {num_poses}", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Activities: {activity_text}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset logs", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def stop(self, cap=None):
        """Stop the pose monitoring system"""
        self.is_running = False
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("🛑 Pose Monitor System stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pose Monitor System")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (default: 0)")
    parser.add_argument("--model", type=str, default="yolo8spose_ncnn_model", 
                       help="Path to YOLOv8 Pose NCNN model (default: yolo8spose_ncnn_model)")
    
    args = parser.parse_args()
    
    # Create and start pose monitoring system
    pose_monitor = PoseMonitor(
        video_source=args.camera,
        model_path=args.model
    )
    
    pose_monitor.start()

if __name__ == "__main__":
    main()
