import cv2
import os
import sys
from ultralytics import YOLO
from datetime import datetime

# Add the services directory to the path to import EventLogger
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
services_path = os.path.join(project_root, 'services')
sys.path.insert(0, services_path)

try:
    from event_logger import EventLogger
except ImportError:
    print("Warning: Could not import EventLogger. Fall detection will work without logging.")
    EventLogger = None

class FallDetector:
    def __init__(self, model_path='model_ncnn_model'):
        """Initialize the fall detector with YOLO model."""
        self.model = YOLO(model_path, task="pose")
        self.fall_count = 0
        self.frame_count = 0
        self.event_logger = EventLogger() if EventLogger else None
        
        # Create saved directory if it doesn't exist
        if not os.path.exists('saved'):
            os.makedirs('saved')
    
    def detect_fall(self, img):
        """Process a single frame for fall detection."""
        try:
            # Resize frame to 320x320 for model processing
            resized_img = cv2.resize(img, (320, 320))
            
            # Run pose detection on resized image
            results = self.model(resized_img, device="cpu", imgsz=320, verbose=False)
            
            # Get scaling factors to convert from 320x320 back to original image size
            scale_x = img.shape[1] / 320.0
            scale_y = img.shape[0] / 320.0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get bounding box coordinates
                    boxes = result.boxes.xywh.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x, y, w, h = box
                        
                        # Scale coordinates back to original image size
                        x_scaled = x * scale_x
                        y_scaled = y * scale_y
                        w_scaled = w * scale_x
                        h_scaled = h * scale_y
                        
                        # Draw keypoints if available
                        if result.keypoints is not None:
                            keypoints = result.keypoints.xy[0].cpu().numpy()
                            for kpt in keypoints:
                                if kpt[0] > 0 and kpt[1] > 0:  # Valid keypoint
                                    kpt_x = int(kpt[0] * scale_x)
                                    kpt_y = int(kpt[1] * scale_y)
                                    cv2.circle(img, (kpt_x, kpt_y), 3, (0, 255, 0), -1)
                        
                        # Fall detection logic: width/height ratio > 1.4 indicates fall
                        if w > 0 and h > 0 and w/h > 1.4:
                            self.fall_count += 1
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            confidence_ratio = w/h
                            
                            print(f"Fall detected at frame {self.frame_count} - Total falls: {self.fall_count}")
                            
                            # Create annotated image for saving
                            annotated_img = img.copy()
                            
                            # Draw fall indicator on annotated image
                            cv2.putText(annotated_img, "FALL DETECTED!", (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.rectangle(annotated_img, (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                        (int(x_scaled+w_scaled/2), int(y_scaled+h_scaled/2)), (0, 0, 255), 2)
                            
                            # Add confidence info to annotated image
                            cv2.putText(annotated_img, f"Confidence: {confidence_ratio:.2f}", 
                                       (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2) - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Save annotated detection image
                            filename = f"saved/fall_detection_{timestamp}_conf{confidence_ratio:.2f}.jpg"
                            cv2.imwrite(filename, annotated_img)
                            
                            # Log the fall detection event
                            if self.event_logger:
                                self.event_logger.log_fall_detection(
                                    confidence=confidence_ratio,
                                    file_location=filename,
                                    frame_count=self.frame_count
                                )
                            
                            # Draw fall indicator on display image
                            cv2.putText(img, "FALL DETECTED!", (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.rectangle(img, (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                        (int(x_scaled+w_scaled/2), int(y_scaled+h_scaled/2)), (0, 0, 255), 2)
                        else:
                            # Draw stable indicator
                            cv2.putText(img, "Stable", (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(img, (int(x_scaled-w_scaled/2), int(y_scaled-h_scaled/2)), 
                                        (int(x_scaled+w_scaled/2), int(y_scaled+h_scaled/2)), (0, 255, 0), 2)
            
            # Add frame info
            cv2.putText(img, f"Frame: {self.frame_count} | Falls: {self.fall_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error in detection: {e}")
        
        return img
    
    def run_camera(self, camera_index=0):
        """Run fall detection on camera feed."""
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Fall Detection System Started")
        print("Press 'q' to quit, 's' to save current frame")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                self.frame_count += 1
                
                # Process frame for fall detection
                processed_frame = self.detect_fall(frame)
                
                # Display the frame
                cv2.imshow('Fall Detection System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"saved/manual_save_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDetection Summary:")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total falls detected: {self.fall_count}")
            print(f"Fall detection rate: {self.fall_count/max(self.frame_count, 1)*100:.2f}%")


def main():
    """Main function to run the fall detection system."""
    detector = FallDetector()
    
    # Try different camera indices if needed
    for camera_index in [0, 1, 2]:
        print(f"Attempting to open camera {camera_index}...")
        detector.run_camera(camera_index)
        break


if __name__ == "__main__":
    main()