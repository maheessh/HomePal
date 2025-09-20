#!/usr/bin/env python3
"""
Simple OpenCV-based Camera Server with MJPEG streaming
Gets frames from camera and serves them as MJPEG stream
"""

import cv2
import threading
import time
from typing import Optional
import numpy as np
from flask import Flask, Response, jsonify

class SimpleCameraServer:
    def __init__(self, camera_id: int = 0):
        """
        Initialize the simple camera server.
        
        Args:
            camera_id: Camera device ID (default 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.smframe = None
        self._running = False
        self._frame_lock = threading.Lock()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for MJPEG streaming and API endpoints"""
        
        @self.app.route('/')
        def index():
            """Serve a simple HTML page with the camera stream"""
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Camera Stream</title>
                <style>
                    body { margin: 0; padding: 20px; background: #000; }
                    .container { max-width: 800px; margin: 0 auto; }
                    h1 { color: white; text-align: center; }
                    #stream { width: 100%; height: auto; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Live Camera Feed</h1>
                    <img id="stream" src="/stream" alt="Camera Stream">
                </div>
            </body>
            </html>
            '''
        
        @self.app.route('/stream')
        def mjpeg_stream():
            """MJPEG stream endpoint"""
            def generate_frames():
                while self._running and self.cap:
                    try:
                        with self._frame_lock:
                            if self.frame is not None:
                                # Encode frame as JPEG
                                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                frame_bytes = buffer.tobytes()
                                
                                # MJPEG format: send frame with proper headers
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            else:
                                time.sleep(0.033)
                                continue
                        
                        # Control frame rate (30 FPS)
                        time.sleep(0.033)
                        
                    except Exception as e:
                        print(f"❌ Error in MJPEG stream: {e}")
                        time.sleep(0.1)
            
            return Response(generate_frames(), 
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint to get camera status"""
            return jsonify(self.get_camera_info())
        
        @self.app.route('/api/frame')
        def api_frame():
            """API endpoint to get a single frame as JPEG"""
            with self._frame_lock:
                if self.frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.frame)
                    frame_bytes = buffer.tobytes()
                    return Response(frame_bytes, mimetype='image/jpeg')
                else:
                    return jsonify({"error": "No frame available"}), 404
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            # Try different camera backends on Windows
            backends = [
                cv2.CAP_DSHOW,  # DirectShow (recommended for Windows)
                cv2.CAP_MSMF,   # Media Foundation
                cv2.CAP_ANY     # Auto-detect
            ]
            
            for backend in backends:
                try:
                    print(f"🔍 Trying camera backend: {backend}")
                    self.cap = cv2.VideoCapture(self.camera_id, backend)
                    
                    if self.cap.isOpened():
                        print(f"✅ Camera opened with backend: {backend}")
                        
                        # Test if we can actually read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print("✅ Camera frame test successful")
                            
                            # Set camera properties for better performance
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            
                            self._running = True
                            
                            # Start frame capture thread automatically
                            if self.start_capture_thread():
                                print("✅ Frame capture thread started")
                                return True
                            else:
                                print("❌ Failed to start frame capture thread")
                                self._running = False
                                self.cap.release()
                                self.cap = None
                                return False
                        else:
                            print(f"❌ Camera backend {backend} opened but can't read frames")
                            self.cap.release()
                            self.cap = None
                    else:
                        print(f"❌ Camera backend {backend} failed to open")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                            
                except Exception as e:
                    print(f"❌ Camera backend {backend} error: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    continue
            
            print("❌ All camera backends failed")
            return False
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False
    
    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from camera.
        
        Returns:
            Latest frame or None if no frame available
        """
        with self._frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_smframe(self) -> Optional[np.ndarray]:
        """
        Get the latest 320x320 frame.
        
        Returns:
            Latest 320x320 frame or None if no frame available
        """
        with self._frame_lock:
            return self.smframe.copy() if self.smframe is not None else None
    
    def capture_frames(self):
        """Main frame capture loop - runs in background thread."""
        while self._running and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                time.sleep(0.1)
                continue
            
            # Create 320x320 copy
            smframe = cv2.resize(frame, (320, 320))
            
            # Update frames with thread safety
            with self._frame_lock:
                self.frame = frame
                self.smframe = smframe
    
    def start_capture_thread(self):
        """Start the frame capture in a background thread."""
        if not self._running:
            return False
        
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
        return True
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera properties
        """
        if not self.cap or not self.cap.isOpened():
            return {"error": "Camera not available"}
        
        return {
            "camera_id": self.camera_id,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "is_running": self._running,
            "frame_available": self.frame is not None,
            "smframe_available": self.smframe is not None
        }
    
    def run_server(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask server"""
        print(f"🚀 Starting camera server on http://{host}:{port}")
        print(f"📡 MJPEG stream available at: http://{host}:{port}/stream")
        print(f"📊 Status API available at: http://{host}:{port}/api/status")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down camera server...")
            self.stop()


# Example usage for testing
if __name__ == "__main__":
    # Create camera server
    camera = SimpleCameraServer(camera_id=0)
    
    # Start camera
    if camera.start():
        # Start frame capture thread
        camera.start_capture_thread()
        
        # Wait a moment for frames to start coming in
        time.sleep(1)
        
        try:
            # Start the Flask server
            camera.run_server(host='0.0.0.0', port=5001, debug=False)
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
    else:
        print("❌ Failed to start camera")