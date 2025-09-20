#!/usr/bin/env python3
"""
Simple Flask app that serves index.html and starts camserve
"""

from flask import Flask, render_template, jsonify, request, Response, redirect, url_for
import subprocess
import sys
import os
import requests
import threading
import time

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

# Global variable to track camera server process
camera_process = None

# Global variable to track system states
system_state = {
    'surveillance': False,
    'monitor': False
}

def start_camera_server():
    """Start the camera server in a separate process"""
    global camera_process
    
    try:
        print("🔄 Starting camera server...")
        
        # Start camera server process
        camera_process = subprocess.Popen([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'camserve', 'camserve.py')
        ])
        
        print("✅ Camera server started successfully")
        print("📡 MJPEG stream available at: http://localhost:5001/stream")
        return True
            
    except Exception as e:
        print(f"❌ Error starting camera server: {e}")
        return False

def stop_camera_server():
    """Stop the camera server process"""
    global camera_process
    
    if camera_process:
        try:
            print("🛑 Stopping camera server...")
            
            # First try to stop gracefully by sending SIGTERM
            camera_process.terminate()
            
            # Wait for graceful shutdown
            try:
                camera_process.wait(timeout=3)
                print("✅ Camera server stopped gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️ Camera server didn't stop gracefully, forcing...")
                camera_process.kill()
                try:
                    camera_process.wait(timeout=2)
                    print("✅ Camera server force-stopped")
                except subprocess.TimeoutExpired:
                    print("❌ Failed to stop camera server")
                    
        except Exception as e:
            print(f"❌ Error stopping camera server: {e}")
        finally:
            camera_process = None

@app.route('/')
def index():
    """Redirect to dashboard page"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@app.route('/dashboard.html')
def dashboard():
    """Serve the dashboard page"""
    return render_template('dashboard.html')

@app.route('/surveillance')
@app.route('/surveillance.html')
def surveillance():
    """Serve the surveillance page"""
    return render_template('surveillance.html')

@app.route('/monitor')
@app.route('/monitor.html')
def monitor():
    """Serve the monitor page"""
    return render_template('monitor.html')

@app.route('/stream')
def stream():
    """Proxy camera stream from camserve"""
    try:
        # Check if camera server process is running first
        if not camera_process or camera_process.poll() is not None:
            return jsonify({"error": "Camera server not running"}), 503
            
        # Check if camera server is responding
        response = requests.get('http://localhost:5001/stream', stream=True, timeout=10)
        if response.status_code == 200:
            def generate():
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk
                except (requests.exceptions.ChunkedEncodingError, 
                        requests.exceptions.ConnectionError,
                        ConnectionResetError) as e:
                    print(f"⚠️ Stream connection lost: {e}")
                    return
                    
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return jsonify({"error": "Camera stream not available"}), 404
    except (requests.exceptions.RequestException, ConnectionResetError) as e:
        print(f"⚠️ Stream proxy error: {e}")
        return jsonify({"error": "Camera server not accessible"}), 503

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start the camera server"""
    global camera_process
    
    # Check if camera server is already running and healthy
    if camera_process and camera_process.poll() is None:
        # Verify it's actually responding
        try:
            response = requests.get('http://localhost:5001/api/status', timeout=2)
            if response.status_code == 200:
                return jsonify({"success": True, "message": "Camera already running"})
        except requests.exceptions.RequestException:
            # Process exists but not responding, clean it up
            print("⚠️ Camera process exists but not responding, cleaning up...")
            camera_process = None
    
    try:
        if start_camera_server():
            # Wait for the server to start and verify it's working
            time.sleep(3)
            try:
                response = requests.get('http://localhost:5001/api/status', timeout=5)
                if response.status_code == 200:
                    return jsonify({"success": True, "message": "Camera started successfully"})
                else:
                    return jsonify({"success": False, "message": "Camera started but not responding"}), 500
            except requests.exceptions.RequestException:
                return jsonify({"success": False, "message": "Camera started but not accessible"}), 500
        else:
            return jsonify({"success": False, "message": "Failed to start camera"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Error starting camera: {str(e)}"}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the camera server"""
    global camera_process
    
    try:
        stop_camera_server()
        return jsonify({"success": True, "message": "Camera stopped successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error stopping camera: {str(e)}"}), 500

@app.route('/api/camera/status')
def camera_status():
    """Get camera server status"""
    global camera_process
    
    if camera_process and camera_process.poll() is None:
        try:
            # Try to get status from camera server
            response = requests.get('http://localhost:5001/api/status', timeout=2)
            if response.status_code == 200:
                return jsonify({"success": True, "status": response.json()})
            else:
                return jsonify({"success": False, "message": "Camera server not responding"})
        except requests.exceptions.RequestException:
            return jsonify({"success": False, "message": "Camera server not accessible"})
    else:
        return jsonify({"success": False, "message": "Camera server not running"})

@app.route('/api/system/state')
def get_system_state():
    """Get current system states"""
    global system_state
    return jsonify({"success": True, "state": system_state})

@app.route('/api/system/state', methods=['POST'])
def set_system_state():
    """Update system state"""
    global system_state
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        system = data.get('system')
        state = data.get('state')
        
        if system not in ['surveillance', 'monitor']:
            return jsonify({"success": False, "message": "Invalid system"}), 400
        
        if not isinstance(state, bool):
            return jsonify({"success": False, "message": "State must be boolean"}), 400
        
        system_state[system] = state
        return jsonify({"success": True, "message": f"{system} state updated to {state}"})
        
    except Exception as e:
        return jsonify({"success": False, "message": f"Error updating state: {str(e)}"}), 500

@app.route('/api/events/recent')
def recent_events():
    """Get recent events (mock data for now)"""
    limit = request.args.get('limit', 10, type=int)
    
    # Mock events data - in a real implementation, this would come from a database
    mock_events = [
        {
            "timestamp": int(time.time()) - 30,
            "class_name": "person",
            "confidence": 0.85
        },
        {
            "timestamp": int(time.time()) - 120,
            "class_name": "car",
            "confidence": 0.92
        },
        {
            "timestamp": int(time.time()) - 300,
            "class_name": "person",
            "confidence": 0.78
        }
    ]
    
    return jsonify({"success": True, "events": mock_events[:limit]})

if __name__ == '__main__':
    print("🚀 Starting Aegis AI application...")
    print("📊 Dashboard available at: http://localhost:5000/")
    print("📦 Packages available at: http://localhost:5000/packages")
    
    # Auto-start camera server
    if start_camera_server():
        print("✅ Camera server started automatically")
    else:
        print("⚠️ Failed to start camera server automatically")
        print("📡 You can start it manually: python camserve/camserve.py")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        stop_camera_server()
    except Exception as e:
        print(f"❌ Error: {e}")
        stop_camera_server()
