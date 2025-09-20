#!/usr/bin/env python3
"""
Main Flask app that acts as the central controller for the Aegis AI system.
Controls camera server, manages system state, and serves frontend APIs.
Now includes a background worker for AI image descriptions and serves static images.
"""
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import subprocess
import sys
import os
import requests
import time
import json
import threading
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
CAMERA_SERVER_URL = "http://localhost:5001"
EVENTS_FILE = "events.json"
CAPTURES_DIR = "captured_images"

app = Flask(__name__, template_folder="frontend", static_folder="captured_images")
CORS(app) # Enable Cross-Origin Resource Sharing

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print("✅ Gemini AI configured successfully.")
except Exception as e:
    print(f"⚠️ Gemini configuration failed. Check your GOOGLE_API_KEY. Error: {e}")

# Handle to the camera server process
camera_process = None
SYSTEM_STATE = {"surveillance": False, "monitor": False}
events_lock = threading.Lock()

# --- Gemini AI Description Worker ---
def get_image_description(image_path):
    """Generates a description for an image using Gemini."""
    try:
        full_path = os.path.join(CAPTURES_DIR, image_path)
        if not os.path.exists(full_path):
            return "Error: Image file not found."

        img = Image.open(full_path)
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(["Describe this scene briefly for a security log.", img])
        response.resolve()
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"❌ Error generating description for {image_path}: {e}")
        return "Description could not be generated."

def description_worker():
    """A worker thread that processes images to add AI descriptions."""
    print("🤖 AI Description worker started.")
    while True:
        time.sleep(10) # Check for new events every 10 seconds
        
        events_to_process = []
        with events_lock:
            if not os.path.exists(EVENTS_FILE):
                continue
            try:
                with open(EVENTS_FILE, 'r') as f:
                    all_events = json.load(f)
                
                # Find events that need a description
                for event in all_events:
                    if "description" in event and event["description"] == "" and event.get("image_path"):
                        events_to_process.append(event)
            except (json.JSONDecodeError, IOError):
                continue # File might be busy, try again later

        if not events_to_process:
            continue

        print(f"Found {len(events_to_process)} events needing description.")
        
        # This part modifies the list outside the lock to avoid holding it for too long
        modified = False
        for event in events_to_process:
            print(f"Generating description for {event['id']}...")
            description = get_image_description(event['image_path'])
            
            # Update the specific event in the main list
            for original_event in all_events:
                if original_event['id'] == event['id']:
                    original_event['description'] = description
                    modified = True
                    break
        
        if modified:
            with events_lock:
                with open(EVENTS_FILE, 'w') as f:
                    json.dump(all_events, f, indent=4)
            print("✅ Events file updated with new descriptions.")


# --- Camera Process Management (IMPROVED) ---
def start_camera_server():
    """
    Starts the camera server process if not already running.
    Captures and prints stdout/stderr from the subprocess for better debugging.
    Returns True if the server is alive, False otherwise.
    """
    global camera_process

    if camera_process and camera_process.poll() is None:
        return True

    try:
        print("🔄 Starting camera server process...")
        script_path = os.path.join(os.path.dirname(__file__), "camserve", "camserv.py")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # Launch camserv.py as a subprocess, capturing its output
        camera_process = subprocess.Popen(
            [sys.executable, script_path],
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Decode stdout/stderr as text
        )

        print("...waiting for camera server to initialize (10s)...")
        time.sleep(10)

        # Check if the process terminated early
        if camera_process.poll() is not None:
            # It crashed. Read and print its output.
            stdout, stderr = camera_process.communicate()
            print("❌ Camera server process terminated unexpectedly.")
            if stdout:
                print("--- Camera Server Output (stdout) ---")
                print(stdout)
            if stderr:
                print("--- Camera Server Error (stderr) ---")
                print(stderr)
            return False

        # Ping the /api/status endpoint to confirm it’s alive
        requests.get(f"{CAMERA_SERVER_URL}/api/status", timeout=5).raise_for_status()
        print("✅ Camera server is up and responding.")
        return True

    except Exception as e:
        print(f"❌ Camera server failed to start or respond: {e}")
        # If it failed, read any output from the process
        if camera_process:
            stdout, stderr = camera_process.communicate()
            if stdout:
                print("--- Camera Server Output (stdout) ---")
                print(stdout)
            if stderr:
                print("--- Camera Server Error (stderr) ---")
                print(stderr)
        stop_camera_server()
        return False

def stop_camera_server():
    global camera_process
    if not camera_process: return
    print("🛑 Stopping camera server process...")
    try:
        camera_process.terminate()
        camera_process.wait(timeout=5)
    except Exception:
        camera_process.kill()
    finally:
        camera_process = None

# --- Frontend & Camera Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/surveillance")
def surveillance():
    return render_template("surveillance.html")

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")

@app.route("/stream")
def stream():
    try:
        req = requests.get(f"{CAMERA_SERVER_URL}/stream", stream=True, timeout=10)
        return Response(
            req.iter_content(chunk_size=1024),
            content_type=req.headers.get("content-type")
        )
    except requests.exceptions.RequestException:
        return Response("Camera server is not available.", status=503)

# NEW: Route to serve captured images
@app.route('/captured_images/<path:filename>')
def serve_captured_image(filename):
    return send_from_directory(CAPTURES_DIR, filename)

# --- Central State Controller (No changes needed) ---
@app.route("/api/system/state", methods=["GET"])
def get_system_state():
    return jsonify({"success": True, "state": SYSTEM_STATE})

@app.route("/api/module/state", methods=["POST"])
def set_module_state():
    global SYSTEM_STATE
    data = request.get_json()
    module, active = data.get("module"), data.get("active")
    if module not in SYSTEM_STATE:
        return jsonify({"success": False, "message": "Invalid module"}), 400

    SYSTEM_STATE[module] = bool(active)
    print(f"📊 System state updated: {SYSTEM_STATE}")

    if any(SYSTEM_STATE.values()):
        if not start_camera_server():
            return jsonify({"success": False, "message": "Failed to start camera"}), 500
        try:
            requests.post(f"{CAMERA_SERVER_URL}/api/detection/config", json=SYSTEM_STATE, timeout=3)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not sync state with camera server: {e}")
    else:
        stop_camera_server()

    return jsonify({"success": True, "newState": SYSTEM_STATE})

# --- Event Logging and Summary API ---
def read_events_from_file():
    with events_lock:
        if not os.path.exists(EVENTS_FILE):
            return []
        try:
            with open(EVENTS_FILE, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except (IOError, json.JSONDecodeError):
            return []

# MODIFIED: More flexible event fetching
@app.route("/api/events")
def get_events():
    """Return events, optionally filtered by module."""
    module_filter = request.args.get("module")
    all_events = read_events_from_file()
    if module_filter:
        filtered = [e for e in all_events if e.get("module") == module_filter]
        return jsonify({"success": True, "events": filtered})
    return jsonify({"success": True, "events": all_events})

@app.route("/api/events/all")
def get_all_events_deprecated():
     return get_events()

@app.route("/api/events/summary")
def get_event_summary():
    events = read_events_from_file()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    events_today = 0
    for e in events:
        try:
            event_time = datetime.fromisoformat(e.get("timestamp", "1970-01-01T00:00:00"))
            if event_time >= today_start:
                events_today += 1
        except (ValueError, TypeError):
            continue
            
    last_event_time_str = "N/A"
    if events:
        try:
           last_event_time_str = datetime.fromisoformat(events[0]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
           last_event_time_str = "Invalid Date"

    summary = {
        "total_events": len(events),
        "events_today": events_today,
        "last_event_time": last_event_time_str,
    }
    return jsonify({"success": True, "summary": summary})

# --- Main Entrypoint ---
if __name__ == "__main__":
    try:
        # Start the background worker
        worker = threading.Thread(target=description_worker, daemon=True)
        worker.start()
        
        print("🚀 Starting Aegis AI main controller at http://0.0.0.0:5000")
        app.run(debug=False, host="0.0.0.0", port=5000)
    finally:
        print("\n🛑 Shutting down application...")
        stop_camera_server()