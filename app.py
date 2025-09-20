#!/usr/bin/env python3
"""
Main Flask app that acts as the central controller for the Aegis AI system.
Controls camera server, manages system state, and serves frontend APIs.
"""

from flask import Flask, render_template, jsonify, request, Response
import subprocess
import sys
import os
import requests
import time
import json
from datetime import datetime

# --- Configuration ---
CAMERA_SERVER_URL = "http://localhost:5001"
EVENTS_FILE = "events.json"

app = Flask(__name__, template_folder="frontend")

# Handle to the camera server process
camera_process = None

# This dictionary is the "single source of truth" for module states
SYSTEM_STATE = {"surveillance": False, "monitor": False}


# --- Camera Process Management ---
def start_camera_server():
    """
    Starts the camera server process if not already running.
    Returns True if the server is alive, False otherwise.
    """
    global camera_process

    if camera_process and camera_process.poll() is None:
        # Already running
        return True

    try:
        print("🔄 Starting camera server process...")

        script_path = os.path.join(os.path.dirname(__file__), "camserve", "camserv.py")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # Launch camserv.py as subprocess
        camera_process = subprocess.Popen(
            [sys.executable, script_path],
            creationflags=creationflags
        )

        # Wait briefly for it to spin up
        time.sleep(3)

        # Ping the /api/status endpoint to confirm it’s alive
        requests.get(f"{CAMERA_SERVER_URL}/api/status", timeout=5).raise_for_status()
        print("✅ Camera server is up and responding.")
        return True

    except Exception as e:
        print(f"❌ Camera server failed to start or respond: {e}")
        stop_camera_server()
        return False


def stop_camera_server():
    """Stops the camera server process if it’s running."""
    global camera_process
    if not camera_process:
        return

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
    """Serve frontend index.html."""
    return render_template("index.html")

# CORRECTED ROUTE
@app.route("/surveillance")
def surveillance():
    """Serve surveillance.html."""
    return render_template("surveillance.html")

@app.route("/monitor")
def monitor():
    """Serve monitor.html."""
    return render_template("monitor.html")

@app.route("/stream")
def stream():
    """
    Proxy the MJPEG stream from the camera server.
    Returns 503 if camera server is unavailable.
    """
    try:
        req = requests.get(f"{CAMERA_SERVER_URL}/stream", stream=True, timeout=10)
        return Response(
            req.iter_content(chunk_size=1024),
            content_type=req.headers.get("content-type", "multipart/x-mixed-replace; boundary=frame")
        )
    except requests.exceptions.RequestException:
        return Response(status=503)


# --- Central State Controller ---
@app.route("/api/system/state", methods=["GET"])
def get_system_state():
    """Returns the current state of all system modules."""
    return jsonify({"success": True, "state": SYSTEM_STATE})


@app.route("/api/module/state", methods=["POST"])
def set_module_state():
    """
    Update system module state (surveillance/monitor).
    Starts/stops camera server based on need.
    """
    global SYSTEM_STATE
    data = request.get_json()
    module = data.get("module")
    active = data.get("active")

    if module not in SYSTEM_STATE:
        return jsonify({"success": False, "message": "Invalid module"}), 400

    SYSTEM_STATE[module] = bool(active)
    print(f"📊 System state updated: {SYSTEM_STATE}")

    # Determine if the camera is needed
    if any(SYSTEM_STATE.values()):
        if not start_camera_server():
            return jsonify({"success": False, "message": "Failed to start camera server"}), 500
        try:
            requests.post(
                f"{CAMERA_SERVER_URL}/api/detection/config",
                json=SYSTEM_STATE,
                timeout=3,
            )
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not sync state with camera server: {e}")
    else:
        stop_camera_server()

    return jsonify({"success": True, "newState": SYSTEM_STATE})


# --- Event Logging and Summary API ---
def read_events_from_file():
    """
    Safely read events.json into memory.
    Returns [] if file is missing, empty, or corrupted.
    """
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        with open(EVENTS_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (IOError, json.JSONDecodeError):
        return []


@app.route("/api/events/all")
def get_all_events():
    """Return all logged events."""
    return jsonify({"success": True, "events": read_events_from_file()})


@app.route("/api/events/summary")
def get_event_summary():
    """Return summary of events (counts + last event timestamp)."""
    events = read_events_from_file()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    summary = {
        "total_events": len(events),
        "events_today": sum(1 for e in events if e.get("timestamp", 0) >= today_start),
        "last_event_time": (
            datetime.fromtimestamp(events[-1]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            if events else "N/A"
        ),
    }
    return jsonify({"success": True, "summary": summary})


# --- Main Entrypoint ---
if __name__ == "__main__":
    try:
        if os.path.exists(EVENTS_FILE):
            os.remove(EVENTS_FILE)
        print("🚀 Starting Aegis AI main controller at http://0.0.0.0:5000")
        app.run(debug=False, host="0.0.0.0", port=5000)
    finally:
        print("\n🛑 Shutting down application...")
        stop_camera_server()

