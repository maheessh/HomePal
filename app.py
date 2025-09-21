#!/usr/bin/env python3
"""
Main Flask app that acts as the central controller for the Aegis AI system.
- Manages camera server lifecycle.
- Handles system state for detection modules.
- Serves all frontend pages and static assets.
- Provides API endpoints for events and summaries.
- Includes a secure proxy for Gemini AI calls.
- Implements a new backend for the medication reminder feature.
"""
import subprocess
import sys
import os
import requests
import time
import json
import threading
import uuid
from datetime import datetime, time as time_obj
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from services.openrouter_service import OpenRouterService
from services.tts_service import TTSService
from services.medication_service import MedicationReminderService
from services.summary_service import SummaryService

# --- Configuration ---
load_dotenv()
CWD = os.path.dirname(os.path.realpath(__file__))
CAMERA_SERVER_URL = "http://localhost:5001"
EVENTS_FILE = "events.json"
# NEW: Medications database file
MEDICATIONS_FILE = "medications.json" 
CAPTURES_DIR = "captured_images"
TTS_DIR = "tts_audio"

# --- Initialization ---
app = Flask(__name__, template_folder="frontend", static_folder="captured_images")
CORS(app)

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print("✅ Gemini AI configured successfully.")
except Exception as e:
    print(f"⚠️ Gemini configuration failed. Check your GOOGLE_API_KEY. Error: {e}")

# Configure OpenRouter
try:
    openrouter_service = OpenRouterService()
    print("✅ OpenRouter service configured successfully.")
except Exception as e:
    print(f"⚠️ OpenRouter configuration failed. Check your OPENROUTER_API_KEY. Error: {e}")
    openrouter_service = None

# Configure TTS service
tts_service = TTSService(output_dir=TTS_DIR)
summary_service = SummaryService()

# Configure Medication Reminder Service
try:
    medication_service = MedicationReminderService(MEDICATIONS_FILE)
    medication_service.start()
    print("✅ Medication reminder service configured successfully.")
except Exception as e:
    print(f"⚠️ Medication reminder service configuration failed. Error: {e}")
    medication_service = None

camera_process = None
SYSTEM_STATE = {"surveillance": False, "monitor": False, "fire": False}
events_lock = threading.Lock()
# NEW: Lock for medication file access
meds_lock = threading.Lock()


# --- Gemini AI Description Worker ---
def get_image_description(image_path: str):
    try:
        full_path = os.path.join(CWD, CAPTURES_DIR, image_path)
        if not os.path.exists(full_path): return "Error: Image file not found."

        img = Image.open(full_path)
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(["Describe this scene briefly for a security log.", img])
        response.resolve()
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"❌ Error generating description for {image_path}: {e}")
        return "Description could not be generated."

def description_worker():
    print("🤖 AI Description worker started.")
    while True:
        time.sleep(10)
        
        full_data = None
        events_to_process = []
        with events_lock:
            if not os.path.exists(EVENTS_FILE): continue
            try:
                with open(EVENTS_FILE, 'r') as f:
                    full_data = json.load(f)
                    all_events = full_data.get("events", [])
                    # Find events needing a description
                    for event in all_events:
                        if event.get("description") is None and event.get("image_path"):
                            events_to_process.append(event)
            except (json.JSONDecodeError, IOError): continue

        if not events_to_process: continue
        print(f"Found {len(events_to_process)} events needing description.")
        
        modified = False
        for event in events_to_process:
            print(f"Generating description for {event['id']}...")
            description = get_image_description(event['image_path'])
            for original_event in full_data.get("events", []):
                if original_event['id'] == event['id']:
                    original_event['description'] = description
                    modified = True
                    break
        
        if modified:
            with events_lock:
                with open(EVENTS_FILE, 'w') as f:
                    json.dump(full_data, f, indent=2)
            print("✅ Events file updated with new descriptions.")


# --- Camera Process Management ---
def start_camera_server():
    global camera_process
    if camera_process and camera_process.poll() is None: return True

    try:
        print("🔄 Starting camera server process...")
        script_path = os.path.join(CWD, "camserv.py")
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        
        camera_process = subprocess.Popen(
            [sys.executable, script_path], creationflags=creationflags,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        print("...waiting for camera server to initialize (10s)...")
        time.sleep(10)

        if camera_process.poll() is not None:
            stdout, stderr = camera_process.communicate()
            print("❌ Camera server terminated unexpectedly.")
            if stdout: print(f"--- Camserv stdout ---\n{stdout}")
            if stderr: print(f"--- Camserv stderr ---\n{stderr}")
            return False

        requests.get(f"{CAMERA_SERVER_URL}/api/status", timeout=5).raise_for_status()
        print("✅ Camera server is up and responding.")
        return True
    except Exception as e:
        print(f"❌ Camera server failed to start or respond: {e}")
        if camera_process:
            stdout, stderr = camera_process.communicate()
            if stdout: print(f"--- Camserv stdout ---\n{stdout}")
            if stderr: print(f"--- Camserv stderr ---\n{stderr}")
        stop_camera_server()
        return False

def stop_camera_server():
    global camera_process
    if not camera_process: return
    print("🛑 Stopping camera server process...")
    try:
        camera_process.terminate()
        camera_process.wait(timeout=5)
    except Exception: camera_process.kill()
    finally: camera_process = None


# --- Frontend & Camera Routes ---
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
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
        return Response(req.iter_content(chunk_size=1024), content_type=req.headers['content-type'])
    except requests.exceptions.RequestException:
        return Response("Camera server is not available.", status=503)

@app.route('/captured_images/<path:filename>')
def serve_captured_image(filename):
    return send_from_directory(os.path.join(CWD, CAPTURES_DIR), filename)

@app.route('/tts/<path:filename>')
def serve_tts_file(filename):
    return send_from_directory(os.path.join(CWD, TTS_DIR), filename)


# --- Central State Controller ---
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
    # Ensure fire runs alongside surveillance from the single UI toggle
    if module == "surveillance":
        SYSTEM_STATE["fire"] = bool(active)
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
        if not os.path.exists(EVENTS_FILE): return []
        try:
            with open(EVENTS_FILE, "r") as f:
                content = f.read().strip()
                if not content: return []
                data = json.loads(content)
                return data.get("events", [])
        except (IOError, json.JSONDecodeError): return []

@app.route("/api/events")
def get_events():
    module_filter = request.args.get("module")
    all_events = read_events_from_file()
    if module_filter:
        filtered = [e for e in all_events if e.get("module") == module_filter]
        return jsonify({"success": True, "events": filtered})
    return jsonify({"success": True, "events": all_events})

@app.route("/api/events/summary")
def get_event_summary():
    events = read_events_from_file()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events_today = 0
    for e in events:
        try:
            if datetime.fromisoformat(e.get("timestamp", "1970-01-01T00:00:00")) >= today_start:
                events_today += 1
        except Exception:
            continue

    def _fmt_time(dt: datetime) -> str:
        try:
            return dt.strftime('%-I:%M %p')
        except ValueError:
            # Windows fallback (no '-' flag support)
            return dt.strftime('%I:%M %p').lstrip('0')

    last_event_time_str = "N/A"
    if events:
        try:
            dt = datetime.fromisoformat(events[0].get("timestamp", ""))
            last_event_time_str = _fmt_time(dt)
        except Exception:
            last_event_time_str = "N/A"

    return jsonify({"success": True, "summary": {
        "total_events": len(events),
        "events_today": events_today,
        "last_event_time": last_event_time_str,
    }})

@app.route("/api/generate_audio_briefing", methods=["POST"])
def generate_audio_briefing():
    """Generate a one-sentence briefing and return an audio URL for gTTS output."""
    try:
        events = read_events_from_file()

        # Default fallback sentence if AI unavailable or no events
        fallback_sentence = "All quiet at home with no concerning activity."
        if events:
            try:
                # Simple human fallback if AI missing
                latest = events[0]
                module = latest.get("module", "system")
                cls = latest.get("class_name", "activity")
                fallback_sentence = f"Latest update: {module} detected {cls.lower()}."
            except Exception:
                pass

        sentence = fallback_sentence
        model_used = "fallback"

        if openrouter_service:
            result = openrouter_service.generate_micro_briefing(events)
            if result.get("success"):
                sentence = result.get("sentence", sentence)
                model_used = result.get("model_used", "google/gemini-2.0-flash-001")
            else:
                model_used = f"fallback ({result.get('error', 'ai_error')})"

        # Synthesize audio
        filename, _ = tts_service.synthesize_to_file(sentence)
        audio_url = f"/tts/{filename}"

        return jsonify({
            "success": True,
            "sentence": sentence,
            "audio_url": audio_url,
            "model_used": model_used
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/generate_audio_rundown", methods=["POST"])
def generate_audio_rundown():
    """Generate a spoken quick rundown: counts, times, upcoming meds."""
    try:
        events = read_events_from_file()
        meds = read_meds_file()

        # Craft rundown text
        text = summary_service.craft_rundown(events, meds)

        # Synthesize audio
        filename, _ = tts_service.synthesize_to_file(text)
        audio_url = f"/tts/{filename}"

        return jsonify({
            "success": True,
            "sentence": text,
            "audio_url": audio_url,
            "model_used": "summary_service+gtts"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/events/clear", methods=["POST"])
def clear_events():
    try:
        with events_lock:
            with open(EVENTS_FILE, 'w') as f:
                json.dump({"events": []}, f)
        return jsonify({"success": True, "message": "Events cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --- OPENROUTER BRIEFING ENDPOINT ---
@app.route("/api/generate_briefing", methods=["POST"])
def generate_briefing_proxy():
    """Generate AI briefing using OpenRouter's Gemini 2.0 Flash model."""
    if not openrouter_service:
        return jsonify({
            "success": False, 
            "error": "OpenRouter service not configured. Please check your OPENROUTER_API_KEY."
        }), 500
    
    try:
        # Get all events from the system
        events = read_events_from_file()
        
        if not events:
            return jsonify({
                "success": False,
                "error": "No events found to generate briefing from."
            }), 400
        
        # Generate briefing using OpenRouter service
        result = openrouter_service.generate_briefing(events)
        
        if result["success"]:
            return jsonify({
                "success": True, 
                "briefing": result["briefing"],
                "model_used": result.get("model_used", "unknown")
            })
        else:
            return jsonify({
                "success": False,
                "error": f"OpenRouter API error: {result.get('error', 'Unknown error')}"
            }), 500
            
    except Exception as e:
        print(f"❌ OpenRouter briefing failed: {e}")
        return jsonify({
            "success": False, 
            "error": f"Briefing generation failed: {str(e)}"
        }), 500


# --- NEW: Medication Management API ---
def read_meds_file():
    with meds_lock:
        if not os.path.exists(MEDICATIONS_FILE):
            with open(MEDICATIONS_FILE, 'w') as f:
                json.dump([], f)
            return []
        try:
            with open(MEDICATIONS_FILE, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return []

def write_meds_file(data):
    with meds_lock:
        with open(MEDICATIONS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

@app.route('/api/medications', methods=['GET'])
def get_medications():
    return jsonify(read_meds_file())

@app.route('/api/medications', methods=['POST'])
def add_medication():
    med_data = request.json
    med_data['id'] = str(uuid.uuid4())
    meds = read_meds_file()
    meds.append(med_data)
    write_meds_file(meds)
    return jsonify(med_data), 201

@app.route('/api/medications/<string:med_id>', methods=['DELETE'])
def delete_medication(med_id):
    meds = read_meds_file()
    meds_to_keep = [m for m in meds if m['id'] != med_id]
    if len(meds_to_keep) < len(meds):
        write_meds_file(meds_to_keep)
        return jsonify({'success': True}), 200
    return jsonify({'success': False, 'message': 'Medication not found'}), 404

@app.route('/api/medications/reminders', methods=['GET'])
def get_reminders():
    meds = read_meds_file()
    now = datetime.now()
    upcoming = []
    for med in meds:
        try:
            times = [t.strip() for t in med.get('times', '').split(',')]
            for t_str in times:
                reminder_time = datetime.strptime(t_str, '%I:%M %p').time()
                reminder_dt = now.replace(hour=reminder_time.hour, minute=reminder_time.minute, second=0, microsecond=0)
                if reminder_dt > now:
                    upcoming.append({'reminder_time': reminder_dt.isoformat(), 'medication': med})
        except ValueError:
            continue
    
    upcoming.sort(key=lambda x: x['reminder_time'])
    return jsonify({'reminders': upcoming})

@app.route('/api/medications/test-announcement', methods=['POST'])
def test_medication_announcement():
    """Test TTS announcement for medication reminders"""
    if not medication_service:
        return jsonify({'success': False, 'error': 'Medication service not available'}), 500
    
    try:
        data = request.get_json() or {}
        medication_name = data.get('name', 'Test Medication')
        dosage = data.get('dosage', '1 tablet')
        
        success = medication_service.test_announcement(medication_name, dosage)
        
        if success:
            return jsonify({'success': True, 'message': 'Test announcement played successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to play announcement'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# --- Main Entrypoint ---
if __name__ == "__main__":
    try:
        worker = threading.Thread(target=description_worker, daemon=True)
        worker.start()
        print("🚀 Starting Aegis AI controller at http://0.0.0.0:5000")
        app.run(debug=False, host="0.0.0.0", port=5000) # Debug=True for development
    finally:
        print("\n🛑 Shutting down application...")
        stop_camera_server()
        if medication_service:
            medication_service.stop()