# HomePal - AI-Powered Security & Care Dashboard (Code scaffold)

This document contains a top-to-bottom, code-first scaffold for **HomePal**: README, key Python files, a minimal frontend template, requirements, and a simple integration test. Use this as a starting point for development.

---

## README.md

````markdown
# HomePal

AI-Powered Security & Care Dashboard

HomePal combines camera streaming, AI monitoring, and personal wellness tracking to provide home security and care features such as motion detection, fall detection, and emergency SMS alerts.

## Features
- Dashboard: Unified overview of security and care systems
- Home Surveillance: Live MJPEG camera feed + motion/activity logging
- Monitor Me: Personal wellness and fall detection with emergency SMS alerts
- Real-time Streaming: Camera server (MJPEG) with REST control
- Responsive UI: Tailwind CSS frontend

## Local setup
### Prerequisites
- Python 3.8+
- A webcam (for streaming)
- (Optional) ngrok or reverse proxy if you want remote access

### Installation
```bash
git clone <repository-url>
cd HomePal
python -m venv venv
# activate venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
````

### Running

Terminal 1: start camera server

```bash
python camserve/camserve.py
```

Terminal 2: start web dashboard

```bash
python app.py
```

Open: [http://localhost:5000](http://localhost:5000) (dashboard)
Camera MJPEG stream: [http://localhost:5001/stream](http://localhost:5001/stream)

## API

* `GET /` → Main dashboard
* `GET /stream` → Camera MJPEG stream
* `POST /api/camera/start` → Start camera server
* `POST /api/camera/stop` → Stop camera server
* `GET /api/camera/status` → Camera status
* `GET /api/events/recent` → Recent events

## Testing

```bash
python test_integration.py
```

## License

© 2025 HomePal. All Rights Reserved.

````

---

## requirements.txt

```text
Flask>=2.0
opencv-python-headless>=4.5
requests>=2.25
python-dotenv>=0.19
paho-mqtt>=1.6
pytest>=7.0
twilio>=8.0    # optional, for SMS alerts (replace with your provider)
numpy
````

---

## app.py (Main Flask application)

```python
from flask import Flask, render_template, jsonify, request
import requests
import subprocess
import threading
import time
import os

CAM_SERVER_URL = os.getenv('CAM_SERVER_URL', 'http://localhost:5001')

app = Flask(__name__, template_folder='templates')

camera_process = None
camera_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def proxy_stream():
    # Simple redirect/proxy to camera server stream
    return requests.get(f"{CAM_SERVER_URL}/stream", stream=True).raw.read()

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera_process
    with camera_lock:
        if camera_process and camera_process.poll() is None:
            return jsonify({'status': 'already_running'})
        # start camserve as a subprocess
        camera_process = subprocess.Popen(['python', 'camserve/camserve.py'])
        time.sleep(1)
        return jsonify({'status': 'started'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera_process
    with camera_lock:
        if not camera_process or camera_process.poll() is not None:
            return jsonify({'status': 'not_running'})
        camera_process.terminate()
        camera_process.wait(timeout=5)
        return jsonify({'status': 'stopped'})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    global camera_process
    with camera_lock:
        running = camera_process and camera_process.poll() is None
    return jsonify({'running': bool(running)})

# Events placeholder
EVENTS = []

@app.route('/api/events/recent', methods=['GET'])
def recent_events():
    # return last 20 events
    return jsonify({'events': EVENTS[-20:]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

---

## camserve/camserve.py (Minimal MJPEG camera server + motion detection skeleton)

```python
import cv2
from flask import Flask, Response, jsonify
import threading
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError('Cannot open camera')

# Simple motion detection state
last_frame = None
motion_threshold = 50000

def gen_frames():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        # encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(port=5001, debug=False)
```

Notes:

* This is a minimal example. For production, handle camera release, exceptions, and resource cleanup.
* To add motion detection, compare grayscale frame diffs and push events into the API `EVENTS` list.

---

## templates/index.html (Minimal dashboard with Tailwind CDN)

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>HomePal Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body class="bg-slate-100 min-h-screen">
    <div class="max-w-6xl mx-auto p-6">
      <h1 class="text-3xl font-bold mb-4">HomePal Dashboard</h1>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="col-span-2 bg-white p-4 rounded shadow">
          <h2 class="font-semibold mb-2">Live Camera</h2>
          <div class="aspect-video">
            <img id="cameraStream" src="/stream" alt="camera stream" class="w-full h-full object-cover rounded" />
          </div>
        </div>

        <div class="bg-white p-4 rounded shadow">
          <h2 class="font-semibold mb-2">Controls</h2>
          <div class="space-y-2">
            <button id="startBtn" class="px-3 py-2 bg-green-500 text-white rounded">Start Camera</button>
            <button id="stopBtn" class="px-3 py-2 bg-red-500 text-white rounded">Stop Camera</button>
            <div id="status" class="mt-2 text-sm text-gray-600">Status: unknown</div>
          </div>
        </div>
      </div>

      <div class="mt-6 bg-white p-4 rounded shadow">
        <h2 class="font-semibold mb-2">Recent Events</h2>
        <ul id="events" class="text-sm text-gray-700"></ul>
      </div>
    </div>

    <script>
      async function getStatus() {
        const r = await fetch('/api/camera/status');
        const j = await r.json();
        document.getElementById('status').innerText = 'Status: ' + (j.running ? 'running' : 'stopped');
      }
      document.getElementById('startBtn').onclick = async () => { await fetch('/api/camera/start', {method:'POST'}); await getStatus(); };
      document.getElementById('stopBtn').onclick = async () => { await fetch('/api/camera/stop', {method:'POST'}); await getStatus(); };

      async function loadEvents(){
        const r = await fetch('/api/events/recent');
        const j = await r.json();
        const ul = document.getElementById('events');
        ul.innerHTML = '';
        j.events.forEach(e => { const li = document.createElement('li'); li.textContent = `${e.time || ''} — ${e.type || 'event'}`; ul.appendChild(li); });
      }

      getStatus();
      loadEvents();
      setInterval(getStatus, 5000);
      setInterval(loadEvents, 5000);
    </script>
  </body>
</html>
```

---

## test\_integration.py (very simple smoke test)

```python
import requests

BASE = 'http://localhost:5000'

def test_index():
    r = requests.get(BASE)
    assert r.status_code == 200

def test_camera_status():
    r = requests.get(BASE + '/api/camera/status')
    assert r.status_code == 200

if __name__ == '__main__':
    print('Running simple integration checks...')
    test_index()
    print('index ok')
    test_camera_status()
    print('camera status ok')
```

---

## .env.example

```text
CAM_SERVER_URL=http://localhost:5001
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_NUMBER=+15551234567
EMERGENCY_CONTACTS=+15559876543,+15557654321
```

---

## Next steps / Tips

* Replace the simple subprocess management with a supervised approach (systemd, docker-compose, or a process manager)
* Implement proper motion/fall detection logic, and persist events to a database (SQLite/Postgres)
* Add authentication to the dashboard and secure HTTPS
* Integrate SMS/notification provider (Twilio, AWS SNS, etc.) for emergency alerts
* Add unit tests, CI pipeline, and Dockerfile for reproducible deployment

---

© 2025 HomePal. All Rights Reserved.

```
```
