HomePal - AI-Powered Security & Care Dashboard

HomePal is more than a home security system—it’s a caring companion. Built with integrated camera streaming, AI monitoring, and personal wellness tracking, HomePal ensures the safety of your home and loved ones.

With motion detection, fall detection, and emergency SMS alerts, it provides real-time protection for children, elderly family members, and anyone who needs extra care. HomePal combines home safety with personal well-being, creating a smarter, safer living space.

Our vision goes beyond emergencies—HomePal will evolve to track activity patterns, strengthen security, and use AI to bring peace of mind to every household.

Features

Dashboard: Unified overview of all security and care systems

Home Surveillance: Live camera feed with motion/activity monitoring

Monitor Me: Personal wellness and fall detection with auto emergency alerts

Real-time Streaming: MJPEG camera streams with live status indicators

Responsive UI: Clean, modern interface built with Tailwind CSS

Setup
Prerequisites

Python 3.8+

Web camera (for streaming functionality)

Installation

Clone the repository:

git clone <repository-url>
cd HomePal


Create and activate a virtual environment:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

Running the Application

Start the main application:

python app.py


The application will:

Launch the main dashboard at http://localhost:5000

Start the camera server at http://localhost:5001

Enable live streaming and provide API endpoints

Open your browser and navigate to:

http://localhost:5000

Usage
Dashboard

View system health and security status

Navigate between modules seamlessly

Home Surveillance

Toggle surveillance on/off

Watch live camera feed in real time

Review motion activity logs

Monitor Me

Toggle personal monitoring on/off

Track wellness and detect falls or accidents

Auto-send emergency SMS notifications when alerts trigger

API Endpoints

GET / → Main dashboard

GET /stream → Camera MJPEG stream

POST /api/camera/start → Start camera server

POST /api/camera/stop → Stop camera server

GET /api/camera/status → Get camera status

GET /api/events/recent → Retrieve recent activity events

Architecture
app.py                 # Main Flask application with API endpoints
camserve/
  camserve.py          # Camera server with MJPEG streaming
templates/
  index.html           # Frontend dashboard with Tailwind CSS

Testing

Run the integration test script to verify functionality:

python test_integration.py

Troubleshooting
Camera Issues

Ensure your webcam is not used by another application

Verify the camera server is running on port 5001

Check system permissions for camera access

Port Conflicts

Dashboard runs on port 5000

Camera server runs on port 5001

Make sure both ports are free before starting the app

Development

HomePal follows a clean architecture pattern:

Frontend: HTML/CSS/JavaScript with Tailwind CSS

Backend: Flask modular API design

Camera Service: OpenCV-powered MJPEG streaming

API Layer: RESTful endpoints for monitoring and control

License

© 2025 HomePal. All Rights Reserved.
