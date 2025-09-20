# Aegis AI - Security Dashboard

A modern security dashboard with integrated camera streaming for home surveillance and personal monitoring.

## Features

- **Dashboard**: Overview of all security systems
- **Home Surveillance**: Live camera feed with activity monitoring
- **Monitor Me**: Personal wellness tracking with camera integration
- **Real-time Streaming**: MJPEG camera streams with live status indicators
- **Responsive UI**: Clean, modern interface built with Tailwind CSS

## Setup

### Prerequisites

- Python 3.8+
- Web camera (for streaming functionality)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Hackrice15
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the main application:
```bash
python app.py
```

The application will:
- Start the main dashboard on `http://localhost:5000`
- Automatically start the camera server on `http://localhost:5001`
- Provide camera streaming and API endpoints

2. Open your browser and navigate to `http://localhost:5000`

## Usage

### Dashboard
- View system status overview
- Navigate to different modules

### Home Surveillance
- Toggle surveillance on/off using the switch
- View live camera feed when activated
- Monitor activity logs in real-time

### Monitor Me
- Toggle personal monitoring on/off
- View live camera feed for personal tracking
- Track wellness metrics and alerts

## API Endpoints

- `GET /` - Main dashboard
- `GET /stream` - Camera MJPEG stream
- `POST /api/camera/start` - Start camera server
- `POST /api/camera/stop` - Stop camera server
- `GET /api/camera/status` - Get camera status
- `GET /api/events/recent` - Get recent activity events

## Architecture

```
app.py                 # Main Flask application with API endpoints
camserve/
  camserve.py         # Camera server with MJPEG streaming
templates/
  index.html          # Frontend dashboard with Tailwind CSS
```

## Testing

Run the integration test script to verify functionality:
```bash
python test_integration.py
```

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Check that the camera server is running on port 5001
- Verify camera permissions on your system

### Port Conflicts
- The main app runs on port 5000
- The camera server runs on port 5001
- Ensure these ports are available

## Development

The application follows a clean architecture pattern:
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS
- **Backend**: Flask with modular structure
- **Camera Service**: Separate OpenCV-based streaming service
- **API Layer**: RESTful endpoints for system control

## License

© 2025 Aegis AI. All Rights Reserved.
