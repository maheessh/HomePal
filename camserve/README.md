# CamServe - Simple Camera Server

CamServe is a Python-based camera server that provides MJPEG streaming and camera management capabilities. It's designed to work seamlessly with web applications and provides both a web interface and API endpoints for camera control.

## Features

- **MJPEG Streaming**: Real-time camera feed streaming via HTTP
- **Multiple Camera Backend Support**: Automatic fallback between DirectShow, Media Foundation, and other OpenCV backends
- **Web Interface**: Built-in HTML page for viewing camera streams
- **REST API**: JSON endpoints for camera status and control
- **Thread-Safe**: Background frame capture with thread-safe frame access
- **Multiple Frame Sizes**: Provides both full resolution and 320x320 frames
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Installation

### Prerequisites

- Python 3.7+
- OpenCV (cv2)
- Flask

### Setup

1. Install required dependencies:
```bash
pip install opencv-python flask
```

2. Or if you're using the project's virtual environment:
```bash
# Activate the virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from camserve import SimpleCameraServer

# Create camera server instance
camera = SimpleCameraServer(camera_id=0)

# Start camera
if camera.start():
    print("Camera started successfully!")
    
    # Run the web server
    camera.run_server(host='0.0.0.0', port=5001)
else:
    print("Failed to start camera")
```

### Running the Standalone Server

```bash
# Run directly
python camserve/camserve.py

# Or with custom parameters
python camserve/camserve.py --host 0.0.0.0 --port 5001
```

The server will start and be available at:
- **Web Interface**: http://localhost:5001/
- **MJPEG Stream**: http://localhost:5001/stream
- **Status API**: http://localhost:5001/api/status

## API Reference

### Endpoints

#### `GET /`
Serves a simple HTML page with the camera stream embedded.

#### `GET /stream`
Provides MJPEG stream of the camera feed.
- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`
- **Usage**: Can be embedded in `<img>` tags or consumed by video players

#### `GET /api/status`
Returns camera status information.

**Response:**
```json
{
    "camera_id": 0,
    "width": 640,
    "height": 480,
    "fps": 30.0,
    "is_running": true,
    "frame_available": true,
    "smframe_available": true
}
```

#### `GET /api/frame`
Returns a single frame as JPEG image.

**Response:**
- **Content-Type**: `image/jpeg`
- **Body**: JPEG image data

## Class Reference

### SimpleCameraServer

#### Constructor
```python
SimpleCameraServer(camera_id: int = 0)
```

**Parameters:**
- `camera_id` (int): Camera device ID (default: 0)

#### Methods

##### `start() -> bool`
Starts camera capture with automatic backend detection.

**Returns:**
- `True` if camera started successfully
- `False` if all backends failed

##### `stop()`
Stops camera capture and releases resources.

##### `get_frame() -> Optional[np.ndarray]`
Gets the latest full-resolution frame.

**Returns:**
- Latest frame as numpy array or `None`

##### `get_smframe() -> Optional[np.ndarray]`
Gets the latest 320x320 frame.

**Returns:**
- Latest 320x320 frame as numpy array or `None`

##### `get_camera_info() -> dict`
Gets camera information and status.

**Returns:**
- Dictionary with camera properties and status

##### `run_server(host='0.0.0.0', port=5001, debug=False)`
Runs the Flask web server.

**Parameters:**
- `host` (str): Server host address
- `port` (int): Server port number
- `debug` (bool): Enable Flask debug mode

## Integration Examples

### With Flask Applications

```python
from flask import Flask
from camserve import SimpleCameraServer

app = Flask(__name__)
camera = SimpleCameraServer()

@app.route('/my-camera')
def my_camera_page():
    return '<img src="/camera-stream" alt="Camera">'

@app.route('/camera-stream')
def camera_stream():
    # Proxy to camserve stream
    return camera.generate_mjpeg_response()

if __name__ == '__main__':
    if camera.start():
        app.run()
```

### With HomePal Application

The camserve module is integrated with the main HomePal application (`app.py`):

```python
# The main app automatically starts camserve as a subprocess
# and provides proxy endpoints for the camera stream
```

Access the camera through:
- **Main Dashboard**: http://localhost:5000/
- **Camera Stream**: http://localhost:5000/stream
- **Camera Control**: POST to `/api/camera/start` and `/api/camera/stop`

## Configuration

### Camera Settings

The camera automatically configures itself with these default settings:
- **Resolution**: 640x480
- **Frame Rate**: 30 FPS
- **JPEG Quality**: 85%

### Backend Priority (Windows)

1. **DirectShow** (`cv2.CAP_DSHOW`) - Recommended for Windows
2. **Media Foundation** (`cv2.CAP_MSMF`)
3. **Auto-detect** (`cv2.CAP_ANY`)

## Troubleshooting

### Common Issues

#### Camera Not Starting
```bash
❌ All camera backends failed
```

**Solutions:**
1. Check if camera is being used by another application
2. Try different camera IDs (0, 1, 2, etc.)
3. Update camera drivers
4. Check camera permissions

#### Stream Not Working
```bash
⚠️ Stream connection lost
```

**Solutions:**
1. Verify camera is running: `GET /api/status`
2. Check firewall settings
3. Ensure port 5001 is available

#### Poor Performance
**Solutions:**
1. Reduce frame resolution in `start()` method
2. Lower JPEG quality in `mjpeg_stream()`
3. Check system resources

### Debug Mode

Enable debug mode for detailed logging:

```python
camera.run_server(debug=True)
```

## File Structure

```
camserve/
├── camserve.py          # Main camera server implementation
├── _init_.py           # Module initialization (note: typo in filename)
└── README.md           # This documentation
```

## Dependencies

- **opencv-python**: Camera capture and image processing
- **flask**: Web server and API endpoints
- **numpy**: Image data handling
- **requests**: HTTP client (for integration)

## License

This module is part of the HomePal project. See the main project README for license information.

## Contributing

When contributing to camserve:

1. Follow the existing code style
2. Add proper error handling
3. Update documentation for new features
4. Test on multiple platforms
5. Ensure thread safety for all camera operations

## Support

For issues related to camserve:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Test with different camera devices
4. Verify all dependencies are installed correctly
