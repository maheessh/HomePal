# HomePal Packages

This directory contains the core detection and monitoring scripts for the HomePal system.

## Structure

### Main Scripts

1. **`home_surveillance.py`** - Home surveillance system
   - Fire detection using Inferno NCNN model
   - Motion detection using background subtraction
   - Event logging to `events.json`
   - Streams to surveillance frontend

2. **`pose_monitor.py`** - Personal monitoring system
   - Pose estimation using YOLOv8 Pose NCNN model
   - Activity recognition (falling, standing, sitting, walking)
   - Event logging to `events.json`
   - Streams to monitor frontend

### NCNN Models

1. **`inferno_ncnn_model/`** - Fire detection model
   - `model_ncnn.py` - Model wrapper and inference
   - `model.ncnn.param` - Model parameters
   - `model.ncnn.bin` - Model weights
   - `metadata.yaml` - Model metadata

2. **`yolo8spose_ncnn_model/`** - Pose estimation model
   - `model_ncnn.py` - Model wrapper and inference
   - `model.ncnn.param` - Model parameters
   - `model.ncnn.bin` - Model weights
   - `metadata.yaml` - Model metadata

### Legacy Files (Kept for Reference)

- `yolov8n-pose.pt` - Original PyTorch pose model
- `yolov8n.pt` - Original PyTorch object detection model

## Usage

### Running Individual Scripts

```bash
# Home surveillance
cd packages
python home_surveillance.py --camera 0 --model inferno_ncnn_model

# Pose monitoring
python pose_monitor.py --camera 0 --model yolo8spose_ncnn_model
```

### Integration with Camera Server

The scripts are integrated with `camserve/camserv.py` which:
- Manages camera capture and streaming
- Handles module toggling from the frontend
- Routes detection tasks to appropriate services
- Manages event logging and video recording

## Features

### Home Surveillance (`home_surveillance.py`)
- ✅ Fire detection with confidence scoring
- ✅ Motion detection with area tracking
- ✅ Event logging with image capture
- ✅ Real-time streaming capability
- ✅ Configurable detection thresholds

### Pose Monitor (`pose_monitor.py`)
- ✅ Human pose estimation
- ✅ Activity classification (falling, standing, sitting, walking)
- ✅ Multi-person detection
- ✅ Event logging with pose data
- ✅ Real-time streaming capability
- ✅ Alert system for falling detection

## Dependencies

- OpenCV (`cv2`)
- NCNN (`ncnn`)
- NumPy (`numpy`)
- JSON handling
- Threading for concurrent processing

## Configuration

Both scripts support command-line arguments:
- `--camera`: Camera source (default: 0)
- `--model`: Model path (default: respective model directories)

Event logging is automatically handled and saves to:
- `events.json` - Event log with timestamps and metadata
- `captured_images/` - Screenshots of detected events

## Performance

- Optimized for real-time processing
- Frame skipping for AI inference (configurable)
- Efficient NCNN model loading
- Background processing for non-critical tasks
