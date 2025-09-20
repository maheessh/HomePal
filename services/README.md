# Services

This directory contains service modules for the HomePal application.

## Event Logger

The `event_logger.py` module provides a comprehensive logging system for fire, smoke, and motion detection events.

### Features

- **Thread-safe logging** to `event_log.json`
- **Automatic log file rotation** (keeps last 1000 events)
- **Single JSON file** containing all event types:
  - Fire detection events
  - Smoke detection events  
  - Motion detection events
- **Rich event data** including timestamps, coordinates, confidence scores, and frame information
- **Event retrieval** functionality for recent events with filtering
- **Automatic cleanup** of old events (configurable retention period)

### Usage

```python
from event_logger import get_logger

# Get the global logger instance
logger = get_logger()

# Log a fire detection
logger.log_fire_detection(
    confidence=0.85,
    coordinates={"x1": 100, "y1": 150, "x2": 200, "y2": 250},
    frame_info={"frame_size": (480, 640), "model_size": 320}
)

# Log a smoke detection
logger.log_smoke_detection(
    confidence=0.72,
    coordinates={"x1": 50, "y1": 80, "x2": 180, "y2": 200}
)

# Log motion detection
logger.log_motion_detection(
    motion_area=1250,
    frame_info={"frame_size": (480, 640)}
)

# Retrieve recent events
recent_events = logger.get_recent_events("all", limit=10)

# Retrieve events by type
fire_events = logger.get_recent_events("fire", limit=5)
smoke_events = logger.get_recent_events("smoke", limit=5)
motion_events = logger.get_recent_events("motion", limit=5)
```

### Event Data Structure

Each logged event contains:

```json
{
  "timestamp": "2025-09-20T01:50:38.776501",
  "event_type": "fire_detection|smoke_detection|motion_detection",
  "confidence": 0.85,
  "motion_area": 1250,
  "coordinates": {
    "x1": 100, "y1": 150, "x2": 200, "y2": 250
  },
  "frame_info": {
    "frame_size": [480, 640],
    "model_size": 320,
    "confidence_threshold": 0.5
  }
}
```

### Log File Location

All events are logged to `event_log.json` in the project's base folder (project root). The file is created automatically if it doesn't exist, regardless of where the event logger is imported from.

### Integration

The event logger is integrated into `packages/homeSurvey/runner.py` and automatically logs:
- Motion detection events when motion is detected
- Fire detection events when fire is detected by the YOLO model
- Smoke detection events when smoke is detected by the YOLO model

The system replaces terminal print statements with structured JSON logging for better data analysis and monitoring.
