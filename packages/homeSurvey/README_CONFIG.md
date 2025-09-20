# Home Surveillance Configuration Guide

This guide explains how to configure the Home Surveillance system using the configuration file.

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp config_example.yaml config.yaml
   ```

2. **Edit the configuration file** to match your needs

3. **Run the surveillance system:**
   ```bash
   python runner.py
   ```

## Configuration File Structure

The configuration file (`config.yaml`) allows you to control various aspects of the surveillance system:

### Detection Settings

#### Motion Detection
- `enabled`: Enable/disable motion detection
- `area_threshold`: Minimum area to count as motion (higher = less sensitive)
- `cooldown`: Seconds to wait between detections

#### Fire Detection
- `enabled`: Enable/disable fire detection
- `confidence_threshold`: Minimum confidence (0.0-1.0, higher = fewer false positives)

#### Smoke Detection
- `enabled`: Enable/disable smoke detection
- `confidence_threshold`: Minimum confidence (0.0-1.0, higher = fewer false positives)

### Camera Settings

- `index`: Camera device index (0 = default camera)
- `image_size`: Input size for AI model (usually 320 or 640)

### Recording Settings

- `enabled`: Enable/disable video recording
- `duration`: How long to record each motion event (seconds)
- `fps`: Video frame rate
- `save_folder`: Where to save videos and images

### Display Settings

- `enabled`: Show/hide the camera window
- `show_motion_boxes`: Display motion detection rectangles
- `show_confidence`: Show confidence scores on detections

### Logging Settings

- `enabled`: Enable/disable event logging
- `level`: Log detail level (DEBUG, INFO, WARNING, ERROR)

## Common Configuration Examples

### 1. Motion Only (No Fire/Smoke Detection)
```yaml
detection:
  motion:
    enabled: true
    area_threshold: 50
    cooldown: 1.0
  fire:
    enabled: false
    confidence_threshold: 0.50
  smoke:
    enabled: false
    confidence_threshold: 0.50
```

### 2. Fire Detection Only
```yaml
detection:
  motion:
    enabled: true  # Still needed to trigger detection
    area_threshold: 50
    cooldown: 1.0
  fire:
    enabled: true
    confidence_threshold: 0.60
  smoke:
    enabled: false
    confidence_threshold: 0.50
```

### 3. Conservative Setup (Fewer False Positives)
```yaml
detection:
  motion:
    enabled: true
    area_threshold: 100  # Less sensitive to motion
    cooldown: 2.0        # Wait longer between detections
  fire:
    enabled: true
    confidence_threshold: 0.70  # Higher confidence required
  smoke:
    enabled: true
    confidence_threshold: 0.70  # Higher confidence required
```

### 4. Sensitive Setup (More Detections)
```yaml
detection:
  motion:
    enabled: true
    area_threshold: 25   # More sensitive to motion
    cooldown: 0.5        # Detect more frequently
  fire:
    enabled: true
    confidence_threshold: 0.30  # Lower confidence required
  smoke:
    enabled: true
    confidence_threshold: 0.30  # Lower confidence required
```

### 5. Headless Operation (No Display Window)
```yaml
display:
  enabled: false
  show_motion_boxes: false
  show_confidence: false

recording:
  enabled: true  # Still record videos
  duration: 10.0
  fps: 30
  save_folder: "../../saved"
```

### 6. No Recording (Detection Only)
```yaml
recording:
  enabled: false
  duration: 10.0
  fps: 30
  save_folder: "../../saved"

logging:
  enabled: true  # Still log events
  level: "INFO"
```

## Tips for Configuration

### Motion Sensitivity
- **Lower `area_threshold`** = More sensitive to small movements
- **Higher `area_threshold`** = Only detects larger movements
- **Lower `cooldown`** = Detects more frequently
- **Higher `cooldown`** = Reduces duplicate detections

### Detection Accuracy
- **Higher confidence thresholds** = Fewer false positives, might miss some real events
- **Lower confidence thresholds** = More detections, but more false positives

### Performance
- **Disable display** for better performance on slower systems
- **Disable recording** if you only want detection alerts
- **Increase cooldown** to reduce CPU usage

### Storage Management
- **Shorter recording duration** = Less disk space usage
- **Lower FPS** = Smaller video files
- **Disable recording** = No video files saved

## Troubleshooting

### Camera Not Working
- Try different `camera.index` values (0, 1, 2, etc.)
- Check if camera is being used by another application

### Too Many False Positives
- Increase confidence thresholds
- Increase motion area threshold
- Increase detection cooldown

### Missing Real Events
- Decrease confidence thresholds
- Decrease motion area threshold
- Decrease detection cooldown

### Performance Issues
- Disable display window
- Increase detection cooldown
- Use lower image size if supported

## File Locations

- **Configuration**: `config.yaml`
- **Example Configuration**: `config_example.yaml`
- **Videos/Images**: `../../saved/` (relative to homeSurvey folder)
- **Logs**: `../../event_log.json`

## Restart Required

After changing the configuration file, you need to restart the surveillance system for changes to take effect:

```bash
# Stop the current system (Ctrl+C)
# Then restart:
python runner.py
```
