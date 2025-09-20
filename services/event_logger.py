#!/usr/bin/env python3
"""
Event Logger Service
Logs fire, smoke, and motion detection events to JSON files
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import threading


class EventLogger:
    """
    Event logger for fire, smoke, and motion detection events.
    Thread-safe logging to event_log.json file.
    """
    
    def __init__(self, log_file: str = None):
        """
        Initialize the event logger.
        
        Args:
            log_file: Path to the JSON log file (if None, will use base folder)
        """
        if log_file is None:
            # Determine the base folder path (project root)
            current_file = os.path.abspath(__file__)
            services_dir = os.path.dirname(current_file)
            base_dir = os.path.dirname(services_dir)  # Go up one level from services
            self.log_file = os.path.join(base_dir, "event_log.json")
        else:
            self.log_file = log_file
            
        self._lock = threading.Lock()
        
        # Initialize log file if it doesn't exist
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize empty JSON structure in log file if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({"events": []}, f, indent=2)
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Thread-safe method to log an event to event_log.json.
        
        Args:
            event_type: Type of event (fire, smoke, motion)
            event_data: Event data dictionary
        """
        with self._lock:
            try:
                # Read existing events
                log_data = {"events": []}
                if os.path.exists(self.log_file):
                    try:
                        with open(self.log_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                log_data = json.loads(content)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Warning: Corrupted JSON file detected, creating backup and starting fresh: {e}")
                        # Create backup of corrupted file
                        backup_file = f"{self.log_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        if os.path.exists(self.log_file):
                            os.rename(self.log_file, backup_file)
                        log_data = {"events": []}
                
                # Ensure events key exists
                if "events" not in log_data:
                    log_data["events"] = []
                
                # Add new event
                log_data["events"].append(event_data)
                
                # Keep only last 1000 events to prevent file from growing too large
                if len(log_data["events"]) > 1000:
                    log_data["events"] = log_data["events"][-1000:]
                
                # Write to temporary file first, then rename (atomic operation)
                temp_file = f"{self.log_file}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                if os.name == 'nt':  # Windows
                    if os.path.exists(self.log_file):
                        os.remove(self.log_file)
                    os.rename(temp_file, self.log_file)
                else:  # Unix-like systems
                    os.rename(temp_file, self.log_file)
                
            except Exception as e:
                print(f"Error logging {event_type} event: {e}")
                # Clean up temp file if it exists
                temp_file = f"{self.log_file}.tmp"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def log_fire_detection(self, confidence: float, file_location: Optional[str] = None):
        """
        Log a fire detection event.
        
        Args:
            confidence: Detection confidence score
            file_location: Path to saved image file (optional)
        """
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "fire_detection",
            "confidence": confidence,
            "file_location": file_location or ""
        }
        
        self._log_event("fire", event_data)
        print(f"[FIRE DETECTED] Confidence: {confidence:.2f}, File: {file_location or 'N/A'}, Time: {event_data['timestamp']}")
    
    def log_smoke_detection(self, confidence: float, file_location: Optional[str] = None):
        """
        Log a smoke detection event.
        
        Args:
            confidence: Detection confidence score
            file_location: Path to saved image file (optional)
        """
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "smoke_detection",
            "confidence": confidence,
            "file_location": file_location or ""
        }
        
        self._log_event("smoke", event_data)
        print(f"[SMOKE DETECTED] Confidence: {confidence:.2f}, File: {file_location or 'N/A'}, Time: {event_data['timestamp']}")
    
    def log_motion_detection(self, file_location: Optional[str] = None):
        """
        Log a motion detection event.
        
        Args:
            file_location: Path to saved video file (optional)
        """
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "motion_detection",
            "file_location": file_location or ""
        }
        
        self._log_event("motion", event_data)
        print(f"[MOTION DETECTED] File: {file_location or 'N/A'}, Time: {event_data['timestamp']}")
    
    def log_fall_detection(self, confidence: float, file_location: Optional[str] = None, frame_count: int = 0):
        """
        Log a fall detection event.
        
        Args:
            confidence: Detection confidence score (width/height ratio)
            file_location: Path to saved annotated image file (optional)
            frame_count: Frame number where fall was detected
        """
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "fall_detection",
            "confidence": confidence,
            "file_location": file_location or "",
            "frame_count": frame_count
        }
        
        self._log_event("fall", event_data)
        print(f"[FALL DETECTED] Confidence: {confidence:.2f}, Frame: {frame_count}, File: {file_location or 'N/A'}, Time: {event_data['timestamp']}")
    
    def get_recent_events(self, event_type: str = "all", limit: int = 10) -> list:
        """
        Get recent events from event_log.json.
        
        Args:
            event_type: Type of events to retrieve (fire, smoke, motion, all)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        with self._lock:
            try:
                if not os.path.exists(self.log_file):
                    return []
                
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                events = log_data.get("events", [])
                
                # Filter by event type if not "all"
                if event_type != "all":
                    events = [event for event in events if event.get("event_type") == f"{event_type}_detection"]
                
                # Return most recent events
                return events[-limit:] if limit > 0 else events
                
            except Exception as e:
                print(f"Error reading {event_type} events: {e}")
                return []
    
    def clear_old_events(self, days_to_keep: int = 7):
        """
        Clear events older than specified days from event_log.json.
        
        Args:
            days_to_keep: Number of days of events to keep
        """
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        if not os.path.exists(self.log_file):
            return
                
        with self._lock:
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                events = log_data.get("events", [])
                
                # Filter events newer than cutoff date
                filtered_events = [
                    event for event in events
                    if datetime.fromisoformat(event['timestamp']).timestamp() > cutoff_date
                ]
                
                log_data["events"] = filtered_events
                
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error clearing old events from {self.log_file}: {e}")


# Global logger instance
logger = EventLogger()


def get_logger() -> EventLogger:
    """Get the global event logger instance."""
    return logger
