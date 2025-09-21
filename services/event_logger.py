#!/usr/bin/env python3
"""
Event Logger Service for HomePal
Centralized event logging for fire, smoke, motion, and other detection events
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

class EventLogger:
    """Centralized event logging service"""
    
    def __init__(self, events_file: str = "events.json", max_events: int = 1000):
        self.events_file = events_file
        self.max_events = max_events
        self.events_lock = threading.Lock()
        self._ensure_events_file()
    
    def _ensure_events_file(self):
        """Ensure events file exists with proper structure"""
        if not os.path.exists(self.events_file):
            with open(self.events_file, 'w') as f:
                json.dump([], f, indent=4)
    
    def log_event(self, 
                  event_type: str, 
                  module: str, 
                  confidence: Optional[float] = None,
                  description: str = "",
                  image_path: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an event to the events file
        
        Args:
            event_type: Type of event (e.g., "Fire Detected", "Motion Detected", "Smoke Detected")
            module: Module that detected the event ("surveillance" or "monitor")
            confidence: Confidence score (0.0 to 1.0)
            description: Human-readable description
            image_path: Path to captured image
            metadata: Additional metadata dictionary
            
        Returns:
            Event ID
        """
        try:
            timestamp = datetime.now().isoformat()
            event_id = f"{module}_{int(datetime.now().timestamp())}"
            
            event = {
                "id": event_id,
                "timestamp": timestamp,
                "module": module,
                "class_name": event_type,
                "confidence": confidence,
                "description": description,
                "image_path": image_path,
                "metadata": metadata or {}
            }
            
            with self.events_lock:
                # Read existing events
                events = self._read_events()
                
                # Add new event at the beginning
                events.insert(0, event)
                
                # Keep only last max_events
                if len(events) > self.max_events:
                    events = events[:self.max_events]
                
                # Write back to file
                self._write_events(events)
                
                print(f"📝 Logged {event_type} event: {event_id} (Module: {module})")
                return event_id
                
        except Exception as e:
            print(f"❌ Failed to log event: {e}")
            return ""
    
    def log_fire_event(self, confidence: float, image_path: Optional[str] = None) -> str:
        """Log a fire detection event"""
        return self.log_event(
            event_type="Fire Detected",
            module="surveillance",
            confidence=confidence,
            description=f"Fire detected with {confidence:.2%} confidence",
            image_path=image_path,
            metadata={"alert_level": "critical", "category": "fire"}
        )
    
    def log_smoke_event(self, confidence: float, image_path: Optional[str] = None) -> str:
        """Log a smoke detection event"""
        return self.log_event(
            event_type="Smoke Detected",
            module="surveillance",
            confidence=confidence,
            description=f"Smoke detected with {confidence:.2%} confidence",
            image_path=image_path,
            metadata={"alert_level": "high", "category": "smoke"}
        )
    
    def log_motion_event(self, motion_areas: List, confidence: Optional[float] = None, image_path: Optional[str] = None) -> str:
        """Log a motion detection event"""
        motion_count = len(motion_areas) if motion_areas else 0
        return self.log_event(
            event_type="Motion Detected",
            module="surveillance",
            confidence=confidence,
            description=f"Motion detected in {motion_count} area(s)",
            image_path=image_path,
            metadata={"alert_level": "medium", "category": "motion", "motion_areas": motion_areas}
        )
    
    def log_person_event(self, confidence: float, image_path: Optional[str] = None) -> str:
        """Log a person detection event"""
        return self.log_event(
            event_type="Person Detected",
            module="surveillance",
            confidence=confidence,
            description=f"Person detected with {confidence:.2%} confidence",
            image_path=image_path,
            metadata={"alert_level": "medium", "category": "person"}
        )
    
    def log_activity_event(self, activity: str, confidence: Optional[float] = None, image_path: Optional[str] = None) -> str:
        """Log an activity detection event"""
        alert_level = "critical" if activity == "Falling" else "low"
        return self.log_event(
            event_type=activity,
            module="monitor",
            confidence=confidence,
            description=f"Activity detected: {activity}",
            image_path=image_path,
            metadata={"alert_level": alert_level, "category": "activity"}
        )
    
    def get_events(self, module: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get events, optionally filtered by module"""
        try:
            events = self._read_events()
            
            if module:
                events = [e for e in events if e.get("module") == module]
            
            if limit:
                events = events[:limit]
            
            return events
        except Exception as e:
            print(f"❌ Failed to get events: {e}")
            return []
    
    def get_events_summary(self, module: Optional[str] = None, hours: int = 24) -> Dict[str, int]:
        """Get summary of events in the last N hours"""
        try:
            events = self.get_events(module)
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            summary = {
                "total_events": 0,
                "fire_events": 0,
                "smoke_events": 0,
                "motion_events": 0,
                "person_events": 0,
                "activity_events": 0,
                "critical_alerts": 0
            }
            
            for event in events:
                try:
                    event_time = datetime.fromisoformat(event.get("timestamp", "")).timestamp()
                    if event_time < cutoff_time:
                        continue
                    
                    summary["total_events"] += 1
                    
                    class_name = event.get("class_name", "").lower()
                    alert_level = event.get("metadata", {}).get("alert_level", "low")
                    
                    if "fire" in class_name:
                        summary["fire_events"] += 1
                    elif "smoke" in class_name:
                        summary["smoke_events"] += 1
                    elif "motion" in class_name:
                        summary["motion_events"] += 1
                    elif "person" in class_name:
                        summary["person_events"] += 1
                    elif event.get("module") == "monitor":
                        summary["activity_events"] += 1
                    
                    if alert_level == "critical":
                        summary["critical_alerts"] += 1
                        
                except (ValueError, TypeError):
                    continue
            
            return summary
        except Exception as e:
            print(f"❌ Failed to get events summary: {e}")
            return {}
    
    def clear_events(self, module: Optional[str] = None):
        """Clear events, optionally for a specific module"""
        try:
            with self.events_lock:
                if module:
                    events = self._read_events()
                    events = [e for e in events if e.get("module") != module]
                    self._write_events(events)
                else:
                    self._write_events([])
                print(f"🗑️ Cleared events{' for module ' + module if module else ''}")
        except Exception as e:
            print(f"❌ Failed to clear events: {e}")
    
    def _read_events(self) -> List[Dict]:
        """Read events from file"""
        try:
            with open(self.events_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    
    def _write_events(self, events: List[Dict]):
        """Write events to file"""
        with open(self.events_file, 'w') as f:
            json.dump(events, f, indent=4)

# Global event logger instance
event_logger = EventLogger()
