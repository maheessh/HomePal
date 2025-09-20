#!/usr/bin/env python3
"""
Configuration loader for Home Surveillance System
Handles loading and validating configuration from YAML file
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Configuration for detection settings"""
    motion_enabled: bool
    motion_area_threshold: int
    motion_cooldown: float
    fire_enabled: bool
    fire_confidence_threshold: float
    smoke_enabled: bool
    smoke_confidence_threshold: float


@dataclass
class CameraConfig:
    """Configuration for camera settings"""
    index: int
    image_size: int


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    path: str
    max_detections_per_frame: int


@dataclass
class RecordingConfig:
    """Configuration for recording settings"""
    enabled: bool
    duration: float
    fps: int
    save_folder: str


@dataclass
class DisplayConfig:
    """Configuration for display settings"""
    enabled: bool
    show_motion_boxes: bool
    show_confidence: bool


@dataclass
class LoggingConfig:
    """Configuration for logging settings"""
    enabled: bool
    level: str


class ConfigLoader:
    """Handles loading and validation of configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                print(f"❌ Configuration file not found: {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
                
            if not self.config_data:
                print("❌ Configuration file is empty or invalid")
                return False
                
            print(f"✅ Configuration loaded from: {self.config_path}")
            return True
            
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML configuration: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate detection settings
            if 'detection' not in self.config_data:
                print("❌ Missing 'detection' section in configuration")
                return False
                
            detection = self.config_data['detection']
            
            # Validate motion settings
            if 'motion' not in detection:
                print("❌ Missing 'motion' section in detection configuration")
                return False
                
            motion = detection['motion']
            if not isinstance(motion.get('enabled'), bool):
                print("❌ Motion 'enabled' must be a boolean")
                return False
            if not isinstance(motion.get('area_threshold'), (int, float)):
                print("❌ Motion 'area_threshold' must be a number")
                return False
            if not isinstance(motion.get('cooldown'), (int, float)):
                print("❌ Motion 'cooldown' must be a number")
                return False
            
            # Validate fire settings
            if 'fire' not in detection:
                print("❌ Missing 'fire' section in detection configuration")
                return False
                
            fire = detection['fire']
            if not isinstance(fire.get('enabled'), bool):
                print("❌ Fire 'enabled' must be a boolean")
                return False
            if not isinstance(fire.get('confidence_threshold'), (int, float)):
                print("❌ Fire 'confidence_threshold' must be a number")
                return False
            if not 0 <= fire.get('confidence_threshold', 0) <= 1:
                print("❌ Fire 'confidence_threshold' must be between 0 and 1")
                return False
            
            # Validate smoke settings
            if 'smoke' not in detection:
                print("❌ Missing 'smoke' section in detection configuration")
                return False
                
            smoke = detection['smoke']
            if not isinstance(smoke.get('enabled'), bool):
                print("❌ Smoke 'enabled' must be a boolean")
                return False
            if not isinstance(smoke.get('confidence_threshold'), (int, float)):
                print("❌ Smoke 'confidence_threshold' must be a number")
                return False
            if not 0 <= smoke.get('confidence_threshold', 0) <= 1:
                print("❌ Smoke 'confidence_threshold' must be between 0 and 1")
                return False
            
            # Validate camera settings
            if 'camera' not in self.config_data:
                print("❌ Missing 'camera' section in configuration")
                return False
                
            camera = self.config_data['camera']
            if not isinstance(camera.get('index'), int):
                print("❌ Camera 'index' must be an integer")
                return False
            if not isinstance(camera.get('image_size'), int):
                print("❌ Camera 'image_size' must be an integer")
                return False
            
            # Validate model settings
            if 'model' not in self.config_data:
                print("❌ Missing 'model' section in configuration")
                return False
                
            model = self.config_data['model']
            if not isinstance(model.get('path'), str):
                print("❌ Model 'path' must be a string")
                return False
            if not isinstance(model.get('max_detections_per_frame'), int):
                print("❌ Model 'max_detections_per_frame' must be an integer")
                return False
            
            # Validate recording settings
            if 'recording' not in self.config_data:
                print("❌ Missing 'recording' section in configuration")
                return False
                
            recording = self.config_data['recording']
            if not isinstance(recording.get('enabled'), bool):
                print("❌ Recording 'enabled' must be a boolean")
                return False
            if not isinstance(recording.get('duration'), (int, float)):
                print("❌ Recording 'duration' must be a number")
                return False
            if not isinstance(recording.get('fps'), int):
                print("❌ Recording 'fps' must be an integer")
                return False
            if not isinstance(recording.get('save_folder'), str):
                print("❌ Recording 'save_folder' must be a string")
                return False
            
            # Validate display settings
            if 'display' not in self.config_data:
                print("❌ Missing 'display' section in configuration")
                return False
                
            display = self.config_data['display']
            if not isinstance(display.get('enabled'), bool):
                print("❌ Display 'enabled' must be a boolean")
                return False
            if not isinstance(display.get('show_motion_boxes'), bool):
                print("❌ Display 'show_motion_boxes' must be a boolean")
                return False
            if not isinstance(display.get('show_confidence'), bool):
                print("❌ Display 'show_confidence' must be a boolean")
                return False
            
            # Validate logging settings
            if 'logging' not in self.config_data:
                print("❌ Missing 'logging' section in configuration")
                return False
                
            logging = self.config_data['logging']
            if not isinstance(logging.get('enabled'), bool):
                print("❌ Logging 'enabled' must be a boolean")
                return False
            if not isinstance(logging.get('level'), str):
                print("❌ Logging 'level' must be a string")
                return False
            
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            if logging.get('level') not in valid_levels:
                print(f"❌ Logging 'level' must be one of: {valid_levels}")
                return False
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Error validating configuration: {e}")
            return False
    
    def get_detection_config(self) -> DetectionConfig:
        """Get detection configuration"""
        detection = self.config_data['detection']
        motion = detection['motion']
        fire = detection['fire']
        smoke = detection['smoke']
        
        return DetectionConfig(
            motion_enabled=motion['enabled'],
            motion_area_threshold=motion['area_threshold'],
            motion_cooldown=motion['cooldown'],
            fire_enabled=fire['enabled'],
            fire_confidence_threshold=fire['confidence_threshold'],
            smoke_enabled=smoke['enabled'],
            smoke_confidence_threshold=smoke['confidence_threshold']
        )
    
    def get_camera_config(self) -> CameraConfig:
        """Get camera configuration"""
        camera = self.config_data['camera']
        return CameraConfig(
            index=camera['index'],
            image_size=camera['image_size']
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        model = self.config_data['model']
        return ModelConfig(
            path=model['path'],
            max_detections_per_frame=model['max_detections_per_frame']
        )
    
    def get_recording_config(self) -> RecordingConfig:
        """Get recording configuration"""
        recording = self.config_data['recording']
        return RecordingConfig(
            enabled=recording['enabled'],
            duration=recording['duration'],
            fps=recording['fps'],
            save_folder=recording['save_folder']
        )
    
    def get_display_config(self) -> DisplayConfig:
        """Get display configuration"""
        display = self.config_data['display']
        return DisplayConfig(
            enabled=display['enabled'],
            show_motion_boxes=display['show_motion_boxes'],
            show_confidence=display['show_confidence']
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        logging = self.config_data['logging']
        return LoggingConfig(
            enabled=logging['enabled'],
            level=logging['level']
        )
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        if not self.config_data:
            print("❌ No configuration loaded")
            return
            
        print("\n📋 Configuration Summary:")
        print("=" * 50)
        
        detection = self.config_data['detection']
        print(f"🔍 Motion Detection: {'✅ Enabled' if detection['motion']['enabled'] else '❌ Disabled'}")
        print(f"🔥 Fire Detection: {'✅ Enabled' if detection['fire']['enabled'] else '❌ Disabled'}")
        print(f"💨 Smoke Detection: {'✅ Enabled' if detection['smoke']['enabled'] else '❌ Disabled'}")
        
        camera = self.config_data['camera']
        print(f"📹 Camera Index: {camera['index']}")
        print(f"📐 Image Size: {camera['image_size']}x{camera['image_size']}")
        
        recording = self.config_data['recording']
        print(f"🎥 Video Recording: {'✅ Enabled' if recording['enabled'] else '❌ Disabled'}")
        if recording['enabled']:
            print(f"⏱️ Recording Duration: {recording['duration']}s")
            print(f"🎬 FPS: {recording['fps']}")
        
        display = self.config_data['display']
        print(f"🖥️ Display Window: {'✅ Enabled' if display['enabled'] else '❌ Disabled'}")
        
        logging = self.config_data['logging']
        print(f"📝 Logging: {'✅ Enabled' if logging['enabled'] else '❌ Disabled'}")
        if logging['enabled']:
            print(f"📊 Log Level: {logging['level']}")
        
        print("=" * 50)


def load_surveillance_config(config_path: str = "config.yaml") -> Optional[ConfigLoader]:
    """
    Load and validate surveillance configuration
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigLoader instance if successful, None otherwise
    """
    config_loader = ConfigLoader(config_path)
    
    if not config_loader.load_config():
        return None
        
    if not config_loader.validate_config():
        return None
        
    config_loader.print_config_summary()
    return config_loader


if __name__ == "__main__":
    # Test the configuration loader
    config = load_surveillance_config()
    if config:
        print("\n✅ Configuration loaded successfully!")
    else:
        print("\n❌ Failed to load configuration")
