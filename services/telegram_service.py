#!/usr/bin/env python3
"""
Telegram Service for HomePal
Handles all Telegram notifications for Fire and Fall detection with cooldown management.
"""
import os
import requests
import time
import threading
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramService:
    """Service for sending Telegram notifications with cooldown management."""
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Cooldown tracking (30 seconds)
        self.cooldown_duration = 30.0
        self._last_fire_notification = 0.0
        self._last_fall_notification = 0.0
        
        # Verify credentials
        if not self.bot_token or not self.chat_id:
            print("⚠️ TELEGRAM_SERVICE: Credentials not found in .env file")
        else:
            print(f"✅ TELEGRAM_SERVICE: Initialized with bot token {self.bot_token[:10]}...")
    
    def _send_message(self, message: str) -> bool:
        """Send a message via Telegram API."""
        if not self.bot_token or not self.chat_id:
            print("❌ TELEGRAM_SERVICE: Cannot send message - credentials missing")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("ok"):
                print("✅ TELEGRAM_SERVICE: Message sent successfully")
                return True
            else:
                print(f"❌ TELEGRAM_SERVICE: API error: {result}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ TELEGRAM_SERVICE: Failed to send message: {e}")
            return False
    
    def _can_send_notification(self, notification_type: str) -> bool:
        """Check if enough time has passed since last notification of this type."""
        current_time = time.time()
        
        if notification_type == "fire":
            return (current_time - self._last_fire_notification) >= self.cooldown_duration
        elif notification_type == "fall":
            return (current_time - self._last_fall_notification) >= self.cooldown_duration
        else:
            return True  # Allow other types
    
    def _update_cooldown(self, notification_type: str):
        """Update the last notification time for the given type."""
        current_time = time.time()
        
        if notification_type == "fire":
            self._last_fire_notification = current_time
        elif notification_type == "fall":
            self._last_fall_notification = current_time
    
    def send_fire_alert(self, confidence: float, image_path: Optional[str] = None) -> bool:
        """Send a fire detection alert."""
        if not self._can_send_notification("fire"):
            print(f"⏳ TELEGRAM_SERVICE: Fire notification skipped (cooldown active)")
            return False
        
        timestamp = datetime.now().strftime('%I:%M:%S %p on %B %d, %Y')
        message = f"🔥 *FIRE ALERT* 🔥\n\n🚨 *FIRE DETECTED*\n\nA fire has been detected by your surveillance system.\n\n🕐 Time: {timestamp}\n📊 Confidence: {confidence:.2f}\n🏠 Location: Home Surveillance System"
        
        if image_path:
            message += f"\n📸 Image: {image_path}"
        
        success = self._send_message(message)
        if success:
            self._update_cooldown("fire")
            print(f"🔥 TELEGRAM_SERVICE: Fire alert sent (confidence: {confidence:.3f})")
        
        return success
    
    def send_fall_alert(self, image_path: Optional[str] = None) -> bool:
        """Send a fall detection alert."""
        if not self._can_send_notification("fall"):
            print(f"⏳ TELEGRAM_SERVICE: Fall notification skipped (cooldown active)")
            return False
        
        timestamp = datetime.now().strftime('%I:%M:%S %p on %B %d, %Y')
        message = f"🚨 *EMERGENCY ALERT* 🚨\n\n⚠️ *FALL DETECTED*\n\nA sudden fall or unusual activity was detected while your monitoring system was active.\n\n🕐 Time: {timestamp}\n🏠 Location: Home Surveillance System"
        
        if image_path:
            message += f"\n📸 Image: {image_path}"
        
        success = self._send_message(message)
        if success:
            self._update_cooldown("fall")
            print(f"🚨 TELEGRAM_SERVICE: Fall alert sent")
        
        return success
    
    def send_test_message(self) -> bool:
        """Send a test message to verify the service is working."""
        message = "🧪 *HomePal Telegram Service Test*\n\n✅ The Telegram notification service is working correctly!\n\n🕐 Time: " + datetime.now().strftime('%I:%M:%S %p on %B %d, %Y')
        
        success = self._send_message(message)
        if success:
            print("✅ TELEGRAM_SERVICE: Test message sent successfully")
        
        return success
    
    def get_cooldown_status(self) -> dict:
        """Get the current cooldown status for both notification types."""
        current_time = time.time()
        
        fire_remaining = max(0, self.cooldown_duration - (current_time - self._last_fire_notification))
        fall_remaining = max(0, self.cooldown_duration - (current_time - self._last_fall_notification))
        
        return {
            "fire_cooldown_remaining": round(fire_remaining, 1),
            "fall_cooldown_remaining": round(fall_remaining, 1),
            "fire_can_send": fire_remaining == 0,
            "fall_can_send": fall_remaining == 0
        }


# Global instance for easy access
telegram_service = TelegramService()

def send_fire_alert(confidence: float, image_path: Optional[str] = None) -> bool:
    """Convenience function to send fire alert."""
    return telegram_service.send_fire_alert(confidence, image_path)

def send_fall_alert(image_path: Optional[str] = None) -> bool:
    """Convenience function to send fall alert."""
    return telegram_service.send_fall_alert(image_path)

def send_test_message() -> bool:
    """Convenience function to send test message."""
    return telegram_service.send_test_message()

__all__ = ["TelegramService", "telegram_service", "send_fire_alert", "send_fall_alert", "send_test_message"]
