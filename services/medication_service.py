#!/usr/bin/env python3
"""
Medication reminder service that handles:
- Checking for upcoming medication reminders
- Playing TTS announcements when it's time to take medication
- Managing reminder state to avoid duplicate announcements
"""
import os
import json
import time
import threading
from datetime import datetime, time as time_obj
from gtts import gTTS
import pygame
import io

class MedicationReminderService:
    def __init__(self, medications_file="medications.json"):
        self.medications_file = medications_file
        self.announced_today = set()  # Track which reminders were already announced today
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Initialize pygame mixer for audio
        try:
            pygame.mixer.init()
            print("✅ Audio system initialized for medication reminders")
        except Exception as e:
            print(f"⚠️ Audio initialization failed: {e}")
    
    def read_medications(self):
        """Read medications from file"""
        try:
            if not os.path.exists(self.medications_file):
                return []
            with open(self.medications_file, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading medications: {e}")
            return []
    
    def should_announce_reminder(self, medication, reminder_time_str):
        """Check if this reminder should be announced (not already announced today)"""
        today = datetime.now().strftime('%Y-%m-%d')
        reminder_key = f"{today}_{medication['id']}_{reminder_time_str}"
        
        with self.lock:
            if reminder_key in self.announced_today:
                return False
            self.announced_today.add(reminder_key)
            return True
    
    def create_announcement(self, medication):
        """Create TTS announcement for medication"""
        message = f"It's time to take your medication: {medication['name']}. Please take {medication['dosage']}."
        
        try:
            # Create TTS audio
            tts = gTTS(text=message, lang='en', slow=False)
            
            # Save to memory buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Play the audio
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print(f"🔊 Medication reminder announced: {medication['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating announcement for {medication['name']}: {e}")
            return False
    
    def check_reminders(self):
        """Check for medications that need to be announced"""
        medications = self.read_medications()
        now = datetime.now()
        current_time = now.time()
        
        for medication in medications:
            try:
                times_str = medication.get('times', '')
                if not times_str:
                    continue
                
                # Parse times (format: "8:00 AM, 8:00 PM")
                reminder_times = [t.strip() for t in times_str.split(',')]
                
                for time_str in reminder_times:
                    try:
                        # Parse time (format: "8:00 AM" or "8:00 PM")
                        reminder_time = datetime.strptime(time_str, '%I:%M %p').time()
                        
                        # Check if it's time to announce (within 1 minute window)
                        time_diff = abs((current_time.hour * 60 + current_time.minute) - 
                                      (reminder_time.hour * 60 + reminder_time.minute))
                        
                        if time_diff <= 1:  # Within 1 minute
                            if self.should_announce_reminder(medication, time_str):
                                self.create_announcement(medication)
                        
                    except ValueError as e:
                        print(f"Error parsing time '{time_str}' for medication {medication['name']}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing medication {medication.get('name', 'Unknown')}: {e}")
                continue
    
    def cleanup_old_announcements(self):
        """Clean up old announcement records (keep only today's)"""
        today = datetime.now().strftime('%Y-%m-%d')
        with self.lock:
            self.announced_today = {key for key in self.announced_today if key.startswith(today)}
    
    def reminder_worker(self):
        """Background worker that checks for reminders every minute"""
        print("🔄 Medication reminder worker started")
        
        while self.running:
            try:
                self.check_reminders()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"❌ Error in reminder worker: {e}")
                time.sleep(60)
    
    def start(self):
        """Start the reminder service"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.reminder_worker, daemon=True)
        self.thread.start()
        print("✅ Medication reminder service started")
    
    def stop(self):
        """Stop the reminder service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 Medication reminder service stopped")
    
    def test_announcement(self, medication_name="Test Medication", dosage="1 tablet"):
        """Test TTS announcement with given medication details"""
        test_medication = {
            'name': medication_name,
            'dosage': dosage
        }
        return self.create_announcement(test_medication)
