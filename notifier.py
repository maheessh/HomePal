import os
import sys
import time
import json
import threading
from datetime import datetime
from dotenv import load_dotenv
import requests # Use requests library to send messages

# --- Configuration ---
load_dotenv()
CWD = os.path.dirname(os.path.realpath(__file__))
EVENTS_FILE = "events.json"

# --- State and Locks ---
events_lock = threading.Lock()
# Tracks event IDs for which notifications have already been sent
SENT_NOTIFICATION_IDS = set()

# --- Telegram Alert Function ---
def send_telegram_alert(message: str):
    """Sends a message using the Telegram Bot API."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not all([bot_token, chat_id]):
        print("🔴 NOTIFIER: Telegram credentials (BOT_TOKEN or CHAT_ID) not set in .env file.")
        return

    # Construct the API URL
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown' # Optional: for bold/italic text
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Will raise an exception for HTTP errors
        print(f"✅ NOTIFIER: Telegram alert sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"❌ NOTIFIER: Failed to send Telegram alert: {e}")

# --- Main Notification Worker ---
def notification_worker():
    """Monitors events.json for new, critical events and sends Telegram notifications."""
    print("✅ NOTIFIER: Emergency notification worker started. Watching for critical events...")
    
    # On startup, read all existing events to avoid sending duplicate alerts for old events
    with events_lock:
        if os.path.exists(EVENTS_FILE):
            try:
                with open(EVENTS_FILE, "r") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        for event in data.get("events", []):
                            SENT_NOTIFICATION_IDS.add(event['id'])
                print(f"✅ NOTIFIER: Synced with {len(SENT_NOTIFICATION_IDS)} existing events.")
            except (IOError, json.JSONDecodeError):
                print("⚠️ NOTIFIER: Could not read events.json on startup.")

    # Main loop to check for new events
    while True:
        try:
            time.sleep(5) # Check every 5 seconds
            
            events = []
            with events_lock:
                if not os.path.exists(EVENTS_FILE): continue
                with open(EVENTS_FILE, "r") as f:
                    content = f.read().strip()
                    if content: events = json.loads(content).get("events", [])
            
            if not events: continue

            for event in events:
                event_id = event.get("id")
                if event_id in SENT_NOTIFICATION_IDS:
                    continue

                is_critical = event.get("metadata", {}).get("alert_level") == "critical"
                needs_notification = event.get("notification") is True

                if is_critical and needs_notification:
                    print(f"NOTIFIER: New critical event found: {event_id}")
                    
                    message = ""
                    event_name = event.get('class_name', 'Unknown Event')

                    if event_name.lower() == 'fall':
                        message = "Emergency Alert: A sudden fall or unusual activity was detected while your monitoring system was active."
                    else:
                        timestamp = datetime.fromisoformat(event['timestamp']).strftime('%I:%M:%S %p')
                        message = f"*HomePal Critical Alert:*\nA '{event_name}' event was detected at {timestamp}."
                    
                    threading.Thread(target=send_telegram_alert, args=(message,)).start()
                    SENT_NOTIFICATION_IDS.add(event_id)

        except Exception as e:
            print(f"ERROR in notifier loop: {e}")

if __name__ == "__main__":
    # Before starting, remind the user to start a chat with the bot
    print("---")
    print("IMPORTANT: Ensure you have started a conversation with your Telegram bot first!")
    print("---")
    notification_worker()

