"""
OpenRouter API service for generating AI briefings using Gemini 2.0 Flash.
"""
import os
import requests
import json
from typing import Dict, Any, Optional


class OpenRouterService:
    """Service for interacting with OpenRouter API using Gemini 2.0 Flash 001 model."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "google/gemini-2.0-flash-001"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    def generate_briefing(self, events_data: list, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate a security briefing from events data using Gemini 2.0 Flash.
        
        Args:
            events_data: List of events from events.json
            system_prompt: Optional custom system prompt
            
        Returns:
            Dict containing the briefing response
        """
        if not system_prompt:
            system_prompt = """You are a professional security analyst AI. Provide clear, concise summaries of security events for a homeowner. 
            
            Your task is to:
            1. Analyze the provided security events
            2. Identify patterns and trends
            3. Highlight critical incidents
            4. Note the busiest time periods
            5. Provide actionable insights
            
            Format your response in clear, readable markdown with proper headings and bullet points.
            Use **bold** for key findings and critical information.
            Keep the tone professional but accessible for homeowners."""
        
        # Format events for the prompt
        events_summary = self._format_events_for_prompt(events_data)
        
        user_query = f"""Please analyze these security events from my home monitoring system and provide a comprehensive briefing:

{events_summary}

Please provide:
1. **Executive Summary** - Key highlights and overall situation
2. **Event Analysis** - Breakdown of different event types
3. **Critical Incidents** - Any high-priority alerts that need attention
4. **Patterns & Trends** - Time patterns, frequency analysis
5. **Recommendations** - Actionable advice for improving security

Focus on insights that would be valuable for a homeowner managing their security system."""

        try:
            response = self._make_api_request(user_query, system_prompt)
            return {
                "success": True,
                "briefing": response.get("choices", [{}])[0].get("message", {}).get("content", "No response generated"),
                "model_used": self.model
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_used": self.model
            }

    def generate_micro_briefing(self, events_data: list) -> Dict[str, Any]:
        """Generate a SINGLE short, speech-friendly sentence summarizing events.

        Constraints:
        - 6 to 14 words total
        - Clear, natural speech
        - Present tense, calm tone
        - No list formatting, no markdown, no emojis
        - End with a period
        """
        system_prompt = (
            "You are a safety narrator. Compose exactly one concise sentence (6-14 words) "
            "that summarizes the provided home security events for a spoken briefing. "
            "Keep it calm, reassuring, and inclusive. No lists, no bullets, no headings, "
            "no emojis, no extra whitespace. End with a period."
        )

        events_summary = self._format_events_for_prompt(events_data)
        user_query = (
            "Generate one short sentence for a spoken briefing about these events.\n\n"
            f"{events_summary}\n\n"
            "One sentence only."
        )

        try:
            response = self._make_api_request(user_query, system_prompt)
            raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            sentence = (raw or "").strip().replace("\n", " ")
            # Post-process to enforce single short sentence
            if sentence:
                # If model returns multiple sentences, take the first
                sentence = sentence.split(".")[0].strip()
                if sentence:
                    sentence += "."
            if not sentence:
                sentence = "All quiet at home with no concerning activity."

            return {"success": True, "sentence": sentence, "model_used": self.model}
        except Exception as e:
            return {"success": False, "error": str(e), "model_used": self.model}
    
    def _format_events_for_prompt(self, events_data: list) -> str:
        """Format events data into a readable string for the AI prompt."""
        if not events_data:
            return "No events found in the system."
        
        formatted_events = []
        for event in events_data:
            timestamp = event.get("timestamp", "Unknown time")
            module = event.get("module", "Unknown module")
            class_name = event.get("class_name", "Unknown event")
            alert_level = event.get("metadata", {}).get("alert_level", "unknown")
            description = event.get("description", "No description available")
            
            formatted_events.append(
                f"• {timestamp} - {module.upper()}: {class_name} ({alert_level} priority)\n"
                f"  Description: {description}"
            )
        
        return "\n\n".join(formatted_events)
    
    def _make_api_request(self, user_query: str, system_prompt: str) -> Dict[str, Any]:
        """Make the actual API request to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",  # Optional: for analytics
            "X-Title": "HomePal Security System"  # Optional: for analytics
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": user_query
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def test_connection(self) -> bool:
        """Test if the OpenRouter API connection is working."""
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello, this is a test."}],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception:
            return False
