#!/usr/bin/env python3
"""
Simple TTS synthesis service using gTTS that writes MP3 files to a directory.
"""
import os
import uuid
from typing import Tuple
from gtts import gTTS


class TTSService:
    def __init__(self, output_dir: str = "tts_audio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def synthesize_to_file(self, text: str, lang: str = "en") -> Tuple[str, str]:
        """Synthesize speech and save to an MP3 file.

        Returns a tuple of (filename, filepath).
        """
        filename = f"briefing_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(self.output_dir, filename)

        # Generate and save audio
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)

        return filename, filepath


