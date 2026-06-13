"""
Runtime application state — shared singleton instances.

Populated by main.py's lifespan handler, consumed by route handlers.
"""

from typing import Optional

from .backend import AIBackend
from .pipeline import VoicePipeline
from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .vad import VoiceActivityDetector

stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None
pipeline: Optional[VoicePipeline] = None
_startup_time: float = 0.0
