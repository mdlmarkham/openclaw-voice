"""
Voice Activity Detection module.
"""

from typing import Optional
import numpy as np
from loguru import logger


class VoiceActivityDetector:
    """Voice Activity Detection."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load VAD model."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self.model = model
            self._get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.warning(f"VAD not available: {e}")
            self.model = None
    
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio contains speech."""
        if self.model is None:
            return True  # Assume speech if no VAD
        try:
            import torch
            audio_tensor = torch.from_numpy(audio).float()
            speech_prob = self.model(audio_tensor, sample_rate).item()
            return speech_prob > self.threshold
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True
