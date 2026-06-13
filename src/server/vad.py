"""
Voice Activity Detection module.

Silero VAD integration + state-machine endpointing for automatic
end-of-utterance detection.
"""

from enum import Enum, auto
from typing import Optional
import numpy as np
from loguru import logger


class VADState(Enum):
    SILENT = auto()
    SPEAKING = auto()
    STOPPING = auto()


class VADEvent:
    """Events emitted by the VAD endpointing state machine."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
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
        """Return True if audio frame contains speech above threshold."""
        if self.model is None:
            return True
        try:
            import torch

            audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
            speech_prob = self.model(audio_tensor, sample_rate).item()
            return speech_prob > self.threshold
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True


class VADEndpoint:
    """
    State-machine based VAD endpointing.

    Tracks a simple 3-state machine (SILENT → SPEAKING → STOPPING → SILENT)
    and emits speech_start / speech_end events when transitions occur.

    The STOPPING state implements a hangover period: once speech ends,
    we wait min_silence_frames before declaring the utterance complete.
    This avoids clipping natural pauses in speech.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        threshold: float = 0.5,
        min_silence_frames: int = 20,  # ~600ms at 512-frame/16kHz
        min_speech_frames: int = 3,    # ~100ms
        sample_rate: int = 16000,
    ):
        self._vad = vad
        self._threshold = threshold
        self._min_silence_frames = min_silence_frames
        self._min_speech_frames = min_speech_frames
        self._sample_rate = sample_rate
        self._state = VADState.SILENT
        self._speech_frames = 0
        self._silence_frames = 0
        self._frame_count = 0

    def reset(self) -> None:
        """Reset state machine to initial silent state."""
        self._state = VADState.SILENT
        self._speech_frames = 0
        self._silence_frames = 0
        self._frame_count = 0

    @property
    def is_speaking(self) -> bool:
        return self._state in (VADState.SPEAKING, VADState.STOPPING)

    def process(self, audio: np.ndarray) -> Optional[str]:
        """
        Process one audio frame. Returns VADEvent or None.

        Args:
            audio: float32 PCM array at sample_rate

        Returns:
            VADEvent.SPEECH_START when user starts speaking
            VADEvent.SPEECH_END when user stops (after hangover)
            None if no state transition
        """
        has_speech = self._vad.is_speech(audio, self._sample_rate)

        if self._state == VADState.SILENT:
            if has_speech:
                self._speech_frames += 1
                if self._speech_frames >= self._min_speech_frames:
                    self._state = VADState.SPEAKING
                    self._speech_frames = 0
                    return VADEvent.SPEECH_START
            else:
                self._speech_frames = 0

        elif self._state == VADState.SPEAKING:
            if has_speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                if self._silence_frames >= self._min_silence_frames:
                    self._state = VADState.SILENT
                    self._silence_frames = 0
                    return VADEvent.SPEECH_END

        return None
