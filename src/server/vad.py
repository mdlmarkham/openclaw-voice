"""
Voice Activity Detection module.

Silero VAD integration + state-machine endpointing for automatic
end-of-utterance detection.
"""

import asyncio
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

    def __init__(self, threshold: float = 0.5, executor: Optional[object] = None):
        self.threshold = threshold
        self.model = None
        self._executor = executor
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
        """Return True if audio contains speech above threshold.

        Silero VAD requires fixed-size input windows (512 samples @16kHz,
        256 @8kHz). Incoming frames from the client are larger (e.g. 4096
        samples), so we chunk into model-sized windows and return True if
        ANY window is speech. This keeps the status indicator responsive
        regardless of the client's capture buffer size.
        """
        if self.model is None:
            return True
        try:
            import torch

            window = 512 if sample_rate == 16000 else 256
            audio = np.ascontiguousarray(audio).astype(np.float32, copy=False)
            # Pad to a multiple of window so the tail isn't dropped
            if len(audio) % window != 0:
                pad = window - (len(audio) % window)
                audio = np.pad(audio, (0, pad))
            for i in range(0, len(audio), window):
                chunk = audio[i : i + window]
                if len(chunk) < window:
                    break
                audio_tensor = torch.from_numpy(np.ascontiguousarray(chunk))
                speech_prob = self.model(audio_tensor, sample_rate).item()
                if speech_prob > self.threshold:
                    return True
            return False
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True

    async def is_speech_async(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Async wrapper running is_speech in executor."""
        if self._executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self.is_speech, audio, sample_rate)
        return self.is_speech(audio, sample_rate)


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

    async def process_async(self, audio: np.ndarray) -> Optional[str]:
        """Async version of process using async VAD.

        Iterates over model-sized windows so the endpointing state machine
        advances per-window (matching the min_silence/min_speech frame counts).
        Returns the LAST transition event, or None.
        """
        last_event = None
        for chunk in self._iter_windows(audio):
            has_speech = await self._vad.is_speech_async(chunk, self._sample_rate)
            ev = self._advance_state(has_speech)
            if ev:
                last_event = ev
        return last_event

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
        last_event = None
        for chunk in self._iter_windows(audio):
            has_speech = self._vad.is_speech(chunk, self._sample_rate)
            ev = self._advance_state(has_speech)
            if ev:
                last_event = ev
        return last_event

    def _iter_windows(self, audio: np.ndarray):
        """Yield model-sized windows (512 @16k / 256 @8k) from the input."""
        window = 512 if self._sample_rate == 16000 else 256
        audio = np.ascontiguousarray(audio).astype(np.float32, copy=False)
        for i in range(0, len(audio) - window + 1, window):
            yield audio[i : i + window]

    def _advance_state(self, has_speech: bool) -> Optional[str]:
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
