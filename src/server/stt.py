"""
Speech-to-Text module using Whisper.
"""

import asyncio
import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        executor: Optional["ThreadPoolExecutor"] = None,
        remote_url: Optional[str] = None,
        remote_timeout: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self._executor = executor
        self.model = None
        self._backend = "mock"
        self._last_latency_ms: Optional[float] = None
        self._last_source: Optional[str] = None

        self._remote_url = remote_url.rstrip("/") if remote_url else None
        self._remote_client: Optional[httpx.AsyncClient] = None
        if self._remote_url:
            self._remote_client = httpx.AsyncClient(
                base_url=self._remote_url, timeout=httpx.Timeout(remote_timeout, connect=3.0)
            )
            logger.info(f"STT remote endpoint configured: {self._remote_url} — skipping local model load")
            self._backend = "remote"
        else:
            self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        if self.model is not None:
            return
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel

            if self.device == "auto":
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                    compute_type = "int8"
                else:
                    self.device = "cpu"
                    compute_type = "int8"
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("✅ faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

        # Try openai-whisper
        try:
            import whisper

            if self.device == "auto":
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("✅ openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")

        # Mock mode for testing
        logger.warning("⚠️ No STT backend - using mock mode")
        self._backend = "mock"

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        start = time.monotonic()
        if self._remote_client is not None:
            try:
                text = await self._transcribe_remote(audio)
                self._last_latency_ms = (time.monotonic() - start) * 1000
                self._last_source = "remote"
                return text
            except Exception as e:
                logger.warning(f"Remote STT failed ({e}), falling back to local")
                if self.model is None:
                    await asyncio.get_event_loop().run_in_executor(self._executor, self._load_model)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(self._executor, self._transcribe_sync, audio)
        self._last_latency_ms = (time.monotonic() - start) * 1000
        self._last_source = "local"
        return text

    async def _transcribe_remote(self, audio: np.ndarray) -> str:
        wav_bytes = self._encode_wav(audio)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model_name, "language": self.language}
        resp = await self._remote_client.post("/audio/transcriptions", data=data, files=files)
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

    @staticmethod
    def _encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def status(self) -> dict:
        """Return STT status dict for health checks."""
        return {
            "backend": self._backend,
            "model": self.model_name,
            "device": self.device,
            "remote_url": self._remote_url,
            "last_source": self._last_source,
            "latency_ms": round(self._last_latency_ms, 1) if self._last_latency_ms is not None else None,
        }

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        elif self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()

        else:
            # Mock mode - return placeholder
            logger.debug(f"Mock STT: received {len(audio)} samples")
            return "[Mock transcription - install whisper for real STT]"
