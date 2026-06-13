"""
Text-to-Speech module — Olympus Edition

Backend priority: ElevenLabs > Supertonic (local CPU) > Edge TTS (cloud) > Chatterbox > XTTS > mock

Supertonic 3: 99M params, ONNX Runtime, 31 languages, ~1-2x realtime on CPU.
Edge TTS: Cloud-based, fast, requires network.
"""

import asyncio
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, AsyncGenerator

if TYPE_CHECKING:
    from .tts_higgs import HiggsTTS

import numpy as np
from loguru import logger

from .audio_utils import float32_to_int16


class ChatterboxTTS:
    """Text-to-Speech using multiple backends with intelligent fallback."""

    # Agent-to-voice mapping for dynamic voice switching
    AGENT_VOICE_MAP = {
        "metis": "F2",
        "atlas": "M2",
        "hephaestus": "M4",
        "clio": "F4",
        "deepthought": "M1",
        "mara": "F5",
    }

    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,
        executor: Optional["ThreadPoolExecutor"] = None,
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"  # Jessica
        self._edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")
        self._supertonic_voice = os.environ.get("SUPERTONIC_VOICE", "F2")
        self._supertonic_model = os.environ.get("SUPERTONIC_MODEL", "supertonic-2")
        self._executor = executor
        self.model = None
        self._backend = "mock"
        self._elevenlabs_client = None
        self._supertonic_tts = None
        self._supertonic_style = None
        self._supertonic_sr = 44100  # Supertonic outputs at 44100Hz
        self._load_model()

    def _load_model(self):
        """Load the TTS model. Priority: ElevenLabs > Supertonic > Edge > Chatterbox > XTTS."""
        # 1. ElevenLabs (cloud, premium)
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            try:
                from elevenlabs import ElevenLabs

                self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                self._backend = "elevenlabs"
                logger.info("✅ ElevenLabs TTS ready")
                return
            except ImportError:
                logger.warning(
                    "ElevenLabs SDK not installed — add 'elevenlabs' to requirements.txt"
                )
            except Exception as e:
                logger.warning(f"ElevenLabs failed: {e}")

        # 2. Supertonic (local CPU, ONNX, fast, 31 languages)
        try:
            from supertonic import TTS

            self._supertonic_tts = TTS(model=self._supertonic_model)
            self._supertonic_sr = self._supertonic_tts.sample_rate
            # Load voice style
            if self._supertonic_voice in self._supertonic_tts.voice_style_names:
                self._supertonic_style = self._supertonic_tts.get_voice_style(
                    self._supertonic_voice
                )
            else:
                logger.warning(
                    f"Supertonic voice '{self._supertonic_voice}' not found, using first available"
                )
                self._supertonic_style = self._supertonic_tts.get_voice_style(
                    self._supertonic_tts.voice_style_names[0]
                )
            self._backend = "supertonic"
            logger.info(
                f"✅ Supertonic TTS ready (model={self._supertonic_model}, voice={self._supertonic_voice}, sr={self._supertonic_sr}Hz)"
            )
            return
        except ImportError:
            logger.debug("supertonic not installed, skipping")
        except Exception as e:
            logger.warning(f"Supertonic failed: {e}")

        # 3. Edge TTS (cloud, free, fast)
        try:
            import edge_tts  # noqa: F401

            self._backend = "edge"
            logger.info("✅ Edge TTS ready (Microsoft, free, cloud)")
            return
        except ImportError:
            logger.warning("edge-tts not installed")
        except Exception as e:
            logger.warning(f"Edge TTS check failed: {e}")

        # 4. Chatterbox (GPU local)
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel

            logger.info("Loading Chatterbox TTS...")
            self.model = CBModel.from_pretrained(device=self._get_device())
            self._backend = "chatterbox"
            logger.info("✅ Chatterbox loaded")
            return
        except ImportError:
            logger.warning("Chatterbox not installed")
        except Exception as e:
            logger.warning(f"Chatterbox failed: {e}")

        # 5. XTTS (GPU local)
        try:
            from TTS.api import TTS

            logger.info("Loading Coqui XTTS...")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._backend = "xtts"
            logger.info("✅ XTTS loaded")
            return
        except ImportError:
            logger.warning("Coqui TTS not installed")
        except Exception as e:
            logger.warning(f"XTTS failed: {e}")

        logger.warning("⚠️ No TTS backend - using mock mode (silence)")
        self._backend = "mock"

    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def sample_rate(self) -> int:
        """Output sample rate for the current backend."""
        if self._backend == "supertonic":
            return self._supertonic_sr
        elif self._backend == "elevenlabs":
            return 24000
        else:
            return 24000  # Edge TTS and others output at 24kHz

    def status(self) -> dict:
        """Return TTS status dict for health checks. Avoids reaching into private attrs."""
        result = {"backend": self._backend}
        if self._backend == "supertonic":
            result.update(
                {
                    "model": self._supertonic_model,
                    "voice": self._supertonic_voice,
                    "sample_rate": self._supertonic_sr,
                    "agent_voice_map": self.AGENT_VOICE_MAP,
                }
            )
        elif self._backend == "edge":
            result["edge_voice"] = self._edge_voice
        elif self._backend == "elevenlabs":
            result["voice_id"] = self.voice_id
            result["sample_rate"] = 24000
        return result

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text. Returns float32 numpy array at native sample rate."""
        if self._backend == "supertonic":
            return await self._synthesize_supertonic(text)
        elif self._backend == "edge":
            return await self._synthesize_edge(text)
        elif self._backend == "elevenlabs":
            return await self._synthesize_elevenlabs(text)
        elif self._backend in ("chatterbox", "xtts"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._synthesize_sync_local, text)
        else:
            return np.zeros(12000, dtype=np.float32)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.

        Yields:
            Raw PCM audio chunks (int16 at native sample rate)
        """
        try:
            async for chunk in self._synthesize_stream_primary(text):
                yield chunk
        except Exception as e:
            logger.error(f"Primary TTS ({self._backend}) failed: {e}")
            # Try fallback
            fallback = self._get_fallback_backend()
            if fallback:
                logger.info(f"Trying fallback TTS: {fallback._backend}")
                try:
                    async for chunk in fallback.synthesize_stream(text):
                        yield chunk
                except Exception as e2:
                    logger.error(f"Fallback TTS ({fallback._backend}) also failed: {e2}")
            else:
                logger.warning("No fallback TTS available")

    def _get_fallback_backend(self) -> Optional["ChatterboxTTS"]:
        """Get a fallback TTS backend if the primary fails."""
        if self._backend == "supertonic":
            # Try Edge TTS as fallback — just delegate to our own _synthesize_edge
            # by creating a lightweight wrapper that uses the same instance
            return _EdgeFallback(self._edge_voice)
        return None

    async def _synthesize_stream_primary(self, text: str) -> AsyncGenerator[bytes, None]:
        """Primary TTS streaming (before fallback)."""
        if self._backend == "elevenlabs":
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                for chunk in audio_generator:
                    yield chunk
            except Exception as e:
                logger.error(f"ElevenLabs streaming error: {e}")

        elif self._backend == "supertonic":
            # Supertonic is synchronous but fast (~1-2x realtime on CPU)
            try:
                loop = asyncio.get_event_loop()
                audio_float32 = await loop.run_in_executor(
                    self._executor, self._synthesize_supertonic_sync, text
                )
                if audio_float32 is not None and len(audio_float32) > 0:
                    audio_int16 = float32_to_int16(audio_float32)
                    yield audio_int16.tobytes()
            except Exception as e:
                logger.error(f"Supertonic TTS error: {e}")

        elif self._backend == "edge":
            # Edge TTS: collect MP3, decode to PCM, send as int16
            try:
                audio_float32 = await self._synthesize_edge(text)
                if audio_float32 is not None and len(audio_float32) > 0:
                    audio_int16 = float32_to_int16(audio_float32)
                    yield audio_int16.tobytes()
            except Exception as e:
                logger.error(f"Edge TTS streaming error: {e}")

        else:
            # Fallback: synthesize then yield as one chunk
            try:
                audio = await self.synthesize(text)
                if audio is not None and len(audio) > 0:
                    audio_int16 = float32_to_int16(audio)
                    yield audio_int16.tobytes()
            except Exception as e:
                logger.error(f"TTS fallback error: {e}")

    def _synthesize_supertonic_sync(self, text: str) -> np.ndarray:
        """Synchronous Supertonic synthesis (runs in executor)."""
        try:
            result = self._supertonic_tts.synthesize(text, voice_style=self._supertonic_style)
            audio = result[0].squeeze()  # shape (1, N) → (N,)
            # Resample from 44100Hz to 24000Hz for client compatibility
            # Use soxr (0.002s) instead of librosa (4.8s) — 2500x faster
            if self._supertonic_sr != 24000:
                import soxr

                audio = soxr.resample(audio, self._supertonic_sr, 24000)
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Supertonic synthesis error: {e}")
            raise

    async def _synthesize_supertonic(self, text: str) -> np.ndarray:
        """Supertonic synthesis via executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._synthesize_supertonic_sync, text)

    async def _synthesize_edge(self, text: str) -> np.ndarray:
        """Edge TTS — fully async, collect MP3 chunks and decode."""
        try:
            import edge_tts

            voice = self._edge_voice or "en-US-JennyNeural"
            communicate = edge_tts.Communicate(text, voice)

            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)

            import soundfile as sf

            data, sr = sf.read(audio_buffer)
            if sr != 24000:
                import soxr

                data = soxr.resample(data, sr, 24000)
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            raise

    async def _synthesize_elevenlabs(self, text: str) -> np.ndarray:
        """ElevenLabs synthesis."""
        try:
            audio_generator = self._elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="pcm_24000",
            )
            audio_bytes = b"".join(audio_generator)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_array.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise

    def _synthesize_sync_local(self, text: str) -> np.ndarray:
        """Synchronous synthesis for local models (chatterbox, xtts)."""
        if self._backend == "chatterbox":
            if self.voice_sample:
                audio = self.model.generate(text, audio_prompt=self.voice_sample)
            else:
                audio = self.model.generate(text)
            return audio.cpu().numpy().astype(np.float32)

        elif self._backend == "xtts":
            if self.voice_sample:
                wav = self.model.tts(text=text, speaker_wav=self.voice_sample, language="en")
            else:
                wav = self.model.tts(text=text, language="en")
            return np.array(wav, dtype=np.float32)

        else:
            logger.debug(f"Mock TTS: '{text[:50]}...'")
            return np.zeros(12000, dtype=np.float32)


class _EdgeFallback:
    """Lightweight Edge TTS fallback — used when Supertonic fails."""

    def __init__(self, voice: str = "en-US-JennyNeural"):
        self._backend = "edge"
        self._edge_voice = voice
        self.sample_rate = 24000

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Edge TTS streaming fallback."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, self._edge_voice)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)

            import soundfile as sf

            data, sr = sf.read(audio_buffer)
            if sr != 24000:
                import soxr

                data = soxr.resample(data, sr, 24000)
            audio_int16 = float32_to_int16(data)
            yield audio_int16.tobytes()
        except Exception as e:
            logger.error(f"Edge TTS fallback error: {e}")


class TTSRouter:
    """Routes TTS requests to the best available backend.

    Primary: Higgs (cloud API, low latency, expressive)
    Fallback: Supertonic (local CPU, always available)
    Edge case: Direct to Supertonic when Higgs can't load or is disabled.
    """

    def __init__(
        self,
        supertonic: ChatterboxTTS,
        higgs: Optional["HiggsTTS"] = None,
        backend: str = "auto",
    ):
        self._supertonic = supertonic
        self._higgs = higgs
        self._backend = backend

    @property
    def supertonic(self) -> "ChatterboxTTS":
        return self._supertonic

    @property
    def available(self) -> bool:
        return self._supertonic is not None

    @property
    def active_backend(self) -> str:
        if self._backend == "higgs" and self._higgs and self._higgs.available:
            return "higgs"
        return self._supertonic._backend if self._supertonic else "mock"

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        agent_hint: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech using the best available backend."""
        prefer_higgs = self._backend != "supertonic" and self._higgs and self._higgs.available

        if prefer_higgs:
            # Inject control tokens for personality
            enriched = _inject_control_tokens(text, agent_hint) if agent_hint else text
            async for chunk in self._higgs.synthesize_stream(enriched, voice=voice):
                yield chunk
            return

        if self._supertonic:
            clean = _strip_control_tokens(text) if not prefer_higgs else text
            async for chunk in self._supertonic.synthesize_stream(clean):
                yield chunk

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        agent_hint: Optional[str] = None,
    ) -> np.ndarray:
        chunks = []
        async for chunk in self.synthesize_stream(text, voice=voice, agent_hint=agent_hint):
            chunks.append(chunk)
        if not chunks:
            return np.zeros(12000, dtype=np.float32)
        raw = b"".join(chunks)
        pcm_s16 = np.frombuffer(raw, dtype=np.int16)
        return pcm_s16.astype(np.float32) / 32768.0

    def status(self) -> dict:
        base = self._supertonic.status() if self._supertonic else {"backend": "not_loaded"}
        base["router_backend"] = self.active_backend
        base["higgs_available"] = bool(self._higgs and self._higgs.available)
        return base

    async def close(self) -> None:
        if self._higgs:
            await self._higgs.close()


# Agent control token definitions
_AGENT_CONTROL_TOKENS = {
    "metis": "<|emotion:contemplation|><|prosody:pause|>",
    "atlas": "<|emotion:determination|>",
    "hephaestus": "<|emotion:contentment|><|prosody:speed_slow|>",
    "clio": "<|emotion:contemplation|><|prosody:speed_slow|>",
    "deepthought": "<|emotion:enthusiasm|>",
    "mara": "<|emotion:affection|><|prosody:pause|>",
}

_CONTROL_TOKEN_RE = re.compile(r"<\|[a-z_]+:[a-z_]+(?::[a-z_]+)?\|>")


def _inject_control_tokens(text: str, agent_hint: Optional[str]) -> str:
    """Prepend Higgs control tokens for the given agent."""
    tokens = _AGENT_CONTROL_TOKENS.get(agent_hint or "")
    if tokens:
        return f"{tokens} {text}"
    return text


def _strip_control_tokens(text: str) -> str:
    """Remove Higgs control tokens (for fallback to non-Higgs TTS)."""
    return _CONTROL_TOKEN_RE.sub("", text).strip()
