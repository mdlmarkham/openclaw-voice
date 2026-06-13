"""
Text-to-Speech module using ElevenLabs, Chatterbox, or fallbacks.

Olympus optimization: Edge TTS now runs natively async (no more event loop
errors from running asyncio.run inside ThreadPoolExecutor threads).
"""

import asyncio
import io
import os
from typing import Optional, AsyncGenerator
from pathlib import Path

import numpy as np
from loguru import logger


class ChatterboxTTS:
    """Text-to-Speech using ElevenLabs, Chatterbox, or fallbacks."""

    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"  # Jessica
        self._edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-JennyNeural")
        self.model = None
        self._backend = "mock"
        self._elevenlabs_client = None
        self._loop = None  # Store our own event loop for Edge TTS
        self._load_model()

    def _load_model(self):
        """Load the TTS model."""
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            try:
                from elevenlabs import ElevenLabs
                self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                self._backend = "elevenlabs"
                logger.info("✅ ElevenLabs TTS ready")
                return
            except ImportError:
                logger.warning("ElevenLabs SDK not installed, trying pip install...")
                try:
                    import subprocess
                    subprocess.check_call(["pip", "install", "elevenlabs", "-q"])
                    from elevenlabs import ElevenLabs
                    self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                    self._backend = "elevenlabs"
                    logger.info("✅ ElevenLabs TTS ready (auto-installed)")
                    return
                except Exception as e:
                    logger.warning(f"ElevenLabs auto-install failed: {e}")
            except Exception as e:
                logger.warning(f"ElevenLabs failed: {e}")

        try:
            import edge_tts  # noqa: F401
            self._backend = "edge"
            logger.info("✅ Edge TTS ready (Microsoft, free, no API key)")
            return
        except ImportError:
            logger.warning("edge-tts not installed")
        except Exception as e:
            logger.warning(f"Edge TTS check failed: {e}")

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

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text. Returns float32 numpy array at 24kHz."""
        if self._backend == "edge":
            return await self._synthesize_edge(text)
        elif self._backend == "elevenlabs":
            return await self._synthesize_elevenlabs(text)
        elif self._backend in ("chatterbox", "xtts"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._synthesize_sync_local, text)
        else:
            return np.zeros(12000, dtype=np.float32)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.

        Yields:
            Raw PCM audio chunks (24kHz, 16-bit signed integer)
        """
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
        elif self._backend == "edge":
            # Edge TTS: async-native, runs in the current event loop
            try:
                import edge_tts

                voice = self._edge_voice or "en-US-JennyNeural"
                communicate = edge_tts.Communicate(text, voice)
                audio_buffer = io.BytesIO()

                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]

            except Exception as e:
                logger.error(f"Edge TTS streaming error: {e}")
        else:
            # Fallback: synthesize then yield as one chunk
            try:
                audio = await self.synthesize(text)
                if audio is not None and len(audio) > 0:
                    # Convert float32 to int16 PCM bytes
                    audio_int16 = (audio * 32768.0).astype(np.int16)
                    yield audio_int16.tobytes()
            except Exception as e:
                logger.error(f"TTS fallback error: {e}")

    async def _synthesize_edge(self, text: str) -> np.ndarray:
        """Edge TTS — fully async, no event loop hacks."""
        try:
            import edge_tts

            voice = self._edge_voice or "en-US-JennyNeural"
            communicate = edge_tts.Communicate(text, voice)
            audio_buffer = io.BytesIO()
            await communicate.save_to_buffer(audio_buffer)
            audio_buffer.seek(0)

            # Decode MP3 to float32 PCM
            import soundfile as sf
            data, sr = sf.read(audio_buffer)
            if sr != 24000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=24000)
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return np.zeros(16000, dtype=np.float32)

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
            return np.zeros(16000, dtype=np.float32)

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