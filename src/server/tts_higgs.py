"""
Higgs Audio v3 TTS backend via Boson Cloud API.

Higgs offers:
- Cloud API (Boson) with streaming PCM output
- Local GPU serving via SGLang-Omni (same API interface)
- Agent voice cloning via ref_audio
- Control tokens for emotion, prosody, style
"""

import os
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
from loguru import logger

from .config import settings

BOSON_API_URL = "https://api.boson.ai/v1/audio/speech"


class HiggsTTS:
    """Higgs Audio v3 TTS via Boson Cloud API.

    Produces 16-bit/24kHz/mono PCM. Same output format as Supertonic
    post-resample, so no conversion needed in the pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_voice: str = "eleanor",
        model: str = "higgs-audio-v3-tts",
        timeout: int = 3,
    ):
        self._api_key = api_key or os.getenv("BOSON_API_KEY") or settings.boson_api_key
        self._default_voice = default_voice or settings.higgs_default_voice
        self._model = model or settings.higgs_model
        self._timeout = timeout or settings.higgs_timeout_seconds
        self._backend = "higgs" if self._api_key else "mock"
        self._client: Optional[httpx.AsyncClient] = None

        if self._api_key:
            self._client = httpx.AsyncClient(
                base_url=settings.higgs_api_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout, connect=2.0),
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream PCM audio from Higgs via Boson API.

        Yields int16 PCM bytes at 24kHz mono.
        """
        if not self._client:
            logger.warning("Higgs TTS not available (no API key)")
            return

        payload = {
            "model": self._model,
            "input": text,
            "voice": voice or self._default_voice,
            "response_format": "pcm",
            "stream": True,
        }

        try:
            async with self._client.stream("POST", "", json=payload) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(
                        f"Higgs API error {resp.status_code}: {error_body[:200]}"
                    )
                    return

                async for raw_chunk in resp.aiter_bytes():
                    if raw_chunk:
                        yield raw_chunk
        except httpx.TimeoutException:
            logger.warning("Higgs API timed out — falling back")
        except httpx.RequestError as e:
            logger.warning(f"Higgs API request failed: {e}")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> np.ndarray:
        """Non-streaming synthesis. Returns float32 PCM array at 24kHz."""
        chunks = []
        async for chunk in self.synthesize_stream(text, voice=voice):
            chunks.append(chunk)

        if not chunks:
            return np.zeros(12000, dtype=np.float32)

        raw = b"".join(chunks)
        pcm_s16 = np.frombuffer(raw, dtype=np.int16)
        return pcm_s16.astype(np.float32) / 32768.0

    def status(self) -> dict:
        return {
            "backend": "higgs",
            "available": self.available,
            "voice": self._default_voice,
            "model": self._model,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
