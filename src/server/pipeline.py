"""
Voice processing pipeline.

Orchestrates the full voice pipeline:
  AudioFrames -> STT -> AI (streaming) -> TTS (streaming) -> AudioChunks

Emits typed ServerEvents that the transport layer serializes to JSON.
"""

import re
import time
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from .backend import AIBackend
from .events import (
    AudioChunkEvent,
    ErrorEvent,
    LatencyMetrics,
    ResponseChunkEvent,
    ResponseCompleteEvent,
    SentenceEndEvent,
    SentenceStartEvent,
    ServerEvent,
    TranscriptEvent,
)
from .session import SessionContext
from .stt import WhisperSTT
from .text_utils import clean_for_speech
from .tts import ChatterboxTTS

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class VoicePipeline:
    """Transcribe buffered audio, stream AI response, yield synthesized speech events."""

    def __init__(
        self,
        stt: WhisperSTT,
        tts: ChatterboxTTS,
        backend: AIBackend,
        sample_rate: int = 16000,
    ):
        self._stt = stt
        self._tts = tts
        self._backend = backend
        self._sample_rate = sample_rate

    async def process_audio(
        self,
        audio_buffer: list[np.ndarray],
        session: SessionContext,
    ) -> AsyncIterator[ServerEvent]:
        """Transcribe audio, stream AI response, and yield speech events."""
        if not audio_buffer:
            return

        t_start = time.monotonic()

        try:
            audio_data = np.concatenate(audio_buffer)
            audio_duration = len(audio_data) / self._sample_rate

            t_stt_start = time.monotonic()
            transcript = await self._stt.transcribe(audio_data)
            t_stt = time.monotonic() - t_stt_start

            yield TranscriptEvent(text=transcript)

            if not transcript.strip():
                return

            full_response = ""
            sentence_buffer = ""
            agent_model = f"openclaw/{session.agent_id}" if session.agent_id else None

            audio_seq = 0
            t_llm_start = time.monotonic()
            t_first_token: Optional[float] = None
            t_first_audio: Optional[float] = None
            sentence_count = 0

            async for chunk in self._backend.chat_stream(
                transcript, model=agent_model, agent_hint=session.agent_id,
                reconnect=session.reconnect,
            ):
                full_response += chunk
                sentence_buffer += chunk

                if t_first_token is None:
                    t_first_token = time.monotonic() - t_llm_start

                yield ResponseChunkEvent(text=chunk)

                while True:
                    parts = _SENTENCE_SPLIT_RE.split(sentence_buffer, maxsplit=1)
                    if len(parts) < 2:
                        break
                    sentence, sentence_buffer = parts[0].strip(), parts[1]

                    if sentence:
                        async for event in self._speak_sentence(sentence, audio_seq):
                            yield event
                        sentence_count += 1
                        if t_first_audio is None:
                            t_first_audio = time.monotonic() - t_start

            if sentence_buffer.strip():
                async for event in self._speak_sentence(sentence_buffer.strip(), audio_seq):
                    yield event
                sentence_count += 1

            t_total = time.monotonic() - t_start
            t_llm_total = time.monotonic() - t_llm_start
            yield ResponseCompleteEvent(
                text=full_response,
                latency=LatencyMetrics(
                    stt_ms=round(t_stt * 1000),
                    first_token_ms=round(t_first_token * 1000) if t_first_token else None,
                    first_audio_ms=round(t_first_audio * 1000) if t_first_audio else None,
                    llm_total_ms=round(t_llm_total * 1000),
                    total_ms=round(t_total * 1000),
                    audio_duration_s=round(audio_duration, 1),
                    sentences=sentence_count,
                ),
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            yield ErrorEvent(message=str(e))

    async def _speak_sentence(
        self,
        text: str,
        seq: int,
    ) -> AsyncIterator[ServerEvent]:
        """Synthesize one sentence to audio chunks, yielding start/chunk/end events."""
        speech_text = clean_for_speech(text)
        if not speech_text:
            return

        yield SentenceStartEvent(seq=seq)
        try:
            async for audio_chunk in self._tts.synthesize_stream(speech_text):
                yield AudioChunkEvent(data=audio_chunk, seq=seq)
        except Exception as tts_err:
            logger.error(f"TTS error: {tts_err}")
            yield ErrorEvent(message=f"TTS synthesis failed: {tts_err}")
        yield SentenceEndEvent(seq=seq)
