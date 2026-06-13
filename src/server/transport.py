"""
Pluggable audio transport layer.

Defines the AudioTransport ABC and provides a WebSocketTransport implementation.
The transport decouples message send/receive from the WebSocket handler,
allowing alternative transports (e.g. WebRTC) without touching pipeline or route code.
"""

import base64
import json
from abc import ABC, abstractmethod
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .events import (
    AudioChunkEvent,
    ErrorEvent,
    ResponseChunkEvent,
    ResponseCompleteEvent,
    SentenceEndEvent,
    SentenceStartEvent,
    ServerEvent,
    TranscriptEvent,
)


class AudioTransport(ABC):
    """Pluggable transport for voice audio and events.

    Implementations handle the wire protocol — JSON framing, audio encoding,
    connection lifecycle — while the voice pipeline and HTTP routes stay
    transport-agnostic.
    """

    @abstractmethod
    async def recv_message(self) -> Optional[dict]:
        """Receive the next client message. Returns None on clean disconnect."""
        ...

    @abstractmethod
    async def send_event(self, event: ServerEvent) -> None:
        """Serialize and send a typed pipeline event to the client."""
        ...

    @abstractmethod
    async def send_json(self, data: dict) -> None:
        """Send a raw JSON message (control acks, status, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        ...


class WebSocketTransport(AudioTransport):
    """AudioTransport implementation over a FastAPI WebSocket."""

    def __init__(self, websocket: WebSocket):
        self._ws = websocket

    async def recv_message(self) -> Optional[dict]:
        try:
            data = await self._ws.receive_text()
        except WebSocketDisconnect:
            return None
        except Exception as e:
            logger.error(f"Transport receive error: {e}")
            return None

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Transport received invalid JSON: {data[:100]}")
            return None

    async def send_event(self, event: ServerEvent) -> None:
        if isinstance(event, TranscriptEvent):
            await self._ws.send_json(
                {"type": "transcript", "text": event.text, "final": event.final}
            )
        elif isinstance(event, ResponseChunkEvent):
            await self._ws.send_json({"type": "response_chunk", "text": event.text})
        elif isinstance(event, SentenceStartEvent):
            await self._ws.send_json({"type": "sentence_start", "seq": event.seq})
        elif isinstance(event, AudioChunkEvent):
            await self._ws.send_json(
                {
                    "type": "audio_chunk",
                    "data": base64.b64encode(event.data).decode(),
                    "sample_rate": event.sample_rate,
                    "seq": event.seq,
                }
            )
        elif isinstance(event, SentenceEndEvent):
            await self._ws.send_json({"type": "sentence_end", "seq": event.seq})
        elif isinstance(event, ResponseCompleteEvent):
            await self._ws.send_json(
                {
                    "type": "response_complete",
                    "text": event.text,
                    "latency": {
                        "stt_ms": event.latency.stt_ms,
                        "first_token_ms": event.latency.first_token_ms,
                        "first_audio_ms": event.latency.first_audio_ms,
                        "llm_total_ms": event.latency.llm_total_ms,
                        "total_ms": event.latency.total_ms,
                        "audio_duration_s": event.latency.audio_duration_s,
                        "sentences": event.latency.sentences,
                    },
                }
            )
        elif isinstance(event, ErrorEvent):
            await self._ws.send_json({"type": "error", "message": event.message})

    async def send_json(self, data: dict) -> None:
        await self._ws.send_json(data)

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
