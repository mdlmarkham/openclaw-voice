"""
Typed events emitted by the VoicePipeline.

Each event maps to a WebSocket JSON message type.
The transport layer (WebSocket handler) serializes these to JSON.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class TranscriptEvent:
    text: str
    final: bool = True


@dataclass
class ResponseChunkEvent:
    text: str


@dataclass
class SentenceStartEvent:
    seq: int


@dataclass
class AudioChunkEvent:
    data: bytes
    sample_rate: int = 24000
    seq: int = 0


@dataclass
class SentenceEndEvent:
    seq: int


@dataclass
class LatencyMetrics:
    stt_ms: float
    first_token_ms: Optional[float]
    first_audio_ms: Optional[float]
    llm_total_ms: float
    total_ms: float
    audio_duration_s: float
    sentences: int


@dataclass
class ResponseCompleteEvent:
    text: str
    latency: LatencyMetrics


@dataclass
class ErrorEvent:
    message: str


ServerEvent = Union[
    TranscriptEvent,
    ResponseChunkEvent,
    SentenceStartEvent,
    AudioChunkEvent,
    SentenceEndEvent,
    ResponseCompleteEvent,
    ErrorEvent,
]
