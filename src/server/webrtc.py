"""
WebRTC audio transport with RTP audio and data channel signaling.

Uses aiortc for WebRTC peer connection management.
Signaling is done via a simple HTTP SDP exchange (POST /api/webrtc/offer).
"""

import asyncio
import fractions
import json
import uuid
from typing import Optional

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler
from loguru import logger

from .audio_utils import float32_to_int16, int16_to_float32
from .events import AudioChunkEvent, ErrorEvent, ServerEvent
from .transport import AudioTransport

STUN_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]


class AudioInputTrack(MediaStreamTrack):
    """Wraps the remote audio track, resampling decoded PCM to 16kHz.

    ``recv()`` delegates to the wrapped remote track (the actual RTP
    receiver). A background pump task continuously drains the remote track
    and pushes resampled float32 PCM onto an internal queue that the session
    collector drains via ``read_frame()``.
    """

    kind = "audio"

    def __init__(self, track: MediaStreamTrack, sample_rate: int = 16000):
        super().__init__()
        self._track = track
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue = asyncio.Queue()
        self._resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )
        self._pump_task = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        """Continuously read frames from the remote track into the queue."""
        try:
            while True:
                frame = await self._track.recv()
                resampled = self._resampler.resample(frame)
                for f in resampled:
                    raw = f.to_ndarray()
                    pcm = int16_to_float32(raw.squeeze())
                    self._queue.put_nowait(pcm.squeeze())
        except asyncio.CancelledError:
            pass
        except (MediaStreamError, ConnectionError) as e:
            logger.debug(f"AudioInputTrack pump stopped: {e}")

    async def recv(self):
        return await self._track.recv()

    async def read_frame(self) -> Optional[np.ndarray]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

    def stop(self) -> None:
        self._pump_task.cancel()
        super().stop()


class AudioOutputTrack(MediaStreamTrack):
    """Outgoing audio track that sends pipeline audio chunks as Opus via RTP."""

    kind = "audio"

    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue = asyncio.Queue()
        self._resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)

    async def push_pcm(self, pcm: np.ndarray, rate: int = 24000) -> None:
        pcm_s16 = float32_to_int16(pcm)
        frame = AudioFrame.from_ndarray(pcm_s16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += len(pcm)
        await self._queue.put(frame)

    async def recv(self):
        frame = await self._queue.get()
        resampled = self._resampler.resample(frame)
        for f in resampled:
            return f
        return frame


class WebRTCTransport(AudioTransport):
    """AudioTransport implemented over WebRTC.

    Uses an RTP audio track for bidirectional audio and a data channel
    for JSON control messages. Signaling is handled externally via SDP exchange.
    """

    def __init__(self):
        self._session_id = uuid.uuid4().hex[:8]
        self._pc = RTCPeerConnection()
        self._dc: Optional[RTCDataChannel] = None
        self._audio_input: Optional[AudioInputTrack] = None
        self._audio_output: Optional[AudioOutputTrack] = None
        self._msg_queue: asyncio.Queue = asyncio.Queue()
        self._audio_buffer: list[np.ndarray] = []

        self._setup_handlers()

    def _setup_handlers(self):
        @self._pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            self._dc = channel

            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    self._msg_queue.put_nowait(data)
                except json.JSONDecodeError:
                    logger.warning(f"WebRTC invalid JSON: {message[:100]}")

        @self._pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                self._audio_input = AudioInputTrack(track)

                @track.on("ended")
                def on_ended():
                    self._msg_queue.put_nowait(None)

        @self._pc.on("connectionstatechange")
        async def on_connection_change():
            if self._pc.connectionState in ("failed", "closed", "disconnected"):
                await self._msg_queue.put(None)

    async def handle_offer(self, offer_sdp: str) -> dict:
        """Accept an SDP offer and return the answer + session_id."""
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

        # Add the output track BEFORE setRemoteDescription. Adding it inside
        # on_track (which fires during setRemoteDescription) creates a
        # transceiver that isn't in the offer, which makes createAnswer fail
        # with a real aiortc client ("None is not in list").
        self._audio_output = AudioOutputTrack()
        self._pc.addTrack(self._audio_output)

        await self._pc.setRemoteDescription(offer)

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        return {
            "session_id": self._session_id,
            "sdp": self._pc.localDescription.sdp,
            "type": "answer",
        }

    # ── AudioTransport interface ─────────────────────────────────

    async def recv_message(self) -> Optional[dict]:
        msg = await self._msg_queue.get()
        return msg

    async def send_event(self, event: ServerEvent) -> None:
        if self._dc is None or self._dc.readyState != "open":
            return

        if isinstance(event, AudioChunkEvent):
            if self._audio_output is not None:
                pcm = np.frombuffer(event.data, dtype=np.float32)
                await self._audio_output.push_pcm(pcm, event.sample_rate)
            return

        payload = self._event_to_dict(event)
        if payload:
            self._dc.send(json.dumps(payload))

    async def send_json(self, data: dict) -> None:
        if self._dc is not None and self._dc.readyState == "open":
            self._dc.send(json.dumps(data))

    async def close(self) -> None:
        if self._audio_input is not None:
            self._audio_input.stop()
        await self._pc.close()

    # ── Internal helpers ─────────────────────────────────────────

    def _event_to_dict(self, event: ServerEvent) -> Optional[dict]:
        from .events import (
            ResponseChunkEvent,
            ResponseCompleteEvent,
            SentenceEndEvent,
            SentenceStartEvent,
            TranscriptEvent,
        )

        if isinstance(event, TranscriptEvent):
            return {"type": "transcript", "text": event.text, "final": event.final}
        if isinstance(event, ResponseChunkEvent):
            return {"type": "response_chunk", "text": event.text}
        if isinstance(event, SentenceStartEvent):
            return {"type": "sentence_start", "seq": event.seq}
        if isinstance(event, SentenceEndEvent):
            return {"type": "sentence_end", "seq": event.seq}
        if isinstance(event, ResponseCompleteEvent):
            return {
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
        if isinstance(event, ErrorEvent):
            return {"type": "error", "message": event.message}
        return None


# Active WebRTC sessions keyed by session_id
_webrtc_sessions: dict[str, WebRTCTransport] = {}


def get_session(session_id: str) -> Optional[WebRTCTransport]:
    return _webrtc_sessions.get(session_id)


def register_session(transport: WebRTCTransport) -> str:
    _webrtc_sessions[transport._session_id] = transport
    return transport._session_id


def remove_session(session_id: str) -> None:
    _webrtc_sessions.pop(session_id, None)
