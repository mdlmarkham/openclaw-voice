"""
Shared session message dispatch for WebSocket and WebRTC voice sessions.

Extracts the duplicated message-handling logic between ``websocket_endpoint``
and ``_run_webrtc_session`` in ``routes.py`` (issue #21), and extends the
shared audio path with VAD auto-endpointing + barge-in for BOTH transports
(issue #22).  Transport-specific glue (WS keepalive, WebRTC RTP collector
wiring) stays in the callers via hooks.
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np
from loguru import logger

from .config import VOICES_DIR, settings
from .session import SessionContext
from .transport import AudioTransport
from .vad import VADEndpoint
from . import state as app_state

MAX_AUDIO_BUFFER_SECONDS = 30


async def check_rate_limit(transport: AudioTransport, api_key) -> bool:
    """Gate a pipeline dispatch on the key's per-minute rate limit.

    Returns True if the dispatch is allowed. When rate-limited, sends the
    ``rate_limited`` error to the client and returns False. No-op (always
    True) when auth is disabled (``api_key`` is None) — shared by the WS and
    WebRTC dispatch paths (issue #36).
    """
    from .auth import token_manager  # local import avoids an import cycle

    if api_key is not None and not await token_manager.check_rate_limit(api_key):
        await transport.send_json({"type": "error", "message": "rate_limited"})
        return False
    return True


@dataclass
class VoiceSessionState:
    """Mutable state for a single voice session.

    Replaces the pile of ``nonlocal`` variables that each handler closure
    previously mutated.  Fields used only by one transport are left at their
    defaults by the other.
    """

    # Shared
    audio_buffer: list[np.ndarray] = field(default_factory=list)
    buffer_samples: int = 0
    is_listening: bool = False
    session_agent: Optional[str] = None
    session_reconnected: bool = False
    session_voice_override: Optional[str] = None
    session_id: Optional[str] = None
    log_prefix: str = ""
    log_tag: Optional[str] = None

    # WS-specific
    is_playing: bool = False
    pipeline_task: Optional[asyncio.Task] = None
    vad_endpoint: Optional[VADEndpoint] = None
    barge_in_vad: Optional[VADEndpoint] = None

    # WebRTC-specific
    rtp_collector_task: Optional[asyncio.Task] = None

    # Behavior flags
    # WebRTC's original set_voice omitted the "voice not found" error; the
    # WS path sends it.  Preserve both exactly (zero-behavior-change dedup).
    report_missing_voice_error: bool = True

    # Transport-specific hooks (called at the right point in start/stop)
    on_start_listening: Optional[Callable[["VoiceSessionState"], Awaitable[None]]] = None
    on_stop_listening: Optional[Callable[["VoiceSessionState"], Awaitable[None]]] = None
    # Dispatch the buffered audio through the pipeline as a cancelable task.
    # The hook owns any rate-limiting (WS) and stores the task on
    # ``state.pipeline_task`` so a later barge-in can cancel it (issue #22).
    dispatch_pipeline: Optional[Callable[["VoiceSessionState"], Awaitable[None]]] = None


def max_audio_buffer_samples() -> int:
    """Sample cap before the buffer is force-flushed."""
    return settings.sample_rate * MAX_AUDIO_BUFFER_SECONDS


def make_session_context(state: VoiceSessionState) -> SessionContext:
    """Build a SessionContext from the shared session state."""
    return SessionContext(
        agent_id=state.session_agent,
        reconnect=state.session_reconnected,
        voice_id=state.session_voice_override,
        session_id=state.session_id,
    )


def reset_buffer(state: VoiceSessionState) -> None:
    """Rebind the buffer list (NOT ``.clear()``) so any in-flight pipeline
    task that captured the old list retains its data."""
    state.audio_buffer = []
    state.buffer_samples = 0


def append_audio_samples(state: VoiceSessionState, audio: np.ndarray, max_samples: int) -> bool:
    """Append audio to the shared buffer, enforcing the cap.

    Returns True if the cap was hit (the caller must dispatch the pipeline
    and reset the buffer).  ``audio`` is always appended, matching the
    original handlers' behavior on the cap-hitting frame.
    """
    state.buffer_samples += len(audio)
    state.audio_buffer.append(audio)
    if state.buffer_samples > max_samples:
        state.is_listening = False
        return True
    return False


def build_vad_endpoint() -> Optional[VADEndpoint]:
    """Construct a VADEndpoint from current settings, or None if VAD is off.

    Mirrors the ``start_listening`` VAD construction in the original
    WebSocket handler (both min_silence_frames and min_speech_frames
    computed from settings).
    """
    if not settings.vad_enabled or app_state.vad is None:
        return None
    return VADEndpoint(
        app_state.vad,
        threshold=settings.vad_threshold,
        min_silence_frames=max(
            1,
            settings.vad_silence_duration_ms
            * settings.sample_rate
            // settings.vad_frame_size
            // 1000,
        ),
        min_speech_frames=max(
            1,
            settings.vad_min_speech_duration_ms
            * settings.sample_rate
            // settings.vad_frame_size
            // 1000,
        ),
        sample_rate=settings.sample_rate,
    )


def build_barge_in_vad() -> Optional[VADEndpoint]:
    """Construct a lightweight VADEndpoint for barge-in detection.

    Detects speech quickly (min_speech_frames=1) so playback is interrupted
    as soon as the user starts talking.
    """
    if not settings.vad_enabled or app_state.vad is None:
        return None
    return VADEndpoint(
        app_state.vad,
        threshold=settings.vad_threshold,
        min_speech_frames=1,
        sample_rate=settings.sample_rate,
    )


async def handle_audio_samples(
    transport: AudioTransport,
    state: VoiceSessionState,
    audio_np: np.ndarray,
    max_samples: int,
    *,
    log_prefix: str = "",
) -> None:
    """Process one decoded audio chunk for BOTH transports (issue #22).

    Combines what the WS handler already did (buffer append + cap, VAD
    auto-endpointing, barge-in) into a single helper so WebRTC gets the same
    behavior.  The caller owns base64 decode; this operates on an
    already-decoded ``np.ndarray``.

    - Buffers audio, force-flushing (via ``state.dispatch_pipeline``) when
      the 30s cap is hit.
    - Runs VAD endpointing to auto-stop on ``speech_end`` — no explicit
      ``stop_listening`` needed.
    - Detects barge-in while playing (``state.is_playing``), cancels the
      in-flight pipeline task, and resumes listening.
    - Emits ``vad_status`` for client visual feedback.
    """
    if len(audio_np) == 0:
        return

    if state.is_listening:
        if append_audio_samples(state, audio_np, max_samples):
            logger.warning(f"{log_prefix}Audio buffer cap reached, processing now")
            state.vad_endpoint = None
            if state.dispatch_pipeline is not None:
                await state.dispatch_pipeline(state)
            reset_buffer(state)

        if state.vad_endpoint is not None:
            event = await state.vad_endpoint.process_async(audio_np)
            if event == "speech_end":
                logger.debug("VAD endpoint: speech ended, processing buffer")
                state.vad_endpoint = None
                if state.dispatch_pipeline is not None:
                    await state.dispatch_pipeline(state)
                reset_buffer(state)

        if app_state.vad is not None:
            has_speech = await app_state.vad.is_speech_async(audio_np)
            await transport.send_json({"type": "vad_status", "speech_detected": has_speech})

    elif state.is_playing:
        if state.barge_in_vad is None:
            state.barge_in_vad = build_barge_in_vad()
        if state.barge_in_vad is not None:
            event = await state.barge_in_vad.process_async(audio_np)
            if event == "speech_start":
                logger.info("Barge-in: user started speaking during playback")
                if state.pipeline_task is not None and not state.pipeline_task.done():
                    state.pipeline_task.cancel()
                    state.pipeline_task = None
                state.barge_in_vad = None
                state.is_listening = True
                reset_buffer(state)
                state.vad_endpoint = build_vad_endpoint()
                await transport.send_json({"type": "listening_started"})


async def handle_session_message(
    transport: AudioTransport,
    msg: dict,
    state: VoiceSessionState,
) -> bool:
    """Dispatch common session messages shared by WS and WebRTC.

    Handles ``start_listening``/``stop_listening`` (common parts + transport
    hooks) and ``set_voice``/``clear_history``/``ping`` (fully identical).

    Returns ``True`` if the message was recognized and handled.
    Audio messages (``audio``/``audio_frame``) are left to the caller.
    """
    msg_type = msg.get("type")

    if msg_type == "start_listening":
        if state.on_start_listening is not None:
            await state.on_start_listening(state)
        state.is_listening = True
        reset_buffer(state)
        if "agent" in msg:
            state.session_agent = msg["agent"]
            logger.info(f"{state.log_prefix}Agent selected: {state.session_agent}")
        state.session_reconnected = msg.get("reconnect", False)
        await transport.send_json({"type": "listening_started"})
        return True

    if msg_type == "stop_listening":
        state.is_listening = False
        if state.on_stop_listening is not None:
            await state.on_stop_listening(state)
        reset_buffer(state)
        await transport.send_json({"type": "listening_stopped"})
        return True

    if msg_type == "ping":
        await transport.send_json({"type": "pong"})
        return True

    if msg_type == "set_voice":
        raw_voice_id = msg.get("voice_id", "")
        if state.log_tag:
            logger.info(f"Switching voice to {raw_voice_id} for {state.log_tag}")
        if raw_voice_id and raw_voice_id != "default":
            safe_voice_id = re.sub(r"[^a-zA-Z0-9_-]", "", raw_voice_id)
            voice_path = str(VOICES_DIR / f"{safe_voice_id}.wav")
            if os.path.isfile(voice_path):
                state.session_voice_override = voice_path
                await transport.send_json({"type": "voice_set", "voice_id": raw_voice_id})
            elif state.report_missing_voice_error:
                await transport.send_json(
                    {"type": "error", "message": f"Voice {raw_voice_id} not found"}
                )
        else:
            state.session_voice_override = None
            await transport.send_json({"type": "voice_set", "voice_id": "default"})
        return True

    if msg_type == "clear_history":
        if app_state.backend is not None:
            app_state.backend.clear_history(session_id=state.session_id)
        if state.log_tag:
            logger.info(f"History cleared for {state.log_tag} (session={state.session_id})")
        await transport.send_json({"type": "history_cleared"})
        return True

    return False
