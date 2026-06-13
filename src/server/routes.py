"""
HTTP and WebSocket route handlers.

All routes are registered on an APIRouter that main.py includes into the app.
"""

import asyncio
import base64
import os
import re
import secrets
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Body, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from .auth import PRICING_TIERS, APIKey, token_manager
from .config import VOICES_DIR, settings
from .session import SessionContext
from . import state as app_state
from .transport import WebSocketTransport
from .tts import ChatterboxTTS

_tts_lock = asyncio.Lock()

try:
    from .webrtc import WebRTCTransport, register_session, remove_session

    _webrtc_available = True
except ImportError:
    _webrtc_available = False
    WebRTCTransport = None  # type: ignore

router = APIRouter()


@router.get("/")
@router.get("/voice")
@router.get("/voice/")
async def index():
    """Serve the demo page."""
    response = FileResponse(Path(__file__).resolve().parent.parent / "client" / "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.get("/health")
async def health():
    """Health check — used by monitoring and client auto-reconnect."""
    return JSONResponse(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - app_state._startup_time, 1) if app_state._startup_time else 0,
            "stt": app_state.stt.status() if app_state.stt else {"backend": "not_loaded"},
            "tts": app_state.tts.status() if app_state.tts else {"backend": "not_loaded"},
            "backend": app_state.backend.backend_type if app_state.backend else "not_loaded",
            "vad": "loaded" if app_state.vad else "not_loaded",
            "config": {
                "stt_model": settings.stt_model,
                "tts_model": settings.tts_model,
                "supertonic_model": os.getenv("SUPERTONIC_MODEL", "supertonic-2"),
                "supertonic_voice": os.getenv("SUPERTONIC_VOICE", "F2"),
                "voice_model": settings.voice_model,
            },
        }
    )


@router.post("/api/keys")
async def create_api_key(
    name: str = Body(..., description="Human-readable name for this key"),
    tier: str = Body("free", description="Billing tier: free, pro, enterprise"),
    master_key: Optional[str] = Body(None, description="Master key for authentication"),
):
    """Create a new API key (requires master key)."""
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}

    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    tier_config = PRICING_TIERS[tier]
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    return {
        "api_key": plaintext_key,
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@router.get("/api/usage")
async def get_usage(api_key: str):
    """Get usage stats for an API key."""
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    return token_manager.get_usage(key)


@router.post("/api/voices")
async def upload_voice(name: str = Body(...), file: bytes = Body(...)):
    """Upload a voice sample for TTS cloning. Returns a voice ID."""
    VOICES_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", name)[:64] or "voice"
    voice_id = f"{safe_name}_{secrets.token_hex(4)}"
    path = VOICES_DIR / f"{voice_id}.wav"
    with open(path, "wb") as f:
        f.write(file)
    logger.info(f"Saved voice sample: {voice_id} ({len(file)} bytes)")
    return {"voice_id": voice_id, "path": str(path)}


@router.get("/api/voices")
async def list_voices():
    """List available voice samples."""
    VOICES_DIR.mkdir(exist_ok=True)
    voices = []
    for f in sorted(VOICES_DIR.iterdir()):
        if f.suffix in (".wav", ".mp3", ".ogg"):
            voices.append(
                {"voice_id": f.stem, "name": f.stem.split("_")[0], "size": f.stat().st_size}
            )
    return {"voices": voices}


@router.websocket("/ws")
@router.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    api_key_str = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    api_key: Optional[APIKey] = None

    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        if not await token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")

    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"Failed to accept WebSocket: {e}")
        return

    transport = WebSocketTransport(websocket)
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connected from {client_id}")

    audio_buffer: list[np.ndarray] = []
    MAX_AUDIO_BUFFER_SECONDS = 30
    MAX_AUDIO_BUFFER_SAMPLES = settings.sample_rate * MAX_AUDIO_BUFFER_SECONDS
    is_listening = False
    session_agent: Optional[str] = None

    try:
        while True:
            msg = await transport.recv_message()
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "start_listening":
                is_listening = True
                audio_buffer = []
                if "agent" in msg:
                    session_agent = msg["agent"]
                    logger.info(f"Agent selected: {session_agent}")
                    async with _tts_lock:
                        if (
                            session_agent in ChatterboxTTS.AGENT_VOICE_MAP
                            and app_state.tts is not None
                            and app_state.tts._backend == "supertonic"
                        ):
                            new_voice = ChatterboxTTS.AGENT_VOICE_MAP[session_agent]
                            if new_voice != app_state.tts._supertonic_voice:
                                try:
                                    app_state.tts._supertonic_style = app_state.tts._supertonic_tts.get_voice_style(
                                        new_voice
                                    )
                                    app_state.tts._supertonic_voice = new_voice
                                    logger.info(
                                        f"Switched TTS voice to {new_voice} for agent {session_agent}"
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to switch voice to {new_voice}: {e}")
                await transport.send_json({"type": "listening_started"})

            elif msg_type == "stop_listening":
                is_listening = False
                session = SessionContext(agent_id=session_agent)
                if app_state.pipeline is not None:
                    async for event in app_state.pipeline.process_audio(audio_buffer, session):
                        await transport.send_event(event)
                audio_buffer = []
                await transport.send_json({"type": "listening_stopped"})

            elif msg_type == "audio" and is_listening:
                try:
                    audio_bytes = base64.b64decode(msg["data"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

                    total_samples = sum(len(chunk) for chunk in audio_buffer) + len(audio_np)
                    if total_samples > MAX_AUDIO_BUFFER_SAMPLES:
                        logger.warning(
                            f"Audio buffer cap reached ({total_samples} samples), processing now"
                        )
                        audio_buffer.append(audio_np)
                        is_listening = False
                        session = SessionContext(agent_id=session_agent)
                        if app_state.pipeline is not None:
                            async for event in app_state.pipeline.process_audio(audio_buffer, session):
                                await transport.send_event(event)
                        audio_buffer = []
                    else:
                        audio_buffer.append(audio_np)

                    if app_state.vad is not None and len(audio_np) > 0:
                        has_speech = app_state.vad.is_speech(audio_np)
                        await transport.send_json(
                            {
                                "type": "vad_status",
                                "speech_detected": has_speech,
                            }
                        )
                except Exception as audio_err:
                    logger.warning(f"Audio decode error: {audio_err}")

            elif msg_type == "ping":
                await transport.send_json({"type": "pong"})

            elif msg_type == "set_voice":
                raw_voice_id = msg.get("voice_id", "")
                logger.info(f"Switching voice to {raw_voice_id} for {client_id}")
                if raw_voice_id and raw_voice_id != "default":
                    safe_voice_id = re.sub(r"[^a-zA-Z0-9_-]", "", raw_voice_id)
                    voice_path = str(VOICES_DIR / f"{safe_voice_id}.wav")
                    if os.path.isfile(voice_path):
                        if app_state.tts is not None:
                            app_state.tts.voice_sample = voice_path
                        await transport.send_json({"type": "voice_set", "voice_id": raw_voice_id})
                    else:
                        await transport.send_json(
                            {"type": "error", "message": f"Voice {raw_voice_id} not found"}
                        )
                else:
                    if app_state.tts is not None:
                        app_state.tts.voice_sample = None
                    await transport.send_json({"type": "voice_set", "voice_id": "default"})

            elif msg_type == "clear_history":
                if app_state.backend is not None:
                    app_state.backend.clear_history()
                logger.info(f"History cleared for {client_id}")
                await transport.send_json({"type": "history_cleared"})

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await transport.close()


# ── WebRTC signaling ─────────────────────────────────────────────

@router.post("/api/webrtc/offer")
async def webrtc_offer(body: dict):
    """Accept a WebRTC SDP offer and return an SDP answer + session_id."""
    if not _webrtc_available:
        return JSONResponse(
            status_code=501,
            content={"error": "WebRTC not available — install aiortc: pip install openclaw-voice[webrtc]"},
        )

    transport = WebRTCTransport()
    result = await transport.handle_offer(body["sdp"])
    register_session(transport)

    connected = await transport.wait_connected(timeout=15.0)
    if not connected:
        remove_session(transport._session_id)
        await transport.close()
        return JSONResponse(
            status_code=504,
            content={"error": "WebRTC connection timed out"},
        )

    asyncio.create_task(_run_webrtc_session(transport))
    return result


async def _run_webrtc_session(transport: WebRTCTransport) -> None:
    """Background handler for a WebRTC session — mirrors the WebSocket handler flow."""
    session_id = transport._session_id
    logger.info(f"WebRTC session started: {session_id}")

    audio_buffer: list[np.ndarray] = []
    MAX_AUDIO_BUFFER_SECONDS = 30
    MAX_AUDIO_BUFFER_SAMPLES = settings.sample_rate * MAX_AUDIO_BUFFER_SECONDS
    is_listening = False
    session_agent: Optional[str] = None

    try:
        while True:
            msg = await transport.recv_message()
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "start_listening":
                is_listening = True
                audio_buffer = []
                if "agent" in msg:
                    session_agent = msg["agent"]
                    logger.info(f"[webrtc:{session_id}] Agent selected: {session_agent}")
                    async with _tts_lock:
                        if (
                            session_agent in ChatterboxTTS.AGENT_VOICE_MAP
                            and app_state.tts is not None
                            and app_state.tts._backend == "supertonic"
                        ):
                            new_voice = ChatterboxTTS.AGENT_VOICE_MAP[session_agent]
                            if new_voice != app_state.tts._supertonic_voice:
                                try:
                                    app_state.tts._supertonic_style = (
                                        app_state.tts._supertonic_tts.get_voice_style(new_voice)
                                    )
                                    app_state.tts._supertonic_voice = new_voice
                                except Exception as e:
                                    logger.warning(f"Failed to switch voice: {e}")
                await transport.send_json({"type": "listening_started"})

            elif msg_type == "stop_listening":
                is_listening = False
                session = SessionContext(agent_id=session_agent)
                if app_state.pipeline is not None:
                    async for event in app_state.pipeline.process_audio(audio_buffer, session):
                        await transport.send_event(event)
                audio_buffer = []
                await transport.send_json({"type": "listening_stopped"})

            elif msg_type == "audio_frame" and is_listening:
                # Audio frames may arrive via data channel in non-RTP mode
                try:
                    audio_np = np.frombuffer(
                        base64.b64decode(msg["data"]), dtype=np.float32
                    )
                except Exception:
                    continue

                total = sum(len(c) for c in audio_buffer) + len(audio_np)
                if total > MAX_AUDIO_BUFFER_SAMPLES:
                    logger.warning(
                        f"[webrtc:{session_id}] Audio buffer cap, processing now"
                    )
                    audio_buffer.append(audio_np)
                    is_listening = False
                    session = SessionContext(agent_id=session_agent)
                    if app_state.pipeline is not None:
                        async for event in app_state.pipeline.process_audio(
                            audio_buffer, session
                        ):
                            await transport.send_event(event)
                    audio_buffer = []
                else:
                    audio_buffer.append(audio_np)

            elif msg_type == "ping":
                await transport.send_json({"type": "pong"})

            elif msg_type == "set_voice":
                raw_voice_id = msg.get("voice_id", "")
                if raw_voice_id and raw_voice_id != "default":
                    safe_voice_id = re.sub(r"[^a-zA-Z0-9_-]", "", raw_voice_id)
                    voice_path = str(VOICES_DIR / f"{safe_voice_id}.wav")
                    if os.path.isfile(voice_path):
                        if app_state.tts is not None:
                            app_state.tts.voice_sample = voice_path
                        await transport.send_json(
                            {"type": "voice_set", "voice_id": raw_voice_id}
                        )
                else:
                    if app_state.tts is not None:
                        app_state.tts.voice_sample = None
                    await transport.send_json({"type": "voice_set", "voice_id": "default"})

            elif msg_type == "clear_history":
                if app_state.backend is not None:
                    app_state.backend.clear_history()
                await transport.send_json({"type": "history_cleared"})

    except Exception as e:
        logger.error(f"WebRTC session error ({session_id}): {e}")
    finally:
        remove_session(session_id)
        await transport.close()
        logger.info(f"WebRTC session ended: {session_id}")
