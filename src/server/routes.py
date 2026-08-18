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
import io
import wave
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from loguru import logger

from .auth import PRICING_TIERS, APIKey, token_manager, require_api_key, _validate_and_ratelimit
from .config import VOICES_DIR, settings
from .events import AudioChunkEvent
from .session import SessionContext
from .session_handler import (
    VoiceSessionState,
    build_vad_endpoint,
    check_rate_limit,
    handle_audio_samples,
    handle_session_message,
    make_session_context,
    max_audio_buffer_samples,
    reset_buffer,
)
from . import state as app_state
from .text_utils import sanitize_tts_symbols
from .transport import WebSocketTransport

# ── REST API session store ──────────────────────────────────────────
# Persists conversation history between /api/chat calls for iOS Shortcuts
_chat_sessions: Dict[str, dict] = {}  # session_id → {backend, agent, last_used}
_SESSION_TTL = 3600  # Sessions expire after 1 hour of inactivity
_FRESH_WINDOW = 300  # Reuse sessions used within 5 minutes


def _get_or_create_session(session_id: Optional[str], agent: str) -> tuple:
    """Get or create a backend session for the REST API.

    If no session_id is provided, automatically reuses the most recent
    session for the same agent if it was used within the fresh window.
    This gives iOS Shortcuts conversation continuity without managing IDs.
    """
    import asyncio
    from .backend import AIBackend

    # Clean expired sessions
    now = time.time()
    expired = [sid for sid, s in _chat_sessions.items() if now - s["last_used"] > _SESSION_TTL]
    for sid in expired:
        del _chat_sessions[sid]

    # Explicit session_id — use it
    if session_id and session_id in _chat_sessions:
        session = _chat_sessions[session_id]
        session["last_used"] = now
        return session["backend"], session_id

    # No session_id — find the freshest session for this agent
    if not session_id:
        freshest = None
        freshest_time = 0
        for sid, s in _chat_sessions.items():
            if s["agent"] == agent and s["last_used"] > freshest_time:
                freshest = sid
                freshest_time = s["last_used"]

        if freshest and (now - freshest_time) < _FRESH_WINDOW:
            session = _chat_sessions[freshest]
            session["last_used"] = now
            return session["backend"], freshest

    # Create new session
    new_id = session_id or f"rest_{secrets.token_hex(8)}"
    # Use the same gateway config as the main backend
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

    if gateway_url and gateway_token:
        # OpenClaw gateway mode — resolve model per-agent (issue #3)
        from .backend import resolve_openclaw_model

        model_id = resolve_openclaw_model(agent, settings.voice_model)
        backend = AIBackend(
            backend_type="openclaw",
            url=f"{gateway_url}/v1",
            model=model_id,
            voice_model_default=settings.voice_model,
            api_key=gateway_token,
            system_prompt="",
        )
    else:
        # Direct OpenAI mode
        backend = AIBackend(
            backend_type="openai",
            url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key=openai_key,
            system_prompt="",
        )
    _chat_sessions[new_id] = {"backend": backend, "agent": agent, "last_used": now}
    return backend, new_id


WAV_MAGIC_RIFF = b"RIFF"
WAV_MAGIC_WAVE = b"WAVE"


def _is_valid_wav(data: bytes) -> bool:
    """Cheap magic-byte check: RIFF....WAVE header."""
    return len(data) >= 12 and data[0:4] == WAV_MAGIC_RIFF and data[8:12] == WAV_MAGIC_WAVE


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


@router.get("/shortcut")
@router.get("/voice/shortcut")
async def shortcut_setup():
    """Serve the iOS Shortcut setup page."""
    response = FileResponse(
        Path(__file__).resolve().parent.parent / "client" / "shortcut-setup.html"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@router.get("/favicon.svg")
@router.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return FileResponse(Path(__file__).resolve().parent.parent / "client" / "favicon.svg")


@router.get("/audio-capture-processor.js")
@router.get("/voice/audio-capture-processor.js")
async def audio_capture_processor():
    """Serve the AudioWorklet capture processor module."""
    response = FileResponse(
        Path(__file__).resolve().parent.parent / "client" / "audio-capture-processor.js",
        media_type="application/javascript",
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.get("/health")
async def health():
    """Health check — used by monitoring and client auto-reconnect."""
    # RSS memory
    try:
        import psutil

        rss_mb = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        rss_mb = None

    memory_info = {"rss_mb": rss_mb}
    model_mem = getattr(app_state, "model_memory_mb", {})
    if model_mem:
        memory_info.update(model_mem)

    return JSONResponse(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - app_state._startup_time, 1)
            if app_state._startup_time
            else 0,
            "memory": memory_info,
            "stt": app_state.stt.status() if app_state.stt else {"backend": "not_loaded"},
            "tts": app_state.tts.status() if app_state.tts else {"backend": "not_loaded"},
            "tts_router": app_state.tts_router.status() if app_state.tts_router else None,
            "auth": {
                "enabled": settings.require_auth,
                "warning": "Authentication is DISABLED — /api/keys and WebSocket access are open. Set OPENCLAW_REQUIRE_AUTH=true in production."
                if not settings.require_auth
                else None,
            },
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


@router.get("/metrics")
async def metrics():
    if not settings.metrics_enabled:
        return JSONResponse(status_code=404, content={"error": "metrics disabled"})
    try:
        import psutil

        rss_bytes = psutil.Process().memory_info().rss
    except Exception:
        rss_bytes = 0

    lines = [
        "# HELP process_resident_memory_bytes Resident memory size in bytes.",
        "# TYPE process_resident_memory_bytes gauge",
        f"process_resident_memory_bytes {rss_bytes}",
    ]
    if app_state.stt:
        latency = app_state.stt.status().get("latency_ms")
        if latency is not None:
            lines += [
                "# HELP openclaw_stt_latency_ms Last STT call latency in milliseconds.",
                "# TYPE openclaw_stt_latency_ms gauge",
                f"openclaw_stt_latency_ms {latency}",
            ]
    if app_state.tts_router:
        # expose active backend as gauge 1/0
        be = app_state.tts_router.active_backend
        lines += [
            "# HELP openclaw_tts_active_backend 1 if backend is active",
            "# TYPE openclaw_tts_active_backend gauge",
            f'openclaw_tts_active_backend{{backend="{be}"}} 1',
        ]

    from fastapi.responses import Response

    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


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
        if not secrets.compare_digest(provided_key, settings.master_key or ""):
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
async def upload_voice(
    name: str = Body(...),
    file: bytes = Body(...),
    api_key: Optional[APIKey] = Depends(require_api_key),
):
    """Upload a voice sample for TTS cloning. Returns a voice ID."""
    if len(file) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty upload"})
    if len(file) > settings.max_voice_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"File too large ({len(file)} bytes, max {settings.max_voice_upload_bytes})"
            },
        )
    if not _is_valid_wav(file):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file: expected a WAV (RIFF/WAVE) audio file"},
        )

    VOICES_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", name)[:64] or "voice"
    voice_id = f"{safe_name}_{secrets.token_hex(4)}"
    path = VOICES_DIR / f"{voice_id}.wav"
    with open(path, "wb") as f:
        f.write(file)
    logger.info(f"Saved voice sample: {voice_id} ({len(file)} bytes)")
    return {"voice_id": voice_id, "path": str(path)}


@router.get("/api/voices")
async def list_voices(api_key: Optional[APIKey] = Depends(require_api_key)):
    """List available voice samples."""
    VOICES_DIR.mkdir(exist_ok=True)
    voices = []
    for f in sorted(VOICES_DIR.iterdir()):
        if f.suffix in (".wav", ".mp3", ".ogg"):
            voices.append(
                {"voice_id": f.stem, "name": f.stem.split("_")[0], "size": f.stat().st_size}
            )
    return {"voices": voices}


@router.post("/api/speak")
async def speak(
    text: str = Body(..., embed=True),
    agent: str = Body(default="metis"),
    voice: str = Body(default=None),
    format: str = Body(default="wav"),
    api_key: Optional[APIKey] = Depends(require_api_key),
):
    """REST TTS endpoint — text in, audio out. For iOS Shortcuts, scripting, etc.

    Request body:
        text: Text to synthesize
        agent: Agent name (metis, atlas, hephaestus, clio, deepthought, mara)
        voice: Override voice preset (optional)
        format: "wav" (default) or "pcm" (raw 16-bit/24kHz mono)

    Returns: audio file (WAV or raw PCM)
    """
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "text is required"})

    text = text.strip()

    # Apply agent voice personality control tokens if Higgs is available
    # For now, just pass text to Supertonic
    tts = app_state.tts
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "TTS not available"})

    try:
        # Sanitize text for TTS
        tts_text = sanitize_tts_symbols(text)

        # Synthesize the full text
        audio_chunks = []
        async for chunk in tts.synthesize_stream(tts_text):
            audio_chunks.append(chunk)

        if not audio_chunks:
            return JSONResponse(status_code=500, content={"error": "TTS produced no audio"})

        # Concatenate all chunks into a single buffer
        # Chunks are raw PCM bytes (int16 at 24kHz mono)
        pcm_bytes = b"".join(audio_chunks)

        if api_key is not None:
            minutes = len(pcm_bytes) / 2 / 24000 / 60
            if not token_manager.check_monthly_quota(api_key, minutes):
                return JSONResponse(status_code=402, content={"error": "Monthly quota exceeded"})
            token_manager.record_usage(api_key, minutes)

        if format == "pcm":
            # Raw 16-bit PCM at 24kHz mono
            return Response(
                content=pcm_bytes,
                media_type="audio/pcm",
                headers={
                    "Content-Disposition": f'attachment; filename="speech.pcm"',
                    "X-Sample-Rate": "24000",
                    "X-Channels": "1",
                    "X-Bits-Per-Sample": "16",
                },
            )
        else:
            # WAV format (16-bit PCM, 24kHz, mono)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)
                wf.writeframes(pcm_bytes)
            wav_bytes = buf.getvalue()

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="speech.wav"',
                },
            )

    except Exception as e:
        logger.error(f"TTS API error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/chat")
async def chat(
    text: str = Body(..., embed=True),
    agent: str = Body(default="metis"),
    session_id: str = Body(default=None),
    format: str = Body(default="wav"),
    api_key: Optional[APIKey] = Depends(require_api_key),
):
    """Full voice chat endpoint — text in, spoken response out.

    1. Sends text to OpenClaw gateway (LLM)
    2. Gets response text
    3. Synthesizes to speech
    4. Returns audio + session_id for continuity

    For iOS Shortcuts: send a question, get a spoken answer.
    Include session_id from previous response to continue the conversation.

    Request body:
        text: User message
        agent: Agent name
        session_id: (optional) Session ID from previous call for continuity
        format: "wav" or "pcm"
    """
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "text is required"})

    backend, sid = _get_or_create_session(session_id, agent)
    tts = app_state.tts
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "Voice pipeline not available"})

    try:
        # Get LLM response (using session backend for continuity)
        response_text = ""
        async for chunk in backend.chat_stream(text, agent_hint=agent, session_id=sid):
            response_text += chunk

        if not response_text.strip():
            return JSONResponse(status_code=500, content={"error": "LLM returned empty response"})

        # Sanitize text for TTS
        tts_text = sanitize_tts_symbols(response_text)

        # Synthesize
        audio_chunks = []
        async for chunk in tts.synthesize_stream(tts_text):
            audio_chunks.append(chunk)

        if not audio_chunks:
            return JSONResponse(status_code=500, content={"error": "TTS produced no audio"})

        pcm_bytes = b"".join(audio_chunks)

        if api_key is not None:
            minutes = len(pcm_bytes) / 2 / 24000 / 60
            if not token_manager.check_monthly_quota(api_key, minutes):
                return JSONResponse(status_code=402, content={"error": "Monthly quota exceeded"})
            token_manager.record_usage(api_key, minutes)

        if format == "pcm":
            return Response(
                content=pcm_bytes,
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Session-Id": sid,
                },
            )
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_bytes)
            wav_bytes = buf.getvalue()

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Session-Id": sid,
                },
            )

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/chat/json")
async def chat_json(
    text: str = Body(..., embed=True),
    agent: str = Body(default="metis"),
    session_id: str = Body(default=None),
    api_key: Optional[APIKey] = Depends(require_api_key),
):
    """Chat endpoint returning JSON with both audio (base64 WAV) and session_id.

    For iOS Shortcuts that need to extract the session_id for continuity.

    Request body:
        text: User message
        agent: Agent name
        session_id: (optional) Session ID from previous call

    Returns JSON: {session_id: str, audio_base64: str, response_text: str}
    """
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "text is required"})

    backend, sid = _get_or_create_session(session_id, agent)
    tts = app_state.tts
    if tts is None:
        return JSONResponse(status_code=503, content={"error": "Voice pipeline not available"})

    try:
        # Get LLM response
        response_text = ""
        async for chunk in backend.chat_stream(text, agent_hint=agent, session_id=sid):
            response_text += chunk

        if not response_text.strip():
            return JSONResponse(status_code=500, content={"error": "LLM returned empty response"})

        # Sanitize for TTS
        tts_text = sanitize_tts_symbols(response_text)

        # Synthesize
        audio_chunks = []
        async for chunk in tts.synthesize_stream(tts_text):
            audio_chunks.append(chunk)

        if not audio_chunks:
            return JSONResponse(status_code=500, content={"error": "TTS produced no audio"})

        pcm_bytes = b"".join(audio_chunks)

        if api_key is not None:
            minutes = len(pcm_bytes) / 2 / 24000 / 60
            if not token_manager.check_monthly_quota(api_key, minutes):
                return JSONResponse(status_code=402, content={"error": "Monthly quota exceeded"})
            token_manager.record_usage(api_key, minutes)

        # Encode as WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_bytes)
        wav_bytes = buf.getvalue()

        import base64

        return JSONResponse(
            content={
                "session_id": sid,
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "response_text": response_text,
            }
        )

    except Exception as e:
        logger.error(f"Chat JSON API error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.websocket("/ws")
@router.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    api_key_str = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")

    try:
        api_key: Optional[APIKey] = await _validate_and_ratelimit(api_key_str)
    except HTTPException as e:
        code = 4001 if e.status_code == 401 else 4003
        await websocket.close(code=code, reason=e.detail)
        return

    if api_key is not None:
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        logger.info("Client connected (auth disabled)")

    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"Failed to accept WebSocket: {e}")
        return

    transport = WebSocketTransport(websocket)
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    ws_session_id = secrets.token_hex(8)
    logger.info(f"WebSocket connected from {client_id} (session={ws_session_id})")

    # Server-side WebSocket ping to keep connection alive on mobile networks.
    # Carrier NAT drops idle TCP after 30-60s. Client pings every 15s,
    # we also send protocol-level pings every 20s.
    last_ping = asyncio.get_event_loop().time()

    async def keepalive():
        """Send periodic WebSocket pings to prevent carrier NAT timeout."""
        nonlocal last_ping
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_json({"type": "ping"})
                last_ping = asyncio.get_event_loop().time()
            except Exception:
                break

    keepalive_task = asyncio.create_task(keepalive())

    MAX_AUDIO_BUFFER_SAMPLES = max_audio_buffer_samples()
    state = VoiceSessionState(
        session_id=ws_session_id,
        log_tag=client_id,
    )

    async def _run_pipeline(buf: list[np.ndarray], session: SessionContext) -> None:
        """Run the voice pipeline in a background task."""
        state.is_playing = True
        try:
            if app_state.pipeline is not None:
                async for event in app_state.pipeline.process_audio(buf, session):
                    await transport.send_event(event)
        except asyncio.CancelledError:
            logger.debug("Pipeline cancelled (barge-in or disconnect)")
            try:
                await transport.send_json({"type": "interrupt"})
            except Exception:
                pass  # socket may already be closed (disconnect case)
            raise
        finally:
            state.is_playing = False

    async def _on_start_listening(s: VoiceSessionState) -> None:
        """WS-specific start_listening hook: barge-in reset + VAD setup."""
        if s.pipeline_task is not None and not s.pipeline_task.done():
            s.pipeline_task.cancel()
            s.pipeline_task = None
        s.vad_endpoint = build_vad_endpoint()

    async def _on_stop_listening(s: VoiceSessionState) -> None:
        """WS-specific stop_listening hook: clear VAD + rate-limit + dispatch."""
        s.vad_endpoint = None
        if not await check_rate_limit(transport, api_key):
            return
        session = make_session_context(s)
        s.pipeline_task = asyncio.create_task(_run_pipeline(s.audio_buffer, session))

    async def _dispatch_pipeline(s: VoiceSessionState) -> None:
        """Dispatch the current buffer to the pipeline as a cancelable task
        (shared cap-flush / VAD auto-stop / barge-in entry point)."""
        if not await check_rate_limit(transport, api_key):
            return
        session = make_session_context(s)
        s.pipeline_task = asyncio.create_task(_run_pipeline(s.audio_buffer, session))

    state.on_start_listening = _on_start_listening
    state.on_stop_listening = _on_stop_listening
    state.dispatch_pipeline = _dispatch_pipeline

    try:
        while True:
            msg = await transport.recv_message()
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "audio":
                try:
                    audio_bytes = base64.b64decode(msg["data"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    await handle_audio_samples(transport, state, audio_np, MAX_AUDIO_BUFFER_SAMPLES)
                except Exception as audio_err:
                    logger.warning(f"Audio decode error: {audio_err}")

            elif msg_type in (
                "start_listening",
                "stop_listening",
                "ping",
                "set_voice",
                "clear_history",
            ):
                await handle_session_message(transport, msg, state)

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await transport.close()
    finally:
        keepalive_task.cancel()
        if state.pipeline_task is not None and not state.pipeline_task.done():
            state.pipeline_task.cancel()
            try:
                await state.pipeline_task
            except asyncio.CancelledError:
                pass
            except Exception as cancel_err:
                logger.debug(f"pipeline_task cleanup error for {client_id}: {cancel_err}")
        logger.info(f"WebSocket disconnected: {client_id}")


# ── WebRTC signaling ─────────────────────────────────────────────

# Strong references to in-flight WebRTC session tasks. asyncio only keeps
# weak references to tasks, so a fire-and-forget task with no external
# reference can be GC'd mid-session (issue #38). The done-callback removes
# the entry once the session ends, so the set never grows unboundedly.
_webrtc_session_tasks: set[asyncio.Task] = set()


@router.post("/api/webrtc/offer")
async def webrtc_offer(body: dict, api_key: Optional[APIKey] = Depends(require_api_key)):
    """Accept a WebRTC SDP offer and return an SDP answer + session_id."""
    if not _webrtc_available:
        return JSONResponse(
            status_code=501,
            content={
                "error": "WebRTC not available — install aiortc: pip install openclaw-voice[webrtc]"
            },
        )

    transport = WebRTCTransport()
    result = await transport.handle_offer(body["sdp"])
    register_session(transport)

    # Start the session task immediately. We must NOT block on
    # wait_connected() here: the client can only open the data channel
    # (which is what wait_connected waits for) AFTER it receives this
    # answer — waiting would deadlock the SDP exchange.
    task = asyncio.create_task(_run_webrtc_session(transport, api_key))
    _webrtc_session_tasks.add(task)
    task.add_done_callback(_webrtc_session_tasks.discard)
    return result


async def _run_webrtc_session(
    transport: WebRTCTransport, api_key: Optional[APIKey] = None
) -> None:
    """Background handler for a WebRTC session — mirrors the WebSocket handler flow."""
    session_id = transport._session_id
    logger.info(f"WebRTC session started: {session_id}")

    MAX_AUDIO_BUFFER_SAMPLES = max_audio_buffer_samples()
    state = VoiceSessionState(
        session_id=session_id,
        log_prefix=f"[webrtc:{session_id}] ",
        report_missing_voice_error=False,
    )

    async def _run_pipeline(buf: list[np.ndarray]) -> None:
        """Run the voice pipeline in a background task."""
        state.is_playing = True
        try:
            if app_state.pipeline is not None:
                session = make_session_context(state)
                pcm_bytes_total = 0
                async for event in app_state.pipeline.process_audio(buf, session):
                    await transport.send_event(event)
                    if isinstance(event, AudioChunkEvent):
                        pcm_bytes_total += len(event.data)
                # Metered usage: deduct from the key's monthly quota after a
                # successful run (issue #36). No-op when auth is disabled
                # (api_key is None).
                if api_key is not None and pcm_bytes_total:
                    minutes = pcm_bytes_total / 2 / 24000 / 60
                    if not token_manager.check_monthly_quota(api_key, minutes):
                        await transport.send_json({"type": "error", "message": "quota_exceeded"})
                    else:
                        token_manager.record_usage(api_key, minutes)
        except asyncio.CancelledError:
            logger.debug(f"{state.log_prefix}Pipeline cancelled (barge-in or disconnect)")
            try:
                await transport.send_json({"type": "interrupt"})
            except Exception:
                pass
            raise
        finally:
            state.is_playing = False

    async def _collect_rtp_audio():
        """Background task: read PCM frames from WebRTC audio track.

        Feeds every frame through the shared ``handle_audio_samples`` so the
        WebRTC path gets the same VAD auto-endpointing and barge-in as the
        WS path (issue #22)."""
        input_track = transport._audio_input
        if input_track is None:
            return
        try:
            while True:
                frame = await input_track.read_frame()
                if frame is None:
                    if not state.is_listening and not state.is_playing:
                        await asyncio.sleep(0.05)
                        continue
                    continue
                await handle_audio_samples(
                    transport, state, frame, MAX_AUDIO_BUFFER_SAMPLES, log_prefix=state.log_prefix
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A dead collector would silently freeze the session's audio
            # input (issue #39) — log it and tell the client.
            logger.error(f"{state.log_prefix}RTP audio collector failed: {e}")
            try:
                await transport.send_json({"type": "error", "message": "audio_input_failed"})
            except Exception:
                pass

    async def _dispatch_pipeline(s: VoiceSessionState) -> None:
        """Dispatch the current buffer to the pipeline as a cancelable task —
        the single choke point for VAD auto-stop, buffer-cap overflow, and
        explicit stop_listening on this transport (issue #36)."""
        if not await check_rate_limit(transport, api_key):
            reset_buffer(s)
            return
        s.pipeline_task = asyncio.create_task(_run_pipeline(s.audio_buffer))

    async def _on_start_listening(s: VoiceSessionState) -> None:
        """WebRTC-specific start_listening hook: start RTP collector + VAD."""
        if transport._audio_input is not None and (
            s.rtp_collector_task is None or s.rtp_collector_task.done()
        ):
            s.rtp_collector_task = asyncio.create_task(_collect_rtp_audio())
        s.vad_endpoint = build_vad_endpoint()

    async def _on_stop_listening(s: VoiceSessionState) -> None:
        """WebRTC-specific stop_listening hook: dispatch buffered audio.

        The RTP collector is intentionally NOT cancelled here — it keeps
        running so barge-in can detect speech during playback (issue #22).
        The collector already idles (sleeps) when neither listening nor
        playing, and is torn down at session end."""
        await _dispatch_pipeline(s)

    state.on_start_listening = _on_start_listening
    state.on_stop_listening = _on_stop_listening
    state.dispatch_pipeline = _dispatch_pipeline

    try:
        while True:
            msg = await transport.recv_message()
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type in ("start_listening", "stop_listening"):
                await handle_session_message(transport, msg, state)

            elif msg_type == "audio_frame":
                try:
                    audio_np = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.float32)
                except Exception:
                    continue
                await handle_audio_samples(
                    transport,
                    state,
                    audio_np,
                    MAX_AUDIO_BUFFER_SAMPLES,
                    log_prefix=state.log_prefix,
                )

            elif msg_type in ("ping", "set_voice", "clear_history"):
                await handle_session_message(transport, msg, state)

    except Exception as e:
        logger.error(f"WebRTC session error ({session_id}): {e}")
    finally:
        if state.pipeline_task is not None and not state.pipeline_task.done():
            state.pipeline_task.cancel()
            try:
                await state.pipeline_task
            except asyncio.CancelledError:
                pass
            except Exception as cancel_err:
                logger.debug(f"pipeline_task cleanup error ({session_id}): {cancel_err}")
        remove_session(session_id)
        await transport.close()
        logger.info(f"WebRTC session ended: {session_id}")
