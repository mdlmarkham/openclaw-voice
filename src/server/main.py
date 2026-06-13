"""
OpenClaw Voice Server — Olympus Optimized

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication (OpenClaw gateway)
- Text-to-Speech via Edge TTS (or ElevenLabs)
- Audio streaming back to browser
- Health check, auto-reconnect support, graceful error handling
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech
from .constants import SYSTEM_PROMPT


class Settings(BaseSettings):
    """Server configuration."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8765

    # Auth
    require_auth: bool = False
    master_key: Optional[str] = None

    # STT
    stt_model: str = "base"
    stt_device: str = "auto"

    # TTS
    tts_model: str = "edge"
    tts_voice: Optional[str] = None

    # AI Backend
    backend_type: str = "openclaw"
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None

    # OpenClaw Gateway
    openclaw_gateway_url: Optional[str] = None
    openclaw_gateway_token: Optional[str] = None

    # Audio
    sample_rate: int = 16000

    # System prompt for voice mode — keep it short and conversational
    system_prompt: str = SYSTEM_PROMPT

    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"
        extra = "ignore"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.2.0-olympus")

# CORS for cross-origin WebSocket
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None
_startup_time: float = 0.0


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad, _startup_time

    _startup_time = time.time()
    logger.info("Initializing OpenClaw Voice server (Olympus)...")

    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")

    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(model_name=settings.stt_model, device=settings.stt_device)
    if stt._backend == "mock":
        logger.warning("⚠️ STT is in MOCK mode — install faster-whisper or openai-whisper for real speech recognition")

    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = ChatterboxTTS(voice_sample=settings.tts_voice)
    if tts._backend == "mock":
        logger.error("❌ No TTS backend available — set ELEVENLABS_API_KEY or install a local TTS backend")
    elif tts._backend == "elevenlabs":
        pass
    else:
        logger.warning(f"⚠️ TTS using fallback backend: {tts._backend}")

    # Auto-detect OpenClaw gateway
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    backend_configured = bool(gateway_url and gateway_token) or bool(openai_key)

    if not backend_configured:
        logger.error("❌ No AI backend configured — set OPENAI_API_KEY or OPENCLAW_GATEWAY_URL+OPENCLAW_GATEWAY_TOKEN")

    if gateway_url and gateway_token:
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url}")
        backend = AIBackend(
            backend_type="openclaw",
            url=f"{gateway_url}/v1",
            model="openclaw/metis",
            api_key=gateway_token,
            system_prompt=settings.system_prompt,
        )
    else:
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=openai_key,
            system_prompt=settings.system_prompt,
        )

    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()

    logger.info("✅ OpenClaw Voice server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse(Path(__file__).parent.parent / "client" / "index.html")


@app.get("/health")
async def health():
    """Health check — used by monitoring and client auto-reconnect."""
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": round(time.time() - _startup_time, 1) if _startup_time else 0,
        "stt": stt._backend if stt else "not_loaded",
        "tts": tts._backend if tts else "not_loaded",
        "tts_detail": {
            "backend": tts._backend if tts else "not_loaded",
            "model": tts._supertonic_model if tts and tts._backend == "supertonic" else None,
            "voice": tts._supertonic_voice if tts and tts._backend == "supertonic" else None,
            "sample_rate": tts._supertonic_sr if tts and tts._backend == "supertonic" else (24000 if tts else 0),
            "edge_voice": tts._edge_voice if tts and tts._backend == "edge" else None,
        },
        "backend": backend.backend_type if backend else "not_loaded",
        "vad": "loaded" if vad else "not_loaded",
        "config": {
            "stt_model": settings.stt_model,
            "tts_model": settings.tts_model,
            "supertonic_model": settings._supertonic_model if hasattr(settings, '_supertonic_model') else os.getenv("SUPERTONIC_MODEL", "supertonic-2"),
            "supertonic_voice": settings._supertonic_voice if hasattr(settings, '_supertonic_voice') else os.getenv("SUPERTONIC_VOICE", "F2"),
        },
    })


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
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

    from .auth import PRICING_TIERS
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    tier_config = PRICING_TIERS[tier]
    plaintext_key, api_key = token_manager.generate_key(
        name=name, tier=tier,
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


@app.get("/api/usage")
async def get_usage(api_key: str):
    """Get usage stats for an API key."""
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    return token_manager.get_usage(key)


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections with improved error recovery."""
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    api_key: Optional[APIKey] = None

    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        if not token_manager.check_rate_limit(api_key):
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

    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connected from {client_id}")

    audio_buffer = []
    is_listening = False
    session_agent = None

    try:
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {client_id}: {data[:100]}")
                continue

            msg_type = msg.get("type")

            if msg_type == "start_listening":
                is_listening = True
                audio_buffer = []
                if "agent" in msg:
                    session_agent = msg["agent"]
                    logger.info(f"Agent selected: {session_agent}")
                await websocket.send_json({"type": "listening_started"})

            elif msg_type == "stop_listening":
                is_listening = False

                if audio_buffer:
                    try:
                        audio_data = np.concatenate(audio_buffer)
                        logger.debug("Transcribing audio...")
                        transcript = await stt.transcribe(audio_data)

                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "final": True,
                        })
                        logger.info(f"Transcript: {transcript}")

                        if transcript.strip():
                            logger.debug("Streaming AI response...")
                            full_response = ""
                            sentence_buffer = ""
                            agent_model = f"openclaw/{session_agent}" if session_agent else None

                            audio_seq = 0  # Sequence counter for ordered playback

                            async for chunk in backend.chat_stream(transcript, model=agent_model):
                                full_response += chunk
                                sentence_buffer += chunk

                                await websocket.send_json({
                                    "type": "response_chunk",
                                    "text": chunk,
                                })

                                # Synthesize completed sentences
                                while any(sep in sentence_buffer for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                                    earliest_idx = len(sentence_buffer)
                                    for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                                        idx = sentence_buffer.find(sep)
                                        if idx != -1 and idx < earliest_idx:
                                            earliest_idx = idx + len(sep)

                                    if earliest_idx < len(sentence_buffer):
                                        sentence = sentence_buffer[:earliest_idx].strip()
                                        sentence_buffer = sentence_buffer[earliest_idx:]

                                        if sentence:
                                            speech_text = clean_for_speech(sentence)
                                            if speech_text:
                                                logger.debug(f"Synthesizing: {speech_text[:50]}...")
                                                # Mark sentence boundary for gapless scheduling
                                                await websocket.send_json({
                                                    "type": "sentence_start",
                                                    "seq": audio_seq,
                                                })
                                                try:
                                                    async for audio_chunk in tts.synthesize_stream(speech_text):
                                                        audio_b64 = base64.b64encode(audio_chunk).decode()
                                                        await websocket.send_json({
                                                            "type": "audio_chunk",
                                                            "data": audio_b64,
                                                            "sample_rate": 24000,
                                                            "seq": audio_seq,
                                                        })
                                                except Exception as tts_err:
                                                    logger.error(f"TTS error: {tts_err}")
                                                await websocket.send_json({
                                                    "type": "sentence_end",
                                                    "seq": audio_seq,
                                                })
                                                audio_seq += 1
                                    else:
                                        break

                            # Handle remaining text
                            if sentence_buffer.strip():
                                speech_text = clean_for_speech(sentence_buffer.strip())
                                if speech_text:
                                    await websocket.send_json({
                                        "type": "sentence_start",
                                        "seq": audio_seq,
                                    })
                                    try:
                                        async for audio_chunk in tts.synthesize_stream(speech_text):
                                            audio_b64 = base64.b64encode(audio_chunk).decode()
                                            await websocket.send_json({
                                                "type": "audio_chunk",
                                                "data": audio_b64,
                                                "sample_rate": 24000,
                                                "seq": audio_seq,
                                            })
                                    except Exception as tts_err:
                                        logger.error(f"TTS error (final): {tts_err}")
                                    await websocket.send_json({
                                        "type": "sentence_end",
                                        "seq": audio_seq,
                                    })

                            await websocket.send_json({
                                "type": "response_complete",
                                "text": full_response,
                            })
                            logger.info(f"Response complete: {full_response[:100]}...")

                    except Exception as inner_err:
                        logger.error(f"Processing error: {inner_err}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(inner_err),
                        })

                audio_buffer = []
                await websocket.send_json({"type": "listening_stopped"})

            elif msg_type == "audio" and is_listening:
                try:
                    audio_bytes = base64.b64decode(msg["data"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_buffer.append(audio_np)

                    if vad and len(audio_np) > 0:
                        has_speech = vad.is_speech(audio_np)
                        await websocket.send_json({
                            "type": "vad_status",
                            "speech_detected": has_speech,
                        })
                except Exception as audio_err:
                    logger.warning(f"Audio decode error: {audio_err}")

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "clear_history":
                # Clear server-side conversation history
                backend.clear_history()
                logger.info(f"History cleared for {client_id}")
                await websocket.send_json({"type": "history_cleared"})

    except WebSocketDisconnect as e:
        logger.info(f"Client {client_id} disconnected (code={e.code})")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# Serve static files
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )