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
import secrets
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

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


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
    # Voice model override: use a faster model for voice interactions.
    # When set, the voice server sends this model to the gateway instead of
    # the agent's configured model. E.g. "deepseek-v4-flash" for lower latency.
    # Format: just the model name without the provider prefix (gateway resolves it),
    # or a full provider/model path like "nvidia/deepseek-ai/deepseek-v4-flash".
    voice_model: Optional[str] = None

    # Audio
    sample_rate: int = 16000

    # System prompt for voice mode — used only in direct OpenAI mode.
    # For OpenClaw gateway mode, VOICE_SYSTEM_HINT from backend.py is used instead.
    system_prompt: str = "unused_in_openclaw_mode"

    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"
        extra = "ignore"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.3.0")

# CORS for cross-origin WebSocket
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
        # Determine voice model: if voice_model is set, use it for lower latency;
        # otherwise route through the agent's configured model
        if settings.voice_model:
            # Direct model for voice — bypasses agent's default model for speed
            # Gateway still provides agent persona/memory
            voice_model_id = settings.voice_model
            if not voice_model_id.startswith(('openclaw/', 'ollama/', 'nvidia/', 'synthetic/')):
                # Bare model name — prefix with openclaw/metis/ so gateway routes to agent + model
                voice_model_id = f"openclaw/metis/{voice_model_id}"
            logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url} (voice model: {voice_model_id})")
            backend = AIBackend(
                backend_type="openclaw",
                url=f"{gateway_url}/v1",
                model=voice_model_id,
                api_key=gateway_token,
                system_prompt=settings.system_prompt,
            )
        else:
            logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url} (agent default model)")
            backend = AIBackend(
                backend_type="openclaw",
                url=f"{gateway_url}/v1",
                model="openclaw/metis",
                api_key=gateway_token,
                system_prompt=settings.system_prompt,
            )
    else:
        logger.warning("⚠️ Using direct OpenAI backend — conversation history is global (shared across all clients). "
                       "For multi-user deployments, use OpenClaw gateway which manages per-session memory.")
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



# Sentence separators for streaming TTS
SENTENCE_SEPS = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
async def speak_sentence(
    websocket: WebSocket,
    text: str,
    seq: int,
) -> int:
    """Clean text for speech, synthesize, and stream audio chunks.

    Sends sentence_start, audio_chunks, and sentence_end messages.
    Returns the next sequence number.
    """
    speech_text = clean_for_speech(text)
    if not speech_text:
        return seq

    logger.debug(f"Synthesizing: {speech_text[:50]}...")
    await websocket.send_json({"type": "sentence_start", "seq": seq})
    try:
        async for audio_chunk in tts.synthesize_stream(speech_text):
            audio_b64 = base64.b64encode(audio_chunk).decode()
            await websocket.send_json({
                "type": "audio_chunk",
                "data": audio_b64,
                "sample_rate": 24000,
                "seq": seq,
            })
    except Exception as tts_err:
        logger.error(f"TTS error: {tts_err}")
    await websocket.send_json({"type": "sentence_end", "seq": seq})
    return seq + 1


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
        "stt": stt.status() if stt else {"backend": "not_loaded"},
        "tts": tts.status() if tts else {"backend": "not_loaded"},
        "backend": backend.backend_type if backend else "not_loaded",
        "vad": "loaded" if vad else "not_loaded",
        "config": {
            "stt_model": settings.stt_model,
            "tts_model": settings.tts_model,
            "supertonic_model": os.getenv("SUPERTONIC_MODEL", "supertonic-2"),
            "supertonic_voice": os.getenv("SUPERTONIC_VOICE", "F2"),
            "voice_model": settings.voice_model,
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


@app.post("/api/voices")
async def upload_voice(name: str, file: bytes):
    """Upload a voice sample for TTS cloning. Returns a voice ID."""
    VOICES_DIR.mkdir(exist_ok=True)
    voice_id = f"{name}_{secrets.token_hex(4)}"
    path = VOICES_DIR / f"{voice_id}.wav"
    with open(path, "wb") as f:
        f.write(file)
    logger.info(f"Saved voice sample: {voice_id} ({len(file)} bytes)")
    return {"voice_id": voice_id, "path": str(path)}


@app.get("/api/voices")
async def list_voices():
    """List available voice samples."""
    VOICES_DIR.mkdir(exist_ok=True)
    voices = []
    for f in sorted(VOICES_DIR.iterdir()):
        if f.suffix in (".wav", ".mp3", ".ogg"):
            voices.append({"voice_id": f.stem, "name": f.stem.split("_")[0], "size": f.stat().st_size})
    return {"voices": voices}


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
    MAX_AUDIO_BUFFER_SECONDS = 30  # Cap audio buffer at 30s of 16kHz float32
    MAX_AUDIO_BUFFER_SAMPLES = settings.sample_rate * MAX_AUDIO_BUFFER_SECONDS
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
                    # Switch TTS voice to match agent persona
                    if session_agent in ChatterboxTTS.AGENT_VOICE_MAP and tts._backend == "supertonic":
                        new_voice = ChatterboxTTS.AGENT_VOICE_MAP[session_agent]
                        if new_voice != tts._supertonic_voice:
                            try:
                                tts._supertonic_style = tts._supertonic_tts.get_voice_style(new_voice)
                                tts._supertonic_voice = new_voice
                                logger.info(f"Switched TTS voice to {new_voice} for agent {session_agent}")
                            except Exception as e:
                                logger.warning(f"Failed to switch voice to {new_voice}: {e}")
                await websocket.send_json({"type": "listening_started"})

            elif msg_type == "stop_listening":
                is_listening = False
                t_start = time.monotonic()  # Pipeline timing start

                if audio_buffer:
                    try:
                        audio_data = np.concatenate(audio_buffer)
                        audio_duration = len(audio_data) / settings.sample_rate
                        logger.info(f"Audio received: {audio_duration:.1f}s, {len(audio_buffer)} chunks")

                        t_stt_start = time.monotonic()
                        logger.debug("Transcribing audio...")
                        transcript = await stt.transcribe(audio_data)
                        t_stt = time.monotonic() - t_stt_start

                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "final": True,
                        })
                        logger.info(f"STT: {t_stt:.2f}s | Transcript: {transcript[:80]}")

                        if transcript.strip():
                            logger.debug("Streaming AI response...")
                            full_response = ""
                            sentence_buffer = ""
                            agent_model = f"openclaw/{session_agent}" if session_agent else None

                            audio_seq = 0  # Sequence counter for ordered playback
                            t_llm_start = time.monotonic()
                            t_first_token = None
                            t_first_audio = None
                            sentence_count = 0

                            async for chunk in backend.chat_stream(transcript, model=agent_model):
                                full_response += chunk
                                sentence_buffer += chunk

                                if t_first_token is None:
                                    t_first_token = time.monotonic() - t_llm_start

                                await websocket.send_json({
                                    "type": "response_chunk",
                                    "text": chunk,
                                })

                                # Synthesize completed sentences
                                while any(sep in sentence_buffer for sep in SENTENCE_SEPS):
                                    earliest_idx = len(sentence_buffer)
                                    for sep in SENTENCE_SEPS:
                                        idx = sentence_buffer.find(sep)
                                        if idx != -1 and idx < earliest_idx:
                                            earliest_idx = idx + len(sep)

                                    if earliest_idx < len(sentence_buffer):
                                        sentence = sentence_buffer[:earliest_idx].strip()
                                        sentence_buffer = sentence_buffer[earliest_idx:]

                                        if sentence:
                                            t_tts_start = time.monotonic()
                                            audio_seq = await speak_sentence(
                                                websocket, sentence, audio_seq
                                            )
                                            tts_time = time.monotonic() - t_tts_start
                                            sentence_count += 1
                                            if t_first_audio is None:
                                                t_first_audio = time.monotonic() - t_start
                                                logger.info(f"First audio: {t_first_audio:.2f}s (STT={t_stt:.2f}s, first_token={t_first_token:.2f}s, TTS_1st={tts_time:.2f}s)")
                                    else:
                                        break

                            # Handle remaining text
                            if sentence_buffer.strip():
                                t_tts_start = time.monotonic()
                                await speak_sentence(
                                    websocket, sentence_buffer.strip(), audio_seq
                                )
                                sentence_count += 1

                            t_total = time.monotonic() - t_start
                            t_llm_total = time.monotonic() - t_llm_start
                            await websocket.send_json({
                                "type": "response_complete",
                                "text": full_response,
                                "latency": {
                                    "stt_ms": round(t_stt * 1000),
                                    "first_token_ms": round(t_first_token * 1000) if t_first_token else None,
                                    "first_audio_ms": round(t_first_audio * 1000) if t_first_audio else None,
                                    "llm_total_ms": round(t_llm_total * 1000),
                                    "total_ms": round(t_total * 1000),
                                    "audio_duration_s": round(audio_duration, 1),
                                    "sentences": sentence_count,
                                },
                            })
                            logger.info(
                                f"Pipeline: {t_total:.2f}s total "
                                f"(STT={t_stt:.2f}s, LLM={t_llm_total:.2f}s, "
                                f"first_token={t_first_token:.2f}s, "
                                f"first_audio={t_first_audio:.2f}s) "
                                f"| {sentence_count} sentences | {len(full_response)} chars"
                            )

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

                    # Cap audio buffer to prevent unbounded memory growth
                    total_samples = sum(len(chunk) for chunk in audio_buffer) + len(audio_np)
                    if total_samples > MAX_AUDIO_BUFFER_SAMPLES:
                        logger.warning(f"Audio buffer cap reached ({total_samples} samples), forcing stop_listening")
                        audio_buffer.append(audio_np)
                        # Trigger processing instead of crashing
                        is_listening = False
                        # Fall through to the stop_listening processing below
                    else:
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

            elif msg_type == "set_voice":
                voice_id = msg.get("voice_id", "")
                logger.info(f"Switching voice to {voice_id} for {client_id}")
                if voice_id and voice_id != "default":
                    voice_path = str(VOICES_DIR / f"{voice_id}.wav")
                    if os.path.isfile(voice_path):
                        tts.voice_sample = voice_path
                        await websocket.send_json({"type": "voice_set", "voice_id": voice_id})
                    else:
                        await websocket.send_json({"type": "error", "message": f"Voice {voice_id} not found"})
                else:
                    tts.voice_sample = None
                    await websocket.send_json({"type": "voice_set", "voice_id": "default"})

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