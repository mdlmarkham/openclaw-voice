"""
OpenClaw Voice Server — application wiring.

Creates the FastAPI app, configures middleware, manages the lifespan
(model initialization / shutdown), and registers routes.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .auth import load_keys_from_env
from .backend import AIBackend
from .config import settings
from .pipeline import VoicePipeline
from .routes import router
from .stt import WhisperSTT
from .tts import ChatterboxTTS, TTSRouter
from .tts_higgs import HiggsTTS
from .vad import VoiceActivityDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on server start and clean up on shutdown."""
    from . import state

    state._startup_time = time.time()
    logger.info("Initializing OpenClaw Voice server (Olympus)...")

    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")

    state.stt_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
    state.tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

    logger.info("Loading STT, TTS, and VAD models (parallel)...")
    state.stt, state.tts, state.vad = await asyncio.gather(
        asyncio.to_thread(
            WhisperSTT,
            model_name=settings.stt_model,
            device=settings.stt_device,
            executor=state.stt_executor,
        ),
        asyncio.to_thread(
            ChatterboxTTS,
            voice_sample=settings.tts_voice,
            executor=state.tts_executor,
        ),
        asyncio.to_thread(VoiceActivityDetector),
    )

    higgs = HiggsTTS(
        api_key=settings.boson_api_key,
        default_voice=settings.higgs_default_voice,
        timeout=settings.higgs_timeout_seconds,
    )
    state.tts_router = TTSRouter(
        supertonic=state.tts,
        higgs=higgs,
        backend=settings.tts_backend,
    )

    if state.stt._backend == "mock":
        logger.warning(
            "⚠️ STT is in MOCK mode — install faster-whisper or openai-whisper for real speech recognition"
        )
    if state.tts._backend == "mock" and not higgs.available:
        logger.error(
            "❌ No TTS backend available — set BOSON_API_KEY or ELEVENLABS_API_KEY"
        )
    elif higgs.available:
        logger.info("🔊 Higgs Audio v3 TTS backend available")
    elif state.tts._backend != "elevenlabs":
        logger.warning(f"⚠️ TTS using fallback backend: {state.tts._backend}")

    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    backend_configured = bool(gateway_url and gateway_token) or bool(openai_key)

    if not backend_configured:
        logger.error(
            "❌ No AI backend configured — set OPENAI_API_KEY or OPENCLAW_GATEWAY_URL+OPENCLAW_GATEWAY_TOKEN"
        )

    if gateway_url and gateway_token:
        if settings.voice_model:
            voice_model_id = settings.voice_model
            if not voice_model_id.startswith(("openclaw/", "ollama/", "nvidia/", "synthetic/")):
                voice_model_id = f"openclaw/metis/{voice_model_id}"
            logger.info(
                f"🦞 Connecting to OpenClaw gateway: {gateway_url} (voice model: {voice_model_id})"
            )
            state.backend = AIBackend(
                backend_type="openclaw",
                url=f"{gateway_url}/v1",
                model=voice_model_id,
                api_key=gateway_token,
                system_prompt=settings.system_prompt,
            )
        else:
            logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url} (agent default model)")
            state.backend = AIBackend(
                backend_type="openclaw",
                url=f"{gateway_url}/v1",
                model="openclaw/metis",
                api_key=gateway_token,
                system_prompt=settings.system_prompt,
            )
    else:
        logger.warning(
            "⚠️ Using direct OpenAI backend — conversation history is global (shared across all clients). "
            "For multi-user deployments, use OpenClaw gateway which manages per-session memory."
        )
        logger.info(f"Connecting to backend: {settings.backend_type}")
        state.backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=openai_key,
            system_prompt=settings.system_prompt,
        )

    logger.info("Initializing voice pipeline")
    state.pipeline = VoicePipeline(
        stt=state.stt,
        tts=state.tts_router or state.tts,
        backend=state.backend,
        sample_rate=settings.sample_rate,
    )

    logger.info("✅ OpenClaw Voice server ready!")

    yield

    logger.info("Shutting down OpenClaw Voice server...")
    state.stt = state.tts = state.tts_router = state.backend = state.vad = state.pipeline = None
    if state.stt_executor:
        state.stt_executor.shutdown(wait=False)
    if state.tts_executor:
        state.tts_executor.shutdown(wait=False)
    state.stt_executor = state.tts_executor = None


app = FastAPI(title="OpenClaw Voice", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

client_dir = Path(__file__).resolve().parent.parent / "client"
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
