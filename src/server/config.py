"""
Server configuration from environment variables.
"""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


class Settings(BaseSettings):
    """Server configuration loaded from env vars (OPENCLAW_ prefix)."""

    host: str = "0.0.0.0"
    port: int = 8765

    require_auth: bool = False
    master_key: Optional[str] = None

    stt_model: str = "base"
    stt_device: str = "auto"

    tts_model: str = "edge"
    tts_voice: Optional[str] = None

    backend_type: str = "openclaw"
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None

    openclaw_gateway_url: Optional[str] = None
    openclaw_gateway_token: Optional[str] = None
    voice_model: Optional[str] = None

    # TTS backend selection: "auto" | "higgs" | "supertonic" (default: auto)
    tts_backend: str = "auto"

    # Higgs / Boson Cloud API
    boson_api_key: Optional[str] = None
    higgs_default_voice: str = "eleanor"
    higgs_model: str = "higgs-audio-v3-tts"
    higgs_api_url: str = "https://api.boson.ai/v1/audio/speech"
    higgs_timeout_seconds: int = 3

    sample_rate: int = 16000
    cors_origins: str = "*"
    system_prompt: str = "unused_in_openclaw_mode"

    model_config = ConfigDict(
        env_prefix="OPENCLAW_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
