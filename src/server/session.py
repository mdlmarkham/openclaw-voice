"""
Session context for a voice pipeline run.

Carries per-request metadata that the pipeline needs
(agent selection, voice ID, etc.) without coupling to WebSocket details.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionContext:
    agent_id: Optional[str] = None
    voice_id: Optional[str] = None
