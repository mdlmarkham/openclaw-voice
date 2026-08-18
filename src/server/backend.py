"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.

OpenClaw gateway mode: sends only the user message + voice-modality system hint.
The gateway already maintains full agent persona, memory, and workspace context,
so we don't duplicate conversation history. This ensures continuity across voice
and text channels — a conversation started in Telegram continues seamlessly in voice.

Direct OpenAI mode: manages its own conversation history (last 10 turns) since
there's no external memory store. Uses per-session history via lock to prevent
cross-session interleaving.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, AsyncGenerator

from loguru import logger

# ── Agent voice configuration ────────────────────────────────────
#
# Each agent can have:
#   hint:        System prompt that shapes HOW the agent speaks
#   higgs_voice: Preset voice for Higgs TTS (e.g. 'eleanor', 'jake')
#   higgs_tags:  Control tokens for emotion/prosody/style
#   ref_audio:   Path to voice cloning reference (Phase 3)
#   ref_text:    Transcript of ref_audio (improves clone quality)
#   supertonic_voice: Voice preset for Supertonic fallback

AGENT_VOICE_CONFIG = {
    "metis": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses concise and conversational — under 50 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be curious, warm, and probing. Ask follow-up questions. Connect ideas. "
            "Your voice should feel like a thinking companion — not a narrator."
        ),
        "higgs_voice": "eleanor",
        "higgs_tags": "<|emotion:contemplation|><|prosody:pause|>",
        "supertonic_voice": "F2",
    },
    "atlas": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses concise and authoritative — under 40 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be direct, steady, and decisive. Give clear status and next steps. "
            "Your voice should feel like a reliable coordinator — calm under pressure."
        ),
        "higgs_voice": "jake",
        "higgs_tags": "<|emotion:determination|>",
        "supertonic_voice": "M2",
    },
    "hephaestus": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses precise and technical — under 50 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be exact, methodical, and thorough. Point out risks and edge cases. "
            "Your voice should feel like a senior engineer reviewing your work."
        ),
        "higgs_voice": "jake",
        "higgs_tags": "<|emotion:contentment|><|prosody:speed_slow|>",
        "supertonic_voice": "M4",
    },
    "clio": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses concise and well-sourced — under 60 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be thoughtful, measured, and evidence-based. Note what's confirmed vs. speculated. "
            "Your voice should feel like a careful researcher presenting findings."
        ),
        "higgs_voice": "eleanor",
        "higgs_tags": "<|emotion:contemplation|><|prosody:speed_slow|>",
        "supertonic_voice": "F4",
    },
    "deepthought": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses concise and narrative — under 50 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be clear, journalistic, and structured. Lead with what happened, then why it matters. "
            "Your voice should feel like a journalist on the ground reporting what they see."
        ),
        "higgs_voice": "jake",
        "higgs_tags": "<|emotion:enthusiasm|>",
        "supertonic_voice": "M1",
    },
    "mara": {
        "hint": (
            "You are speaking through a voice interface — your reply will be read aloud. "
            "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
            "Keep responses warm and empathetic — under 50 words unless the user asks for depth. "
            "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
            "Be gentle, present, and understanding. Acknowledge feelings before problem-solving. "
            "Your voice should feel like a trusted friend who truly listens."
        ),
        "higgs_voice": "eleanor",
        "higgs_tags": "<|emotion:affection|><|prosody:pause|>",
        "supertonic_voice": "F5",
    },
}

# Default hint for agents not in the map
DEFAULT_VOICE_HINT = (
    "You are speaking through a voice interface — your reply will be read aloud. "
    "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
    "Keep responses concise and conversational — under 50 words unless the user asks for depth. "
    "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
    "Be warm, direct, and associative."
)

# ── Configurable per-agent voice hints (issue #33) ──────────────────
#
# Hints resolve in this order:
#   1. Environment variable  VOICE_HINT_<AGENT>  (e.g. VOICE_HINT_METIS)
#   2. Config file (path via VOICE_HINT_CONFIG env, default ./voice_hints.json)
#   3. Built-in AGENT_VOICE_CONFIG hints
#   4. DEFAULT_VOICE_HINT
#
# Config file schema (JSON or YAML):
#   {
#     "metis": { "hint": "...", "word_budget": 50 },
#     "atlas": { "hint": "..." }
#   }
# - "hint": full system-message text (replaces the built-in for that agent).
# - "word_budget": optional positive int; injects a "keep it under N words"
#   clause into the hint at build time so budgets stay tunable without
#   editing the prose. If absent, the hint is used verbatim.

VOICE_HINT_CONFIG_ENV = "VOICE_HINT_CONFIG"
VOICE_HINT_CONFIG_DEFAULT = "voice_hints.json"

_voice_hint_config: Optional[Dict[str, Dict]] = None
_voice_hint_config_path: Optional[str] = None


def _voice_hint_config_path_from_env() -> Optional[str]:
    """Return the config file path from VOICE_HINT_CONFIG, or None if unset."""
    return os.environ.get(VOICE_HINT_CONFIG_ENV)


def _load_voice_hint_config(path: str) -> Dict[str, Dict]:
    """Parse and validate a voice-hint config file.

    Raises ValueError with a clear message naming the file and offending key
    on malformed JSON/YAML or wrong types, so a bad config can't silently
    ship a broken voice experience. Unknown keys are ignored (forward
    compatible).
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"VOICE_HINT_CONFIG points to '{path}' but no such file exists")
    import yaml

    try:
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Malformed voice-hint config file '{path}': {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Voice-hint config file '{path}' must be a JSON/YAML object mapping "
            f"agent ids to objects, got {type(data).__name__}"
        )

    validated: Dict[str, Dict] = {}
    for agent_id, entry in data.items():
        if not isinstance(agent_id, str):
            raise ValueError(
                f"Voice-hint config file '{path}': agent id must be a string, "
                f"got {type(agent_id).__name__}"
            )
        if not isinstance(entry, dict):
            raise ValueError(
                f"Voice-hint config file '{path}': entry for '{agent_id}' must be "
                f"an object, got {type(entry).__name__}"
            )
        hint = entry.get("hint")
        if not isinstance(hint, str) or not hint.strip():
            raise ValueError(
                f"Voice-hint config file '{path}': entry for '{agent_id}' must have "
                f"a non-empty string 'hint'"
            )
        budget = entry.get("word_budget")
        if budget is not None and (
            not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
        ):
            raise ValueError(
                f"Voice-hint config file '{path}': 'word_budget' for '{agent_id}' "
                f"must be a positive integer, got {budget!r}"
            )
        unknown = set(entry) - {"hint", "word_budget"}
        if unknown:
            logger.warning(
                f"Voice-hint config file '{path}': ignoring unknown key(s) "
                f"{sorted(unknown)} for agent '{agent_id}'"
            )
        validated[agent_id] = {"hint": hint, "word_budget": budget}
    return validated


def _get_voice_hint_config() -> Dict[str, Dict]:
    """Load (and cache) the voice-hint config file, if any.

    Returns {} when no config file is configured or present. Raises ValueError
    on a configured-but-malformed file (fail fast at startup).
    """
    global _voice_hint_config, _voice_hint_config_path
    path = _voice_hint_config_path_from_env()
    if path is None:
        default = Path(VOICE_HINT_CONFIG_DEFAULT)
        if default.is_file():
            path = str(default)
        else:
            return {}
    if _voice_hint_config is not None and _voice_hint_config_path == path:
        return _voice_hint_config
    _voice_hint_config = _load_voice_hint_config(path)
    _voice_hint_config_path = path
    return _voice_hint_config


def _apply_word_budget(hint: str, word_budget: Optional[int]) -> str:
    """Inject a word-budget clause into a hint, if a budget is configured."""
    if word_budget is None:
        return hint
    clause = f"Keep responses under {word_budget} words unless more detail is needed."
    if "under " in hint and " words" in hint:
        return hint
    return f"{hint} {clause}"


def resolve_voice_hint(agent_id: Optional[str]) -> str:
    """Resolve the voice-modality system hint for an agent.

    Priority: env override > config file > built-in AGENT_VOICE_CONFIG >
    DEFAULT_VOICE_HINT. Env overrides are validated at call time (cheap
    string check).
    """
    agent = agent_id or "metis"
    env_key = f"VOICE_HINT_{agent.upper().replace('-', '_')}"
    env_hint = os.environ.get(env_key)
    if env_hint is not None:
        if not env_hint.strip():
            logger.warning(f"{env_key} is empty; ignoring env override")
        else:
            return env_hint

    cfg = _get_voice_hint_config().get(agent)
    if cfg is not None:
        return _apply_word_budget(cfg["hint"], cfg.get("word_budget"))

    builtin = AGENT_VOICE_CONFIG.get(agent, {}).get("hint")
    if builtin:
        return builtin
    return DEFAULT_VOICE_HINT


# Full system prompt for direct OpenAI mode (no gateway memory)
FULL_SYSTEM_PROMPT = (
    "You are Métis, a wisdom companion speaking through a voice interface — your reply will be read aloud. "
    "Lead with the answer in 1-2 sentences. If the question genuinely needs depth, say so and OFFER the full version (\"want the short version or the full picture?\") rather than launching into it — let the user pull for more. "
    "Keep responses concise and conversational — under 50 words unless the user asks for depth. "
    "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
    "Be warm, direct, and associative. Connect ideas. Ask probing questions."
)


def resolve_openclaw_model(agent_id: Optional[str], global_default: Optional[str] = None) -> str:
    """Resolve the OpenClaw model ID for a given agent.

    Priority: per-agent env override > global OPENCLAW_VOICE_MODEL > bare
    "openclaw/<agent>" (lets the gateway apply the agent's own default model).

    Args:
        agent_id: The agent name (e.g. "metis", "atlas"). Defaults to "metis".
        global_default: The global voice_model setting (settings.voice_model).

    Returns:
        A model ID string like "openclaw/metis/glm-5.1:cloud" or "openclaw/atlas".
    """
    from .config import settings

    agent = agent_id or "metis"
    per_agent = getattr(settings, f"voice_model_{agent}", None)
    model_suffix = per_agent or global_default
    if not model_suffix:
        return f"openclaw/{agent}"
    if model_suffix.startswith(("openclaw/", "ollama/", "nvidia/", "synthetic/")):
        return model_suffix
    return f"openclaw/{agent}/{model_suffix}"


class AIBackend:
    """AI backend for processing user messages."""

    _SESSION_HISTORY_TTL = 3600  # Sessions expire after 1 hour of inactivity

    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        voice_model_default: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or FULL_SYSTEM_PROMPT
        self._voice_model_default = voice_model_default
        self._session_histories: Dict[str, List[Dict]] = {}
        self._session_last_used: Dict[str, float] = {}
        self._history_lock = asyncio.Lock()
        self._client = None
        self._setup_client()

    def _setup_client(self):
        """Set up the API client."""
        if self.backend_type == "openai":
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                )
                logger.info(f"✅ OpenAI client ready (model: {self.model})")
            except ImportError:
                logger.error("openai package not installed")
            except Exception as e:
                logger.warning(f"OpenAI client failed ({e}), using echo fallback")
        elif self.backend_type == "openclaw":
            # OpenClaw gateway uses OpenAI-compatible API
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key or "openclaw-voice",
                    base_url=self.url,
                )
                # self.model is NOT overwritten here — whatever was passed into
                # __init__ (or resolved per-agent at call time) stands. The
                # per-request model is resolved in chat_stream via
                # resolve_openclaw_model(agent_hint, ...). See issue #3.
                logger.info(
                    f"✅ OpenClaw gateway client ready (url: {self.url}, model: {self.model})"
                )
            except ImportError:
                logger.error("openai package not installed")
            except Exception as e:
                logger.warning(f"Gateway client failed ({e}), using echo fallback")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")

    async def _get_session_history(self, session_id: str) -> List[Dict]:
        """Get (or create) the conversation history for a session.

        Also opportunistically evicts histories that have been idle longer
        than _SESSION_HISTORY_TTL. Must be called under _history_lock.
        """
        now = time.monotonic()
        self._session_last_used[session_id] = now
        stale = [
            sid
            for sid, ts in self._session_last_used.items()
            if now - ts > self._SESSION_HISTORY_TTL
        ]
        for sid in stale:
            self._session_histories.pop(sid, None)
            self._session_last_used.pop(sid, None)
        return self._session_histories.setdefault(session_id, [])

    def _resolve_model(self, agent_hint: Optional[str] = None) -> str:
        """Resolve the model ID for this request.

        For openclaw backend: uses resolve_openclaw_model() to respect
        per-agent env overrides and global default (issue #3).
        For other backends: returns self.model.
        """
        if self.backend_type == "openclaw":
            return resolve_openclaw_model(agent_hint, self._voice_model_default)
        return self.model

    async def chat(
        self,
        user_message: str,
        model: str = None,
        session_id: str = "default",
        agent_hint: Optional[str] = None,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: The user's transcribed speech
            model: Override model/agent for this request (e.g. 'openclaw/metis')
            session_id: Per-session ID for conversation history isolation
            agent_hint: Per-request agent ID for model resolution

        Returns:
            AI response text
        """
        use_model = model or self._resolve_model(agent_hint)
        if (self.backend_type in ("openai", "openclaw")) and self._client:
            return await self._chat_openai(user_message, model=use_model, session_id=session_id)
        else:
            # Fallback echo response
            return f"I heard you say: {user_message}"

    async def chat_stream(
        self,
        user_message: str,
        model: str = None,
        agent_hint: Optional[str] = None,
        reconnect: bool = False,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.

        Args:
            user_message: The user's transcribed speech
            model: Override model/agent for this request
            agent_hint: Per-request agent ID for voice hint selection
            reconnect: If True, send context resumption message for gateway
            session_id: Per-session ID for conversation history isolation.
                In gateway mode this is accepted but not used server-side
                (the gateway manages its own memory).

        Yields:
            Text chunks as they're generated
        """
        use_model = model or self._resolve_model(agent_hint)
        if (self.backend_type in ("openai", "openclaw")) and self._client:
            async for chunk in self._chat_openai_stream(
                user_message,
                model=use_model,
                agent_hint=agent_hint,
                reconnect=reconnect,
                session_id=session_id,
            ):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"

    async def _build_messages(
        self,
        user_message: str,
        agent_hint: Optional[str] = None,
        reconnect: bool = False,
        session_id: str = "default",
    ) -> tuple[list[dict], bool]:
        """Build the messages list for an OpenAI API call.

        Returns (messages, is_openclaw).
        """
        is_openclaw = self.backend_type == "openclaw"

        if is_openclaw:
            voice_hint = resolve_voice_hint(agent_hint)
            messages: list[dict] = [{"role": "system", "content": voice_hint}]
            if reconnect:
                messages.append(
                    {
                        "role": "system",
                        "content": "The user has reconnected after a brief disconnection. "
                        "Continue the conversation naturally from where you left off. "
                        "Do not acknowledge the reconnection unless asked.",
                    }
                )
            messages.append({"role": "user", "content": user_message})
            return messages, is_openclaw

        async with self._history_lock:
            history = await self._get_session_history(session_id)
            history.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(history[-10:])
            return messages, is_openclaw

    async def _record_assistant_response(self, text: str, session_id: str = "default") -> None:
        """Append assistant response to conversation history (direct mode only)."""
        if self.backend_type != "openclaw":
            async with self._history_lock:
                history = await self._get_session_history(session_id)
                history.append(
                    {
                        "role": "assistant",
                        "content": text,
                    }
                )

    async def _chat_openai(
        self, user_message: str, model: str = None, session_id: str = "default"
    ) -> str:
        """Chat via OpenAI-compatible API."""
        messages, _ = await self._build_messages(user_message, session_id=session_id)

        try:
            response = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            assistant_message = response.choices[0].message.content
            await self._record_assistant_response(assistant_message, session_id=session_id)
            return assistant_message

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"

    async def _chat_openai_stream(
        self,
        user_message: str,
        model: str = None,
        agent_hint: Optional[str] = None,
        reconnect: bool = False,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI-compatible API."""
        messages, _ = await self._build_messages(
            user_message, agent_hint=agent_hint, reconnect=reconnect, session_id=session_id
        )

        full_response = ""

        try:
            stream = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            await self._record_assistant_response(full_response, session_id=session_id)

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "Sorry, I had trouble processing that."

    def clear_history(self, session_id: Optional[str] = None):
        """Clear conversation history.

        For OpenClaw gateway mode, this is a no-op since the gateway
        manages its own conversation memory. Clearing server-side history
        would have no effect on cross-channel continuity.

        For direct OpenAI mode, clears the in-memory history for the given
        session_id. If session_id is None, clears all sessions' histories.
        """
        if self.backend_type == "openclaw":
            logger.info("Clear history requested (OpenClaw mode — gateway manages memory, no-op)")
            # Gateway owns the conversation. We don't clear server-side
            # because we never accumulate it in OpenClaw mode.
        elif session_id is not None:
            self._session_histories.pop(session_id, None)
            self._session_last_used.pop(session_id, None)
            logger.info(f"Conversation history cleared for session {session_id}")
        else:
            self._session_histories.clear()
            self._session_last_used.clear()
            logger.info("All conversation histories cleared (direct OpenAI mode)")
