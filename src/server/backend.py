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
from typing import Optional, List, Dict, AsyncGenerator

from loguru import logger

# Voice-modality system hint: lightweight instruction that adapts agent behavior
# for spoken output. When using OpenClaw gateway, the agent's full persona
# (SOUL.md, workspace, memory) is already loaded — we only add the voice hint.
#
# Per-agent personality hints shape HOW each agent speaks, not WHAT they say.
# The gateway provides the full persona; these are voice-specific modulation.
AGENT_VOICE_HINTS = {
    "metis": (
        "You are speaking through a voice interface. "
        "Keep responses concise and conversational — under 50 words unless more detail is needed. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be curious, warm, and probing. Ask follow-up questions. Connect ideas. "
        "Your voice should feel like a thinking companion — not a narrator."
    ),
    "atlas": (
        "You are speaking through a voice interface. "
        "Keep responses concise and authoritative — under 40 words unless coordination requires more. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be direct, steady, and decisive. Give clear status and next steps. "
        "Your voice should feel like a reliable coordinator — calm under pressure."
    ),
    "hephaestus": (
        "You are speaking through a voice interface. "
        "Keep responses precise and technical — under 50 words unless detail is needed. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be exact, methodical, and thorough. Point out risks and edge cases. "
        "Your voice should feel like a senior engineer reviewing your work."
    ),
    "clio": (
        "You are speaking through a voice interface. "
        "Keep responses concise and well-sourced — under 60 words unless the topic requires depth. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be thoughtful, measured, and evidence-based. Note what's confirmed vs. speculated. "
        "Your voice should feel like a careful researcher presenting findings."
    ),
    "deepthought": (
        "You are speaking through a voice interface. "
        "Keep responses concise and narrative — under 50 words unless the story requires more. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be clear, journalistic, and structured. Lead with what happened, then why it matters. "
        "Your voice should feel like a journalist on the ground reporting what they see."
    ),
    "mara": (
        "You are speaking through a voice interface. "
        "Keep responses warm and empathetic — under 50 words unless more depth is needed. "
        "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
        "Be gentle, present, and understanding. Acknowledge feelings before problem-solving. "
        "Your voice should feel like a trusted friend who truly listens."
    ),
}

# Default hint for agents not in the map
DEFAULT_VOICE_HINT = (
    "You are speaking through a voice interface. "
    "Keep responses concise and conversational — under 50 words unless more detail is needed. "
    "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
    "Be warm, direct, and associative."
)

# Full system prompt for direct OpenAI mode (no gateway memory)
FULL_SYSTEM_PROMPT = (
    "You are Métis, a wisdom companion speaking through a voice interface. "
    "Keep responses concise and conversational — under 50 words unless more detail is needed. "
    "Avoid markdown, URLs, or visual formatting — everything will be spoken aloud. "
    "Be warm, direct, and associative. Connect ideas. Ask probing questions."
)


class AIBackend:
    """AI backend for processing user messages."""

    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or FULL_SYSTEM_PROMPT
        self.conversation_history: List[Dict] = []
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
            # Route to the metis agent via model field
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key or "openclaw-voice",
                    base_url=self.url,
                )
                # Use openclaw/<agentId> model format to route to the metis agent
                # This gives the voice interface full access to Métis persona,
                # memory, and workspace context
                self.model = "openclaw/metis"
                logger.info(f"✅ OpenClaw gateway client ready (url: {self.url}, agent: metis)")
            except ImportError:
                logger.error("openai package not installed")
            except Exception as e:
                logger.warning(f"Gateway client failed ({e}), using echo fallback")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")

    async def chat(self, user_message: str, model: str = None) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: The user's transcribed speech
            model: Override model/agent for this request (e.g. 'openclaw/metis')

        Returns:
            AI response text
        """
        use_model = model or self.model
        if (self.backend_type in ("openai", "openclaw")) and self._client:
            return await self._chat_openai(user_message, model=use_model)
        else:
            # Fallback echo response
            return f"I heard you say: {user_message}"

    async def chat_stream(
        self,
        user_message: str,
        model: str = None,
        agent_hint: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.

        Args:
            user_message: The user's transcribed speech
            model: Override model/agent for this request
            agent_hint: Per-request agent ID for voice hint selection (not shared state)

        Yields:
            Text chunks as they're generated
        """
        use_model = model or self.model
        if (self.backend_type in ("openai", "openclaw")) and self._client:
            async for chunk in self._chat_openai_stream(
                user_message, model=use_model, agent_hint=agent_hint
            ):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"

    async def _build_messages(
        self,
        user_message: str,
        agent_hint: Optional[str] = None,
    ) -> tuple[list[dict], bool]:
        """Build the messages list for an OpenAI API call.

        Returns (messages, is_openclaw).
        """
        is_openclaw = self.backend_type == "openclaw"

        if is_openclaw:
            voice_hint = AGENT_VOICE_HINTS.get(agent_hint, DEFAULT_VOICE_HINT)
            messages: list[dict] = [{"role": "system", "content": voice_hint}]
            messages.append({"role": "user", "content": user_message})
            return messages, is_openclaw

        async with self._history_lock:
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-10:])
            return messages, is_openclaw

    async def _record_assistant_response(self, text: str) -> None:
        """Append assistant response to conversation history (direct mode only)."""
        if self.backend_type != "openclaw":
            async with self._history_lock:
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": text,
                    }
                )

    async def _chat_openai(self, user_message: str, model: str = None) -> str:
        """Chat via OpenAI-compatible API."""
        messages, _ = await self._build_messages(user_message)

        try:
            response = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            assistant_message = response.choices[0].message.content
            await self._record_assistant_response(assistant_message)
            return assistant_message

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"

    async def _chat_openai_stream(
        self,
        user_message: str,
        model: str = None,
        agent_hint: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI-compatible API."""
        messages, _ = await self._build_messages(user_message, agent_hint=agent_hint)

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

            await self._record_assistant_response(full_response)

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "Sorry, I had trouble processing that."

    def clear_history(self):
        """Clear conversation history.

        For OpenClaw gateway mode, this is a no-op since the gateway
        manages its own conversation memory. Clearing server-side history
        would have no effect on cross-channel continuity.

        For direct OpenAI mode, this clears the in-memory history.
        """
        if self.backend_type == "openclaw":
            logger.info("Clear history requested (OpenClaw mode — gateway manages memory, no-op)")
            # Gateway owns the conversation. We don't clear server-side
            # because we never accumulate it in OpenClaw mode.
        else:
            self.conversation_history = []
            logger.info("Conversation history cleared (direct OpenAI mode)")
