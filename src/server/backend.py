"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.
"""

import asyncio
from typing import Optional, List, Dict, AsyncGenerator

from loguru import logger

from .constants import SYSTEM_PROMPT


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
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.conversation_history: List[Dict] = []
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
    
    async def chat_stream(self, user_message: str, model: str = None) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.
        
        Args:
            user_message: The user's transcribed speech
            model: Override model/agent for this request
            
        Yields:
            Text chunks as they're generated
        """
        use_model = model or self.model
        if (self.backend_type in ("openai", "openclaw")) and self._client:
            async for chunk in self._chat_openai_stream(user_message, model=use_model):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"
    
    async def _chat_openai(self, user_message: str, model: str = None) -> str:
        """Chat via OpenAI API."""
        use_model = model or self.model
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])  # Last 10 turns
        
        try:
            response = await self._client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=500,  # Allow longer for voice
                temperature=0.7,
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"
    
    async def _chat_openai_stream(self, user_message: str, model: str = None) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI API."""
        use_model = model or self.model
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])
        
        full_response = ""
        
        try:
            stream = await self._client.chat.completions.create(
                model=use_model,
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
            
            # Add complete response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
            })
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "Sorry, I had trouble processing that."
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
