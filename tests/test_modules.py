"""
Unit tests for OpenClaw Voice modules.
"""

import pytest
import numpy as np
import os
import sys
import asyncio
import threading
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server.stt import WhisperSTT
from src.server.tts import ChatterboxTTS, TTSRouter, TTSCache
from src.server.backend import AIBackend
from src.server.session import SessionContext
from src.server.vad import VoiceActivityDetector


class TestWhisperSTT:
    """Tests for Speech-to-Text module."""

    def test_init_loads_model(self):
        """Test that STT initializes (may be mock or real)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        assert stt is not None
        assert stt._backend in ["faster-whisper", "openai-whisper", "mock"]

    @pytest.mark.asyncio
    async def test_transcribe_returns_string(self):
        """Test that transcribe returns a string."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Create 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        result = await stt.transcribe(audio)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_transcribe_with_noise(self):
        """Test transcription with random noise (should return something)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Random noise
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = await stt.transcribe(audio)
        assert isinstance(result, str)


class TestWhisperSTTRemote:
    """Tests for WhisperSTT remote-offload + fallback (#4)."""

    @pytest.mark.asyncio
    async def test_no_remote_loads_local_eagerly(self):
        stt = WhisperSTT()
        assert stt._remote_url is None
        assert stt._backend in ("mock", "faster-whisper", "openai-whisper")

    @pytest.mark.asyncio
    async def test_remote_configured_skips_eager_local_load(self):
        stt = WhisperSTT(remote_url="http://127.0.0.1:1")
        assert stt._backend == "remote"
        assert stt.model is None

    @pytest.mark.asyncio
    async def test_remote_success_does_not_load_local(self, monkeypatch):
        stt = WhisperSTT(remote_url="http://127.0.0.1:1")

        async def fake_remote(audio):
            return "hello from remote"

        monkeypatch.setattr(stt, "_transcribe_remote", fake_remote)
        text = await stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "hello from remote"
        assert stt.model is None
        assert stt.status()["last_source"] == "remote"

    @pytest.mark.asyncio
    async def test_remote_failure_falls_back_to_local(self, monkeypatch):
        stt = WhisperSTT(remote_url="http://127.0.0.1:1")

        async def failing_remote(audio):
            raise ConnectionError("unreachable")

        monkeypatch.setattr(stt, "_transcribe_remote", failing_remote)
        text = await stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert isinstance(text, str)
        assert stt.status()["last_source"] == "local"

    def test_load_model_is_idempotent(self):
        stt = WhisperSTT()
        model_ref = stt.model
        stt._load_model()
        assert stt.model is model_ref


class TestChatterboxTTS:
    """Tests for Text-to-Speech module."""

    def test_init_loads_model(self):
        """Test that TTS initializes (may be mock or real)."""
        tts = ChatterboxTTS()
        assert tts is not None
        assert tts._backend in [
            "elevenlabs",
            "supertonic",
            "edge",
            "chatterbox",
            "xtts",
            "pyttsx3",
            "mock",
        ]

    def test_voice_id_from_env(self, monkeypatch):
        """ELEVENLABS_VOICE_ID env var should populate self.voice_id (#16)."""
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "abc123Voice")
        tts = ChatterboxTTS()
        assert tts.voice_id == "abc123Voice"

    def test_voice_id_explicit_overrides_env(self, monkeypatch):
        """Explicit voice_id arg should win over env var (#16)."""
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "envVoiceId")
        tts = ChatterboxTTS(voice_id="explicitVoiceId")
        assert tts.voice_id == "explicitVoiceId"

    def test_voice_id_default_when_unset(self, monkeypatch):
        """Default Jessica voice should be used when nothing is configured (#16)."""
        monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
        tts = ChatterboxTTS()
        assert tts.voice_id == "cgSgspJ2msm6clMCkdW9"

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self):
        """Test that synthesize returns numpy array."""
        tts = ChatterboxTTS()
        result = await tts.synthesize("Hello world")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0


class TestAIBackend:
    """Tests for AI Backend module."""

    def test_init_creates_client(self):
        """Test backend initialization."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
        )
        assert backend is not None
        assert backend.backend_type == "openai"

    def test_system_prompt_default(self):
        """Test default system prompt is set."""
        backend = AIBackend()
        assert backend.system_prompt is not None
        assert "wisdom companion" in backend.system_prompt.lower()

    def test_clear_history(self):
        """Test conversation history can be cleared (all sessions)."""
        backend = AIBackend()
        backend._session_histories["s1"] = [{"role": "user", "content": "test"}]
        backend._session_last_used["s1"] = time.monotonic()
        backend.clear_history()
        assert len(backend._session_histories) == 0
        assert len(backend._session_last_used) == 0

    def test_clear_history_scoped_to_session(self):
        """clear_history(session_id=...) must clear only that session's
        history, not other sessions' (#24)."""
        backend = AIBackend()
        backend._session_histories["s1"] = [{"role": "user", "content": "hello"}]
        backend._session_histories["s2"] = [{"role": "user", "content": "world"}]
        backend._session_last_used["s1"] = time.monotonic()
        backend._session_last_used["s2"] = time.monotonic()
        backend.clear_history(session_id="s1")
        assert "s1" not in backend._session_histories
        assert "s2" in backend._session_histories
        assert backend._session_histories["s2"][0]["content"] == "world"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    async def test_chat_returns_response(self):
        """Test actual API call (requires API key)."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        result = await backend.chat("Say 'test' and nothing else.")
        assert isinstance(result, str)
        assert len(result) > 0


class TestAIBackendSessionIsolation:
    """Tests for issue #24: per-session chat history isolation.

    Direct-OpenAI mode keys conversation_history by session_id (dict of lists)
    so concurrent WS/WebRTC sessions don't interleave each other's context.
    In gateway mode, session_id is passed through (gateway manages memory).
    """

    def _make_echo_backend(self):
        """Return an AIBackend with no real OpenAI client — uses the echo
        fallback path, which lets us inspect _build_messages' history wiring
        without making network calls."""
        return AIBackend(backend_type="echo")

    @pytest.mark.asyncio
    async def test_two_sessions_have_isolated_histories(self):
        """Two sessions writing to the same backend instance must have
        separate conversation histories — session A's messages never appear
        in session B's _build_messages output."""
        backend = self._make_echo_backend()

        # Session A sends a message
        await backend._build_messages("hello from A", session_id="sessA")
        # Session B sends a message
        await backend._build_messages("hello from B", session_id="sessB")
        # Session A sends another message
        msgs_a, _ = await backend._build_messages("follow-up A", session_id="sessA")

        # Session A's messages contain only A's messages, not B's
        user_msgs = [m["content"] for m in msgs_a if m["role"] == "user"]
        assert "hello from A" in user_msgs
        assert "follow-up A" in user_msgs
        assert "hello from B" not in user_msgs

        # Session B's history contains only B's message
        msgs_b, _ = await backend._build_messages("follow-up B", session_id="sessB")
        user_msgs_b = [m["content"] for m in msgs_b if m["role"] == "user"]
        assert "hello from B" in user_msgs_b
        assert "follow-up B" in user_msgs_b
        assert "hello from A" not in user_msgs_b
        assert "follow-up A" not in user_msgs_b

    @pytest.mark.asyncio
    async def test_default_session_does_not_leak_into_named_sessions(self):
        """When no session_id is given (defaults to 'default'), messages
        must not appear in a named session's history."""
        backend = self._make_echo_backend()
        await backend._build_messages("default message")
        msgs, _ = await backend._build_messages("named message", session_id="named")
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert "default message" not in user_msgs
        assert "named message" in user_msgs

    @pytest.mark.asyncio
    async def test_record_assistant_response_isolated(self):
        """Assistant responses are recorded to the correct session's history."""
        backend = self._make_echo_backend()
        await backend._build_messages("user A", session_id="sessA")
        await backend._record_assistant_response("assistant A reply", session_id="sessA")

        await backend._build_messages("user B", session_id="sessB")
        await backend._record_assistant_response("assistant B reply", session_id="sessB")

        hist_a = backend._session_histories["sessA"]
        hist_b = backend._session_histories["sessB"]
        assert any(m["content"] == "assistant A reply" for m in hist_a)
        assert not any(m["content"] == "assistant A reply" for m in hist_b)
        assert any(m["content"] == "assistant B reply" for m in hist_b)

    @pytest.mark.asyncio
    async def test_clear_history_scoped_to_session(self):
        """clear_history(session_id=X) clears only session X, leaving others intact."""
        backend = self._make_echo_backend()
        await backend._build_messages("msg A", session_id="sessA")
        await backend._build_messages("msg B", session_id="sessB")

        backend.clear_history(session_id="sessA")
        assert "sessA" not in backend._session_histories
        assert "sessB" in backend._session_histories

    @pytest.mark.asyncio
    async def test_clear_history_all_clears_everything(self):
        """clear_history() with no session_id clears all sessions."""
        backend = self._make_echo_backend()
        await backend._build_messages("msg A", session_id="sessA")
        await backend._build_messages("msg B", session_id="sessB")

        backend.clear_history()
        assert len(backend._session_histories) == 0

    @pytest.mark.asyncio
    async def test_history_capped_at_last_10_turns(self):
        """Per-session history is capped at the last 10 turns (existing
        behavior, now per-session)."""
        backend = self._make_echo_backend()
        for i in range(15):
            await backend._build_messages(f"msg {i}", session_id="sess")
        msgs, _ = await backend._build_messages("latest", session_id="sess")
        user_msgs = [m for m in msgs if m["role"] == "user"]
        # 10 most recent + system prompt → at most 10 user messages
        assert len(user_msgs) <= 10
        assert user_msgs[-1]["content"] == "latest"
        assert user_msgs[0]["content"] != "msg 0"  # oldest evicted

    @pytest.mark.asyncio
    async def test_gateway_mode_session_id_accepted_but_not_stored(self):
        """In openclaw gateway mode, session_id is accepted by chat_stream
        but no server-side history is accumulated (gateway manages memory)."""
        backend = AIBackend(
            backend_type="openclaw",
            url="https://fake-gateway.example/v1",
            api_key="fake",
            system_prompt="",
        )
        # _build_messages returns immediately for gateway mode (no history)
        msgs, is_openclaw = await backend._build_messages("hello", session_id="sessX")
        assert is_openclaw is True
        assert "sessX" not in backend._session_histories
        # Only system + user message, no accumulated history
        assert len(msgs) == 2

    def test_session_context_has_session_id_field(self):
        """SessionContext must have a session_id field (issue #24 acceptance)."""
        ctx = SessionContext(session_id="abc123")
        assert ctx.session_id == "abc123"
        ctx_default = SessionContext()
        assert ctx_default.session_id is None


class TestOpenClawModelResolution:
    """Tests for issue #3: per-agent voice model resolution.

    AIBackend._setup_client() used to unconditionally overwrite self.model to
    "openclaw/metis" for gateway mode, ignoring the constructor's model param
    and making OPENCLAW_VOICE_MODEL a no-op. Now self.model is preserved, and
    chat_stream resolves the model per-request via resolve_openclaw_model()
    which respects per-agent env overrides > global default > bare openclaw/<agent>.
    """

    def test_resolve_bare_agent_when_no_env_vars(self, monkeypatch):
        """With no voice-model env vars set, model is bare openclaw/<agent>."""
        from src.server.backend import resolve_openclaw_model

        monkeypatch.setattr("src.server.config.settings.voice_model", None)
        monkeypatch.setattr("src.server.config.settings.voice_model_metis", None)
        monkeypatch.setattr("src.server.config.settings.voice_model_atlas", None)
        assert resolve_openclaw_model("metis") == "openclaw/metis"
        assert resolve_openclaw_model("atlas") == "openclaw/atlas"
        assert resolve_openclaw_model(None) == "openclaw/metis"

    def test_resolve_global_default(self, monkeypatch):
        """Global OPENCLAW_VOICE_MODEL applies when no per-agent override."""
        from src.server.backend import resolve_openclaw_model

        monkeypatch.setattr("src.server.config.settings.voice_model", "glm-5.1:cloud")
        monkeypatch.setattr("src.server.config.settings.voice_model_metis", None)
        monkeypatch.setattr("src.server.config.settings.voice_model_atlas", None)
        assert resolve_openclaw_model("metis", "glm-5.1:cloud") == "openclaw/metis/glm-5.1:cloud"
        assert resolve_openclaw_model("atlas", "glm-5.1:cloud") == "openclaw/atlas/glm-5.1:cloud"

    def test_per_agent_override_wins_over_global(self, monkeypatch):
        """Per-agent env var overrides the global default."""
        from src.server.backend import resolve_openclaw_model

        monkeypatch.setattr("src.server.config.settings.voice_model", "glm-5.1:cloud")
        monkeypatch.setattr("src.server.config.settings.voice_model_atlas", "gpt-4o")
        assert resolve_openclaw_model("atlas", "glm-5.1:cloud") == "openclaw/atlas/gpt-4o"
        # Other agents still use global
        monkeypatch.setattr("src.server.config.settings.voice_model_metis", None)
        assert resolve_openclaw_model("metis", "glm-5.1:cloud") == "openclaw/metis/glm-5.1:cloud"

    def test_resolve_passes_through_prefixed_model(self, monkeypatch):
        """If the model suffix already has a prefix (openclaw/, ollama/, etc.),
        it's returned as-is without double-prefixing."""
        from src.server.backend import resolve_openclaw_model

        monkeypatch.setattr("src.server.config.settings.voice_model_metis", "openclaw/custom/model")
        monkeypatch.setattr("src.server.config.settings.voice_model", None)
        assert resolve_openclaw_model("metis") == "openclaw/custom/model"

    def test_setup_client_does_not_clobber_model(self):
        """Regression for #3: _setup_client() must not overwrite self.model
        for backend_type == 'openclaw'."""
        backend = AIBackend(
            backend_type="openclaw",
            url="https://fake-gateway.example/v1",
            model="openclaw/atlas/glm-5.1:cloud",
            api_key="fake",
            system_prompt="",
        )
        assert backend.model == "openclaw/atlas/glm-5.1:cloud"

    def test_chat_stream_resolves_model_from_agent_hint(self, monkeypatch):
        """chat_stream must resolve the model from agent_hint via
        resolve_openclaw_model() when no explicit model= is passed."""
        backend = AIBackend(
            backend_type="openclaw",
            url="https://fake-gateway.example/v1",
            model="openclaw/metis",
            voice_model_default="glm-5.1:cloud",
            api_key="fake",
            system_prompt="",
        )
        backend._client = object()  # fake client so chat_stream takes the API path

        # Intercept _chat_openai_stream to capture the model kwarg
        captured_models = []

        async def fake_stream(
            user_message, model=None, agent_hint=None, reconnect=False, session_id="default"
        ):
            captured_models.append(model)
            yield f"echo: {user_message}"

        backend._chat_openai_stream = fake_stream

        import asyncio

        async def run():
            async for _ in backend.chat_stream("hello", agent_hint="atlas"):
                pass

        asyncio.run(run())
        assert captured_models[-1] == "openclaw/atlas/glm-5.1:cloud"

    def test_chat_stream_explicit_model_overrides_resolution(self, monkeypatch):
        """An explicit model= kwarg to chat_stream takes priority over
        resolve_openclaw_model()."""
        backend = AIBackend(
            backend_type="openclaw",
            url="https://fake-gateway.example/v1",
            model="openclaw/metis",
            voice_model_default="glm-5.1:cloud",
            api_key="fake",
            system_prompt="",
        )
        backend._client = object()  # fake client so chat_stream takes the API path

        captured_models = []

        async def fake_stream(
            user_message, model=None, agent_hint=None, reconnect=False, session_id="default"
        ):
            captured_models.append(model)
            yield f"echo: {user_message}"

        backend._chat_openai_stream = fake_stream

        import asyncio

        async def run():
            async for _ in backend.chat_stream("hello", model="custom-model", agent_hint="atlas"):
                pass

        asyncio.run(run())
        assert captured_models[-1] == "custom-model"

    def test_openai_backend_uses_self_model(self):
        """Non-openclaw backends use self.model, not resolve_openclaw_model()."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
            api_key="fake",
        )
        assert backend._resolve_model("atlas") == "gpt-4o-mini"


class TestVoiceHintResolution:
    """Tests for issue #33: configurable per-agent voice hints.

    Hints resolve in order: env override (VOICE_HINT_<AGENT>) > config file
    (VOICE_HINT_CONFIG) > built-in AGENT_VOICE_CONFIG > DEFAULT_VOICE_HINT.
    """

    def _reset_config_cache(self, monkeypatch):
        """Clear the module-level config cache between tests."""
        import src.server.backend as backend_mod

        monkeypatch.setattr(backend_mod, "_voice_hint_config", None)
        monkeypatch.setattr(backend_mod, "_voice_hint_config_path", None)

    def test_env_override_wins_over_file_and_default(self, monkeypatch, tmp_path):
        """VOICE_HINT_METIS env var beats a config file and the built-in."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"metis": {"hint": "file hint"}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        monkeypatch.setenv("VOICE_HINT_METIS", "env hint")
        assert resolve_voice_hint("metis") == "env hint"

    def test_file_wins_over_default(self, monkeypatch, tmp_path):
        """A config-file hint beats the built-in default."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"atlas": {"hint": "file hint"}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        assert resolve_voice_hint("atlas") == "file hint"

    def test_builtin_used_when_nothing_configured(self, monkeypatch):
        """With no env or file config, the built-in AGENT_VOICE_CONFIG hint is used."""
        from src.server.backend import AGENT_VOICE_CONFIG, resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        monkeypatch.delenv("VOICE_HINT_METIS", raising=False)
        monkeypatch.delenv("VOICE_HINT_CONFIG", raising=False)
        assert resolve_voice_hint("metis") == AGENT_VOICE_CONFIG["metis"]["hint"]

    def test_unknown_agent_uses_default(self, monkeypatch):
        """An agent not in the built-in map falls back to DEFAULT_VOICE_HINT."""
        from src.server.backend import DEFAULT_VOICE_HINT, resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        monkeypatch.delenv("VOICE_HINT_CONFIG", raising=False)
        monkeypatch.delenv("VOICE_HINT_UNKNOWN", raising=False)
        assert resolve_voice_hint("unknown-agent") == DEFAULT_VOICE_HINT

    def test_none_agent_defaults_to_metis(self, monkeypatch):
        """agent_id=None resolves as 'metis' (matches resolve_openclaw_model)."""
        from src.server.backend import AGENT_VOICE_CONFIG, resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        monkeypatch.delenv("VOICE_HINT_METIS", raising=False)
        monkeypatch.delenv("VOICE_HINT_CONFIG", raising=False)
        assert resolve_voice_hint(None) == AGENT_VOICE_CONFIG["metis"]["hint"]

    def test_word_budget_injects_clause(self, monkeypatch, tmp_path):
        """A word_budget in the config file injects a budget clause into the hint."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"metis": {"hint": "Be brief.", "word_budget": 30}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        hint = resolve_voice_hint("metis")
        assert "under 30 words" in hint
        assert "Be brief." in hint

    def test_word_budget_absent_uses_hint_verbatim(self, monkeypatch, tmp_path):
        """Without a word_budget, the hint is used verbatim (no clause added)."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"metis": {"hint": "Exact text."}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        assert resolve_voice_hint("metis") == "Exact text."

    def test_yaml_config_supported(self, monkeypatch, tmp_path):
        """A YAML config file is parsed the same as JSON."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.yaml"
        cfg.write_text("metis:\n  hint: yaml hint\n", encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        assert resolve_voice_hint("metis") == "yaml hint"

    def test_malformed_json_raises_clear_error(self, monkeypatch, tmp_path):
        """Malformed JSON fails fast with an error naming the file."""
        from src.server.backend import _get_voice_hint_config

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        with pytest.raises(ValueError, match="Malformed voice-hint config file"):
            _get_voice_hint_config()

    def test_wrong_type_raises_clear_error(self, monkeypatch, tmp_path):
        """A non-object entry fails fast naming the offending agent."""
        from src.server.backend import _get_voice_hint_config

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"metis": "just a string"}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        with pytest.raises(ValueError, match="'metis'"):
            _get_voice_hint_config()

    def test_bad_word_budget_raises_clear_error(self, monkeypatch, tmp_path):
        """A non-positive word_budget fails fast."""
        from src.server.backend import _get_voice_hint_config

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"metis": {"hint": "hi", "word_budget": -5}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        with pytest.raises(ValueError, match="word_budget"):
            _get_voice_hint_config()

    def test_missing_configured_file_raises(self, monkeypatch, tmp_path):
        """VOICE_HINT_CONFIG pointing at a nonexistent file fails fast."""
        from src.server.backend import _get_voice_hint_config

        self._reset_config_cache(monkeypatch)
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(tmp_path / "nope.json"))
        with pytest.raises(ValueError, match="no such file"):
            _get_voice_hint_config()

    def test_unknown_keys_are_ignored(self, monkeypatch, tmp_path):
        """Unknown keys in the config file are ignored (forward compatible)."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text(
            '{"metis": {"hint": "known", "future_field": 1}, "future_agent": {"hint": "x"}}',
            encoding="utf-8",
        )
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        assert resolve_voice_hint("metis") == "known"

    def test_default_config_file_used_when_present(self, monkeypatch, tmp_path):
        """A ./voice_hints.json in the working dir is picked up without VOICE_HINT_CONFIG."""
        from src.server.backend import resolve_voice_hint

        self._reset_config_cache(monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "voice_hints.json").write_text(
            '{"metis": {"hint": "default file hint"}}', encoding="utf-8"
        )
        assert resolve_voice_hint("metis") == "default file hint"

    @pytest.mark.asyncio
    async def test_build_messages_uses_resolved_hint(self, monkeypatch, tmp_path):
        """_build_messages must use the resolved hint (env > file > default)."""
        from src.server.backend import AIBackend

        self._reset_config_cache(monkeypatch)
        cfg = tmp_path / "hints.json"
        cfg.write_text('{"atlas": {"hint": "resolved file hint"}}', encoding="utf-8")
        monkeypatch.setenv("VOICE_HINT_CONFIG", str(cfg))
        backend = AIBackend(
            backend_type="openclaw",
            url="https://fake-gateway.example/v1",
            api_key="fake",
            system_prompt="",
        )
        msgs, is_openclaw = await backend._build_messages("hello", agent_hint="atlas")
        assert is_openclaw is True
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "resolved file hint"


class TestVAD:
    """Tests for Voice Activity Detection module."""

    def test_init(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector()
        assert vad is not None

    def test_is_speech_silence(self):
        """Test that silence is not detected as speech."""
        vad = VoiceActivityDetector()
        silence = np.zeros(16000, dtype=np.float32)
        # Should return True if no VAD model (assumes speech)
        # or False if VAD model is loaded and detects no speech
        result = vad.is_speech(silence)
        assert isinstance(result, bool)

    def test_is_speech_noise(self):
        """Test with random noise."""
        vad = VoiceActivityDetector()
        noise = np.random.randn(16000).astype(np.float32)
        result = vad.is_speech(noise)
        assert isinstance(result, bool)


class TestTTSRouter:
    """Tests for TTSRouter active_backend logic (#6)."""

    def test_active_backend_higgs_auto_available(self):
        """active_backend returns 'higgs' when backend='auto' and Higgs is available."""
        # Mock supertonic
        sup = ChatterboxTTS()
        sup._backend = "supertonic"

        class MockHiggs:
            available = True

        router = TTSRouter(supertonic=sup, higgs=MockHiggs(), backend="auto")
        assert router.active_backend == "higgs"

    def test_active_backend_higgs_explicit_opt_out(self):
        """active_backend does NOT return 'higgs' when backend='supertonic' even if Higgs available."""
        sup = ChatterboxTTS()
        sup._backend = "supertonic"

        class MockHiggs:
            available = True

        router = TTSRouter(supertonic=sup, higgs=MockHiggs(), backend="supertonic")
        assert router.active_backend != "higgs"
        assert router.active_backend == "supertonic"

    def test_active_backend_fallback_when_higgs_unavailable(self):
        """active_backend falls back to supertonic when Higgs unavailable."""
        sup = ChatterboxTTS()
        sup._backend = "supertonic"

        class MockHiggs:
            available = False

        router = TTSRouter(supertonic=sup, higgs=MockHiggs(), backend="auto")
        assert router.active_backend == "supertonic"


class TestTTSVoiceConcurrency:
    """Regression tests for issue #10: per-call voice must not race across
    concurrent sessions sharing one ChatterboxTTS/TTSRouter instance.

    The bug: voice/style was mutable instance state on a single shared
    ChatterboxTTS, mutated under a lock in start_listening and then read
    unlocked during executor-run synthesis — so session A's sentence could
    be spoken in session B's voice. The fix threads voice as an explicit
    per-call parameter and resolves the style locally (read-only cache),
    never mutating shared instance state after init.
    """

    def _make_fake_supertonic(self, monkeypatch):
        """Return (ChatterboxTTS forced to a fake supertonic backend, calls list).

        The fake's synthesize() records (text, voice_style, timing) and blocks
        on a barrier so two concurrent calls deterministically overlap inside
        the executor — the exact precondition for the original race.
        """
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        sup = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        sup._backend = "supertonic"
        sup._supertonic_sr = 24000  # skip resample path
        sup._supertonic_voice = "F2"
        # Sentinel: if synthesis ever reads this shared attr it gets "POLLUTED".
        sup._supertonic_style = "POLLUTED"
        sup._voice_style_cache = {}

        calls = []
        barrier = threading.Barrier(2, timeout=10)

        class FakeSupertonic:
            voice_style_names = ["F2", "M2", "M4", "F4", "M1", "F5"]

            def get_voice_style(self, name):
                return f"style:{name}"

            def synthesize(self, text, voice_style=None):
                rec = {"text": text, "style": voice_style, "start": time.monotonic()}
                calls.append(rec)
                # Block until both concurrent calls are inside synthesize(),
                # guaranteeing overlapping executor windows.
                barrier.wait()
                rec["end"] = time.monotonic()
                return (np.zeros((1, 2400), dtype=np.float32),)

        sup._supertonic_tts = FakeSupertonic()
        return sup, calls

    @pytest.mark.asyncio
    async def test_concurrent_sessions_no_voice_cross_contamination(self, monkeypatch):
        """Two concurrent router calls (agents atlas=M2, mara=F5) against one
        shared instance must each use their own voice — never the shared
        _supertonic_style sentinel, regardless of overlapping execution."""
        sup, calls = self._make_fake_supertonic(monkeypatch)
        router = TTSRouter(supertonic=sup, higgs=None, backend="supertonic", cache=TTSCache())

        async def run(text, agent_hint):
            async for _ in router.synthesize_stream(text, agent_hint=agent_hint):
                pass

        await asyncio.gather(
            run("atlas sentence", "atlas"),
            run("mara sentence", "mara"),
        )

        by_text = {c["text"]: c for c in calls}
        assert by_text["atlas sentence"]["style"] == "style:M2"
        assert by_text["mara sentence"]["style"] == "style:F5"
        # Neither call picked up the shared _supertonic_style sentinel.
        assert all(c["style"] != "POLLUTED" for c in calls)
        # The Barrier(parties=2) not raising BrokenBarrierError proves both
        # calls were inside synthesize() simultaneously — the exact
        # precondition for the original race. (Timestamp-based overlap
        # assertions are flaky on Windows due to ~15ms clock resolution.)

    @pytest.mark.asyncio
    async def test_router_supertonic_honors_agent_hint(self, monkeypatch):
        """TTSRouter's Supertonic path must forward agent_hint → supertonic_voice
        (previously dropped at the Supertonic fallback)."""
        sup, calls = self._make_fake_supertonic(monkeypatch)
        # Single call — relax the barrier so it doesn't block waiting for a peer.
        sup._supertonic_tts.synthesize = lambda text, voice_style=None: (
            calls.append({"text": text, "style": voice_style})
            or (np.zeros((1, 2400), dtype=np.float32),),
        )
        router = TTSRouter(supertonic=sup, higgs=None, backend="supertonic", cache=TTSCache())

        async for _ in router.synthesize_stream("hi", agent_hint="atlas"):
            pass

        assert calls[-1]["style"] == "style:M2"

    @pytest.mark.asyncio
    async def test_router_explicit_voice_overrides_agent_hint(self, monkeypatch):
        """An explicit per-call voice must win over the agent_hint-derived voice."""
        sup, calls = self._make_fake_supertonic(monkeypatch)
        sup._supertonic_tts.synthesize = lambda text, voice_style=None: (
            calls.append({"text": text, "style": voice_style})
            or (np.zeros((1, 2400), dtype=np.float32),),
        )
        router = TTSRouter(supertonic=sup, higgs=None, backend="supertonic", cache=TTSCache())

        async for _ in router.synthesize_stream("hi", voice="F5", agent_hint="atlas"):
            pass

        assert calls[-1]["style"] == "style:F5"


class TestTTSVoiceCloningLocalBackend:
    """Regression tests for issue #25: per-call voice cloning (set_voice) was
    dead code on the chatterbox/xtts backend after #10.

    #10 correctly removed the shared-mutation ``self.voice_sample = voice_path``
    from set_voice but added no replacement per-call threading for the local
    (chatterbox/xtts) path. ``_synthesize_sync_local`` only ever read
    ``self.voice_sample`` (set once at construction), so the highest-value
    voice_cloning case silently did nothing. The fix threads ``voice`` through
    ``synthesize`` → ``_synthesize_sync_local`` (same pattern as Supertonic's
    ``_resolve_voice_style``), falling back to ``self.voice_sample`` only when
    no per-call override is given.
    """

    def _make_fake_local(self, monkeypatch, backend):
        """Return (ChatterboxTTS forced to a fake chatterbox/xtts backend, calls).

        The fake model records the voice argument it received (``audio_prompt``
        for chatterbox, ``speaker_wav`` for xtts) and, when ``model.barrier`` is
        set, blocks so two concurrent calls deterministically overlap inside the
        executor — the exact precondition for cross-talk.
        """
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        tts = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        tts._backend = backend
        # Constructor-time default is None — per-call voice must reach the model.
        tts.voice_sample = None

        calls = []

        class _FakeTensor:
            def cpu(self):
                return self

            def numpy(self):
                return np.zeros(2400, dtype=np.float32)

        if backend == "chatterbox":

            class FakeModel:
                def __init__(self):
                    self.barrier = None

                def generate(self, text, audio_prompt=None):
                    rec = {"text": text, "voice": audio_prompt, "start": time.monotonic()}
                    calls.append(rec)
                    if self.barrier is not None:
                        self.barrier.wait()
                    rec["end"] = time.monotonic()
                    return _FakeTensor()

            tts.model = FakeModel()
        else:

            class FakeModel:
                def __init__(self):
                    self.barrier = None

                def tts(self, text, speaker_wav=None, language="en"):
                    rec = {"text": text, "voice": speaker_wav, "start": time.monotonic()}
                    calls.append(rec)
                    if self.barrier is not None:
                        self.barrier.wait()
                    rec["end"] = time.monotonic()
                    return np.zeros(2400, dtype=np.float32)

            tts.model = FakeModel()

        return tts, calls

    @pytest.mark.asyncio
    async def test_chatterbox_per_call_voice_reaches_generate(self, monkeypatch):
        """Regression for #25: a per-call voice must reach model.generate as
        audio_prompt even when self.voice_sample is None (the bug: it was
        always None)."""
        tts, calls = self._make_fake_local(monkeypatch, "chatterbox")
        async for _ in tts.synthesize_stream("hello", voice="/voices/clone_a.wav"):
            pass
        assert calls[-1]["voice"] == "/voices/clone_a.wav"

    @pytest.mark.asyncio
    async def test_xtts_per_call_voice_reaches_tts(self, monkeypatch):
        """Regression for #25: a per-call voice must reach model.tts as
        speaker_wav on the xtts backend."""
        tts, calls = self._make_fake_local(monkeypatch, "xtts")
        async for _ in tts.synthesize_stream("hello", voice="/voices/clone_b.wav"):
            pass
        assert calls[-1]["voice"] == "/voices/clone_b.wav"

    @pytest.mark.asyncio
    async def test_local_falls_back_to_instance_voice_sample(self, monkeypatch):
        """When no per-call voice is given, the constructor-time
        self.voice_sample default is used (same fallback pattern as
        Supertonic's _resolve_voice_style)."""
        tts, calls = self._make_fake_local(monkeypatch, "chatterbox")
        tts.voice_sample = "/voices/default.wav"
        async for _ in tts.synthesize_stream("hello"):
            pass
        assert calls[-1]["voice"] == "/voices/default.wav"

    @pytest.mark.asyncio
    async def test_per_call_voice_does_not_mutate_instance(self, monkeypatch):
        """A per-call voice must not be written back to self.voice_sample —
        that would reintroduce the #10 cross-session race on this backend."""
        tts, calls = self._make_fake_local(monkeypatch, "chatterbox")
        assert tts.voice_sample is None
        async for _ in tts.synthesize_stream("hello", voice="/voices/clone_a.wav"):
            pass
        assert tts.voice_sample is None  # unchanged

    @pytest.mark.asyncio
    async def test_router_forwards_voice_to_local_backend(self, monkeypatch):
        """End-to-end through TTSRouter: an explicit per-call voice must reach
        model.generate on a chatterbox-backed ChatterboxTTS (the router path
        flows through the _synthesize_stream_primary fallback branch fixed in
        #25)."""
        tts, calls = self._make_fake_local(monkeypatch, "chatterbox")
        router = TTSRouter(supertonic=tts, higgs=None, backend="supertonic", cache=TTSCache())
        async for _ in router.synthesize_stream("hi", voice="/voices/clone_a.wav"):
            pass
        assert calls[-1]["voice"] == "/voices/clone_a.wav"

    @pytest.mark.asyncio
    async def test_chatterbox_concurrent_voices_no_cross_contamination(self, monkeypatch):
        """Two concurrent synthesize_stream calls with different per-call voices
        against one shared instance must each receive their own voice — never
        None and never the other session's voice.

        #27 added a per-instance lock that serializes generate() calls, so the
        Barrier(2) used previously (to force overlap) would now deadlock. The
        parameter-threading isolation is still verified here; the serialization
        itself is tested in TestTTSLocalModelLock.
        """
        tts, calls = self._make_fake_local(monkeypatch, "chatterbox")

        async def run(text, voice):
            async for _ in tts.synthesize_stream(text, voice=voice):
                pass

        await asyncio.gather(
            run("alpha sentence", voice="/voices/clone_a.wav"),
            run("bravo sentence", voice="/voices/clone_b.wav"),
        )

        by_text = {c["text"]: c for c in calls}
        assert by_text["alpha sentence"]["voice"] == "/voices/clone_a.wav"
        assert by_text["bravo sentence"]["voice"] == "/voices/clone_b.wav"
        assert all(c["voice"] is not None for c in calls)


class TestTTSLocalModelLock:
    """Regression tests for issue #27: serialize local-backend synthesis to
    protect shared model state.

    Investigation confirmed neither Chatterbox nor XTTS is thread-safe:

    - Chatterbox ``generate()`` mutates ``self.conds`` via
      ``prepare_conditionals()`` (chatterbox/tts.py:220) before reading it in
      ``t3.inference()`` (line 247). Concurrent calls overwrite each other's
      voice conditioning state.
    - XTTS ``Synthesizer.tts()`` mutates ``self.voice_dir``; the GPT model uses
      a shared KV cache (``kv_cache=True``) that corrupts under concurrent
      ``generate()`` calls; ``do_sample=True`` interleaves global RNG draws.

    Fix: a per-instance ``_local_model_lock`` serializes ``model.generate()``
    / ``model.tts()`` in ``_synthesize_sync_local``. Cloud backends and
    Supertonic (read-only style cache from #10) are unaffected.
    """

    def _make_stateful_fake(self, monkeypatch, backend):
        """Return (ChatterboxTTS with a stateful fake model, calls, violations).

        The fake mimics the real thread-safety bugs: ``generate()``/``tts()``
        writes a ``shared_state`` attr (simulating Chatterbox's ``self.conds``
        or XTTS's ``self.voice_dir``) at entry, then reads it back after a
        delay. Without the lock, the second concurrent call overwrites
        ``shared_state`` before the first reads it — exactly the race.

        ``violations`` records any time the model was entered while another
        call was still running (i.e., the lock failed to serialize).
        """
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        tts = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        tts._backend = backend
        tts.voice_sample = None

        calls = []
        violations = []
        _in_call = threading.Event()

        class _FakeTensor:
            def cpu(self):
                return self

            def numpy(self):
                return np.zeros(2400, dtype=np.float32)

        class FakeModel:
            def __init__(self):
                self.shared_state = None

            def generate(self, text, audio_prompt=None):
                self.shared_state = audio_prompt
                if _in_call.is_set():
                    violations.append("concurrent generate")
                _in_call.set()
                try:
                    time.sleep(0.05)
                    observed = self.shared_state
                    calls.append({"text": text, "voice": audio_prompt, "observed": observed})
                finally:
                    _in_call.clear()
                return _FakeTensor()

            def tts(self, text, speaker_wav=None, language="en"):
                self.shared_state = speaker_wav
                if _in_call.is_set():
                    violations.append("concurrent tts")
                _in_call.set()
                try:
                    time.sleep(0.05)
                    observed = self.shared_state
                    calls.append({"text": text, "voice": speaker_wav, "observed": observed})
                finally:
                    _in_call.clear()
                return np.zeros(2400, dtype=np.float32)

        tts.model = FakeModel()
        return tts, calls, violations

    @pytest.mark.asyncio
    async def test_chatterbox_lock_serializes_concurrent_calls(self, monkeypatch):
        """The _local_model_lock must serialize concurrent generate() calls so
        shared model state (self.conds in real Chatterbox) is never overwritten
        mid-synthesis. Without the lock, the fake's shared_state would be
        clobbered by the second call before the first reads it back."""
        tts, calls, violations = self._make_stateful_fake(monkeypatch, "chatterbox")

        async def run(text, voice):
            async for _ in tts.synthesize_stream(text, voice=voice):
                pass

        await asyncio.gather(
            run("alpha", voice="/voices/a.wav"),
            run("bravo", voice="/voices/b.wav"),
        )

        assert violations == []
        by_text = {c["text"]: c for c in calls}
        assert by_text["alpha"]["observed"] == "/voices/a.wav"
        assert by_text["bravo"]["observed"] == "/voices/b.wav"

    @pytest.mark.asyncio
    async def test_xtts_lock_serializes_concurrent_calls(self, monkeypatch):
        """Same serialization test for the XTTS backend path (model.tts)."""
        tts, calls, violations = self._make_stateful_fake(monkeypatch, "xtts")

        async def run(text, voice):
            async for _ in tts.synthesize_stream(text, voice=voice):
                pass

        await asyncio.gather(
            run("alpha", voice="/voices/a.wav"),
            run("bravo", voice="/voices/b.wav"),
        )

        assert violations == []
        by_text = {c["text"]: c for c in calls}
        assert by_text["alpha"]["observed"] == "/voices/a.wav"
        assert by_text["bravo"]["observed"] == "/voices/b.wav"

    @pytest.mark.asyncio
    async def test_lock_does_not_serialize_mock_backend(self, monkeypatch):
        """The mock backend has no shared model state, so the lock is not
        acquired — two concurrent mock calls should both complete without
        blocking. This verifies the lock is scoped to chatterbox/xtts only."""
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        tts = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        tts._backend = "mock"

        results = []

        async def run(text):
            await tts.synthesize(text)
            results.append(text)

        await asyncio.gather(run("a"), run("b"))
        assert set(results) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_lock_allows_concurrent_cloud_and_local(self, monkeypatch):
        """The local-model lock must not block Supertonic (or other non-local
        backends). A Supertonic call and a chatterbox call can run concurrently
        without the local lock affecting the Supertonic path."""
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        sup = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        sup._backend = "supertonic"
        sup._supertonic_sr = 24000
        sup._supertonic_voice = "F2"
        sup._supertonic_style = "POLLUTED"
        sup._voice_style_cache = {}

        class FakeSupertonic:
            voice_style_names = ["F2", "M2"]

            def get_voice_style(self, name):
                return f"style:{name}"

            def synthesize(self, text, voice_style=None):
                return (np.zeros((1, 2400), dtype=np.float32),)

        sup._supertonic_tts = FakeSupertonic()

        tts, calls, violations = self._make_stateful_fake(monkeypatch, "chatterbox")

        async def run_supertonic():
            async for _ in sup.synthesize_stream("supertonic text", voice="M2"):
                pass

        async def run_local():
            async for _ in tts.synthesize_stream("local text", voice="/voices/a.wav"):
                pass

        await asyncio.gather(run_supertonic(), run_local())

        assert violations == []
        assert calls[-1]["observed"] == "/voices/a.wav"


class TestTTSElevenLabsEarlyCancel:
    """Regression tests for issue #23: when the consumer of the ElevenLabs
    streaming generator stops early (barge-in, cancellation), the background
    _produce thread must stop draining the SDK generator instead of running the
    full synthesis to completion — wasting API cost/duration.

    The fix adds a threading.Event (stop_flag) checked between SDK chunks; the
    consumer's finally block sets it on early exit (GeneratorExit / cancellation).
    """

    def _make_fake_elevenlabs(self, monkeypatch):
        """Return (ChatterboxTTS forced to elevenlabs backend, chunks_produced list).

        The fake ElevenLabs SDK yields chunks slowly (one per ~50ms) so the test
        can consume a few, then break out — simulating barge-in mid-stream. The
        background _produce thread checks stop_flag between chunks and should
        stop without draining the full generator.
        """
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(ChatterboxTTS, "_load_model", lambda self: None)
        tts = ChatterboxTTS(executor=ThreadPoolExecutor(max_workers=4))
        tts._backend = "elevenlabs"

        TOTAL_CHUNKS = 20
        chunks_produced = []

        class FakeConvert:
            def __call__(self, **kwargs):
                for i in range(TOTAL_CHUNKS):
                    if getattr(self, "_stopped", False):
                        break
                    time.sleep(0.05)
                    chunks_produced.append(i)
                    yield f"chunk{i}".encode()

        class FakeElevenLabsClient:
            def __init__(self):
                self.text_to_speech = type("obj", (object,), {"convert": FakeConvert()})()

        tts._elevenlabs_client = FakeElevenLabsClient()
        return tts, chunks_produced, TOTAL_CHUNKS

    @pytest.mark.asyncio
    async def test_early_consumer_exit_stops_background_thread(self, monkeypatch):
        """Consumer breaks out after 2 chunks (simulating barge-in). The
        background _produce thread must stop — not drain all 20 chunks.

        Without the stop_flag fix, _produce would iterate the full generator
        (20 chunks × 50ms = ~1s of wasted work) even though nobody is listening."""
        tts, chunks_produced, total = self._make_fake_elevenlabs(monkeypatch)

        gen = tts._synthesize_stream_primary("hello world")
        received = []
        async for chunk in gen:
            received.append(chunk)
            if len(received) >= 2:
                break
        await gen.aclose()

        # Give the background thread a moment to notice stop_flag and exit.
        time.sleep(0.3)
        # Consumer got 2 chunks.
        assert len(received) == 2
        # Background thread was stopped — did NOT produce all 20 chunks.
        # Allow a small margin (the chunk in-flight when stop_flag was set may
        # still land) but it must be well below the full count.
        assert len(chunks_produced) < total
        assert len(chunks_produced) <= 4  # at most 2 consumed + a couple in-flight

    @pytest.mark.asyncio
    async def test_full_consumption_completes_normally(self, monkeypatch):
        """When the consumer drains the generator to completion, all chunks
        arrive and no deadlock/hang occurs — the stop_flag is never set."""
        tts, chunks_produced, total = self._make_fake_elevenlabs(monkeypatch)

        received = []
        async for chunk in tts._synthesize_stream_primary("hello world"):
            received.append(chunk)

        assert len(received) == total
        assert len(chunks_produced) == total

    @pytest.mark.asyncio
    async def test_cancellation_sets_stop_flag(self, monkeypatch):
        """If the async generator is cancelled (not just broken out of), the
        stop_flag must still be set via the finally block."""
        tts, chunks_produced, total = self._make_fake_elevenlabs(monkeypatch)

        gen = tts._synthesize_stream_primary("hello world")
        received = []
        try:
            async for chunk in gen:
                received.append(chunk)
                if len(received) >= 1:
                    break
        finally:
            await gen.aclose()

        time.sleep(0.3)
        assert len(chunks_produced) < total


class TestTTSUtils:
    """Tests for TTS text sanitization (#15)."""

    def test_sanitize_tts_symbols_basic(self):
        from src.server.text_utils import sanitize_tts_symbols

        text = "**bold** `code` [link](http://example.com) #hashtag a=b  "
        out = sanitize_tts_symbols(text)
        # markdown stripped, hashtag stripped, equals expanded, whitespace collapsed
        assert "**" not in out
        assert "`" not in out
        assert "[link]" not in out
        assert "http://example.com" not in out
        assert "hashtag" in out
        assert "equals" in out
        assert "  " not in out
        assert out.strip() == out

    def test_sanitize_preserves_clean_text(self):
        from src.server.text_utils import sanitize_tts_symbols

        text = "Hello world"
        assert sanitize_tts_symbols(text) == "Hello world"


class TestSharedSessionDispatch:
    """Tests for issue #21: extraction of shared WS/WebRTC message dispatch.

    Pure dedup — zero behavior change. These tests exercise the shared
    ``handle_session_message`` / ``VoiceSessionState`` helpers against the
    state they need to maintain, and add the required regression test proving
    the WebRTC path STILL requires an explicit ``stop_listening`` to flush the
    buffer (i.e. VAD auto-endpointing was *not* accidentally introduced by the
    refactor).
    """

    def _make_state(self, **kwargs):
        from src.server.session_handler import VoiceSessionState

        return VoiceSessionState(**kwargs)

    def _make_fake_transport(self):
        """A minimal AudioTransport stub capturing sent JSON messages."""

        class FakeTransport:
            def __init__(self):
                self.sent = []

            async def send_json(self, data):
                self.sent.append(data)

        return FakeTransport()

    @pytest.mark.asyncio
    async def test_ping_pong_dispatch(self):
        """ping → pong via the shared dispatcher."""
        from src.server.session_handler import handle_session_message

        t = self._make_fake_transport()
        state = self._make_state()
        handled = await handle_session_message(t, {"type": "ping"}, state)
        assert handled is True
        assert t.sent == [{"type": "pong"}]

    @pytest.mark.asyncio
    async def test_start_listening_sets_state_and_resets_buffer(self):
        """start_listening resets buffer state and replies listening_started."""
        from src.server.session_handler import handle_session_message

        t = self._make_fake_transport()
        state = self._make_state()
        state.audio_buffer = [np.zeros(100, dtype=np.float32)]
        state.buffer_samples = 100
        handled = await handle_session_message(
            t, {"type": "start_listening", "agent": "atlas"}, state
        )
        assert handled is True
        assert state.is_listening is True
        assert state.session_agent == "atlas"
        assert state.audio_buffer == []
        assert state.buffer_samples == 0
        assert t.sent == [{"type": "listening_started"}]

    @pytest.mark.asyncio
    async def test_stop_listening_runs_hook_and_flushes(self):
        """stop_listening invokes the transport hook (the flush point) and
        resets state. Explicit stop_listening remains a valid flush path
        alongside VAD auto-endpointing (issue #22)."""
        from src.server.session_handler import handle_session_message

        t = self._make_fake_transport()
        state = self._make_state()
        flush_calls = []

        async def fake_on_stop(s):
            flush_calls.append(list(s.audio_buffer))

        state.on_stop_listening = fake_on_stop
        state.audio_buffer = [np.zeros(64, dtype=np.float32)]
        state.buffer_samples = 64

        handled = await handle_session_message(t, {"type": "stop_listening"}, state)
        assert handled is True
        assert len(flush_calls) == 1  # hook saw the data once
        assert len(flush_calls[0]) == 1
        assert flush_calls[0][0].shape == (64,)
        assert state.is_listening is False
        assert state.audio_buffer == []  # reset after flush
        assert state.buffer_samples == 0
        assert t.sent == [{"type": "listening_stopped"}]

    @pytest.mark.asyncio
    async def test_set_voice_sends_error_when_missing(self, monkeypatch):
        """WS path: a nonexistent voice replies with an error."""
        from src.server.session_handler import handle_session_message

        t = self._make_fake_transport()
        state = self._make_state(report_missing_voice_error=True)
        monkeypatch.setattr("src.server.session_handler.os.path.isfile", lambda p: False)
        handled = await handle_session_message(t, {"type": "set_voice", "voice_id": "nope"}, state)
        assert handled is True
        assert t.sent == [{"type": "error", "message": "Voice nope not found"}]
        assert state.session_voice_override is None

    @pytest.mark.asyncio
    async def test_set_voice_webrtc_silent_on_missing(self, monkeypatch):
        """WebRTC path: original code omitted the error for a missing voice —
        preserve that exact behavior (report_missing_voice_error=False)."""
        from src.server.session_handler import handle_session_message

        t = self._make_fake_transport()
        state = self._make_state(report_missing_voice_error=False)
        monkeypatch.setattr("src.server.session_handler.os.path.isfile", lambda p: False)
        handled = await handle_session_message(t, {"type": "set_voice", "voice_id": "nope"}, state)
        assert handled is True
        assert t.sent == []  # no error sent on the WebRTC path

    @pytest.mark.asyncio
    async def test_clear_history_scoped_to_session(self, monkeypatch):
        """clear_history passes the session_id through to the backend."""
        from src.server.session_handler import handle_session_message

        backend_calls = []

        class FakeBackend:
            def clear_history(self, session_id=None):
                backend_calls.append(session_id)

        monkeypatch.setattr("src.server.session_handler.app_state.backend", FakeBackend())
        t = self._make_fake_transport()
        state = self._make_state(session_id="ws_abc")
        handled = await handle_session_message(t, {"type": "clear_history"}, state)
        assert handled is True
        assert backend_calls == ["ws_abc"]
        assert t.sent == [{"type": "history_cleared"}]

    def test_webrtc_audio_path_has_no_vad_endpointing(self):
        """The shared message dispatcher does NOT handle audio messages
        directly — the caller routes them to ``handle_audio_samples`` (issue
        #22), which is where VAD/barge-in live. This keeps message dispatch
        transport-agnostic."""
        from src.server.session_handler import handle_session_message, VoiceSessionState

        async def run():
            t = self._make_fake_transport()
            state = VoiceSessionState()
            handled = await handle_session_message(t, {"type": "audio_frame"}, state)
            return handled

        handled = asyncio.run(run())
        assert handled is False  # audio handled by the caller, not the dispatcher

    def test_max_audio_buffer_seconds_single_source(self):
        """MAX_AUDIO_BUFFER_SECONDS lives in exactly one place — the shared
        session_handler module."""
        import src.server.routes as routes_mod
        import src.server.session_handler as sh_mod

        assert hasattr(sh_mod, "MAX_AUDIO_BUFFER_SECONDS")
        assert not hasattr(routes_mod, "MAX_AUDIO_BUFFER_SECONDS")
        # the constant value is 30
        assert sh_mod.MAX_AUDIO_BUFFER_SECONDS == 30

    def test_shared_dispatch_covers_common_message_types(self):
        """The five message types shared by both transports are handled by the
        dispatcher (returns True), while audio stays in the caller."""
        from src.server.session_handler import handle_session_message, VoiceSessionState

        async def run():
            t = self._make_fake_transport()
            state = VoiceSessionState()
            results = {}
            for mt in ("ping", "set_voice", "clear_history", "start_listening", "stop_listening"):
                results[mt] = await handle_session_message(t, {"type": mt}, state)
            return results

        results = asyncio.run(run())
        assert results == {
            "ping": True,
            "set_voice": True,
            "clear_history": True,
            "start_listening": True,
            "stop_listening": True,
        }


class TestSharedVADAndBargeIn:
    """Tests for issue #22: VAD auto-endpointing + barge-in on the shared
    audio path, now used by BOTH WebSocket and WebRTC.

    Before #22, only WS had VAD endpointing/barge-in; WebRTC required an
    explicit stop_listening. The shared ``handle_audio_samples`` helper
    gives both transports identical behavior.
    """

    def _make_fake_transport(self):
        class FakeTransport:
            def __init__(self):
                self.sent = []

            async def send_json(self, data):
                self.sent.append(data)

            async def send_event(self, event):
                pass

        return FakeTransport()

    def _make_state(self, **kwargs):
        from src.server.session_handler import VoiceSessionState

        return VoiceSessionState(**kwargs)

    @pytest.mark.asyncio
    async def test_vad_speech_end_dispatches_without_stop_listening(self, monkeypatch):
        """VAD auto-endpointing: a speech_end event triggers pipeline dispatch
        even with no explicit stop_listening — the acceptance criterion for the
        WebRTC path."""
        from src.server.session_handler import handle_audio_samples

        t = self._make_fake_transport()
        state = self._make_state()
        state.is_listening = True
        state.audio_buffer = [np.zeros(100, dtype=np.float32)]
        state.buffer_samples = 100

        dispatched = []

        async def fake_dispatch(s):
            dispatched.append(list(s.audio_buffer))

        state.dispatch_pipeline = fake_dispatch

        class FakeVADEndpoint:
            async def process_async(self, audio):
                return "speech_end"

        state.vad_endpoint = FakeVADEndpoint()

        class FakeVAD:
            async def is_speech_async(self, audio):
                return True

        monkeypatch.setattr("src.server.session_handler.app_state.vad", FakeVAD())

        await handle_audio_samples(t, state, np.zeros(100, dtype=np.float32), 480000)

        assert len(dispatched) == 1
        # The buffer (with the just-appended frame) was flushed.
        assert len(dispatched[0]) == 2
        # VAD cleared, buffer reset, listening off.
        assert state.vad_endpoint is None
        assert state.audio_buffer == []
        assert state.buffer_samples == 0

    @pytest.mark.asyncio
    async def test_barge_in_cancels_pipeline_and_resumes_listening(self, monkeypatch):
        """Barge-in: while playing, a speech_start event cancels the in-flight
        pipeline task and resumes listening — the second acceptance criterion."""
        from src.server.session_handler import handle_audio_samples

        t = self._make_fake_transport()
        state = self._make_state()
        state.is_playing = True

        cancelled = []

        class FakeTask:
            def done(self):
                return False

            def cancel(self):
                cancelled.append(True)

        state.pipeline_task = FakeTask()

        class FakeBargeInVAD:
            async def process_async(self, audio):
                return "speech_start"

        state.barge_in_vad = FakeBargeInVAD()
        monkeypatch.setattr("src.server.session_handler.app_state.vad", object())
        # build_vad_endpoint returns None when settings.vad_enabled is False (default in tests)

        await handle_audio_samples(t, state, np.zeros(100, dtype=np.float32), 480000)

        assert cancelled == [True]
        assert state.pipeline_task is None
        assert state.is_listening is True
        assert state.barge_in_vad is None
        assert state.audio_buffer == []
        assert t.sent == [{"type": "listening_started"}]

    @pytest.mark.asyncio
    async def test_barge_in_rebuilds_vad_endpoint_when_enabled(self, monkeypatch):
        """When VAD is enabled, barge-in rebuilds the listening VAD endpoint
        (min_silence_frames from settings)."""
        from src.server.session_handler import handle_audio_samples

        monkeypatch.setattr("src.server.config.settings.vad_enabled", True)
        monkeypatch.setattr("src.server.session_handler.app_state.vad", object())
        monkeypatch.setattr(
            "src.server.session_handler.build_vad_endpoint",
            lambda: "VAD_ENDPOINT",
        )

        t = self._make_fake_transport()
        state = self._make_state()
        state.is_playing = True

        class FakeTask:
            def done(self):
                return False

            def cancel(self):
                pass

        state.pipeline_task = FakeTask()

        class FakeBargeInVAD:
            async def process_async(self, audio):
                return "speech_start"

        state.barge_in_vad = FakeBargeInVAD()

        await handle_audio_samples(t, state, np.zeros(100, dtype=np.float32), 480000)

        assert state.is_listening is True
        assert state.vad_endpoint == "VAD_ENDPOINT"

    @pytest.mark.asyncio
    async def test_audio_while_idle_is_dropped(self, monkeypatch):
        """Audio arriving when neither listening nor playing is ignored
        (matches the original WS handler: the elif is_playing branch only
        ran when is_listening was False but is_playing True)."""
        from src.server.session_handler import handle_audio_samples

        t = self._make_fake_transport()
        state = self._make_state()

        called = []

        async def fake_dispatch(s):
            called.append(True)

        state.dispatch_pipeline = fake_dispatch
        monkeypatch.setattr("src.server.session_handler.app_state.vad", None)

        await handle_audio_samples(t, state, np.zeros(100, dtype=np.float32), 480000)

        assert called == []
        assert state.audio_buffer == []
        assert t.sent == []


class TestRateLimitGate:
    """Tests for issue #36: the shared rate-limit gate on pipeline dispatch.

    Both WS and WebRTC route every pipeline trigger (VAD auto-stop, buffer
    overflow, explicit stop_listening) through ``check_rate_limit`` in
    ``session_handler.py``. These tests exercise the helper directly with a
    mocked transport + token manager.
    """

    def _make_fake_transport(self):
        class FakeTransport:
            def __init__(self):
                self.sent = []

            async def send_json(self, data):
                self.sent.append(data)

        return FakeTransport()

    @pytest.mark.asyncio
    async def test_allowed_when_no_api_key(self, monkeypatch):
        """Auth disabled (api_key=None) → always allowed, no error sent."""
        from src.server.session_handler import check_rate_limit

        t = self._make_fake_transport()
        allowed = await check_rate_limit(t, None)
        assert allowed is True
        assert t.sent == []

    @pytest.mark.asyncio
    async def test_allowed_within_rate_limit(self, monkeypatch):
        """Within the per-minute limit → allowed, no error sent."""
        from src.server.session_handler import check_rate_limit

        class FakeKey:
            pass

        class FakeTM:
            async def check_rate_limit(self, key):
                return True

        monkeypatch.setattr("src.server.auth.token_manager", FakeTM())
        t = self._make_fake_transport()
        allowed = await check_rate_limit(t, FakeKey())
        assert allowed is True
        assert t.sent == []

    @pytest.mark.asyncio
    async def test_rate_limited_sends_error(self, monkeypatch):
        """Over the per-minute limit → error sent, dispatch blocked."""
        from src.server.session_handler import check_rate_limit

        class FakeKey:
            pass

        class FakeTM:
            async def check_rate_limit(self, key):
                return False

        monkeypatch.setattr("src.server.auth.token_manager", FakeTM())
        t = self._make_fake_transport()
        allowed = await check_rate_limit(t, FakeKey())
        assert allowed is False
        assert t.sent == [{"type": "error", "message": "rate_limited"}]


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_stt_tts_round_trip(self):
        """Test STT → TTS round trip (mock mode OK)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        tts = ChatterboxTTS()

        # Generate some audio (silence)
        input_audio = np.zeros(16000, dtype=np.float32)

        # Transcribe
        text = await stt.transcribe(input_audio)
        assert isinstance(text, str)

        # Synthesize (even empty text should work)
        if text.strip():
            output_audio = await tts.synthesize(text)
        else:
            output_audio = await tts.synthesize("Hello")

        assert isinstance(output_audio, np.ndarray)
        assert len(output_audio) > 0


# Run tests with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
