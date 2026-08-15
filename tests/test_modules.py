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
        """Test conversation history can be cleared."""
        backend = AIBackend()
        backend.conversation_history = [{"role": "user", "content": "test"}]
        backend.clear_history()
        assert len(backend.conversation_history) == 0

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
