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
