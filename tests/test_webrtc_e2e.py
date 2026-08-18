"""
End-to-end WebRTC integration tests using a REAL aiortc client over real RTP.

This is the manual-verification gate for issue #22: VAD auto-endpointing and
barge-in must work over actual RTP audio timing, not just the mocked
transports used in the unit tests. These tests spin up a real uvicorn server
in a background thread and drive it with a real aiortc RTCPeerConnection.

Skipped when aiortc is not installed (the webrtc extra is optional).
"""

import asyncio
import fractions
import os
import sys
import threading
import time

import numpy as np
import pytest
from av import AudioFrame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

aiortc = pytest.importorskip("aiortc")
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from src.server import state as app_state

# ── Test doubles ──────────────────────────────────────────────────────


class FakeVAD:
    """VAD that reports speech based on audio amplitude — tone frames are
    speech, silence frames are not. Drives both the endpointing state machine
    (speech_end on silence) and barge-in (speech_start on tone during
    playback)."""

    def __init__(self):
        self.calls = 0

    async def is_speech_async(self, audio, sample_rate=16000):
        self.calls += 1
        return bool(np.abs(audio).max() > 0.01)


class FakePipeline:
    """Pipeline that records dispatches and yields slowly (simulating a
    long-running response so barge-in can interrupt it)."""

    def __init__(self):
        self.dispatches = 0
        self.buffers = []

    async def process_audio(self, buf, session):
        self.dispatches += 1
        self.buffers.append([b.copy() for b in buf])
        # Yield a few events slowly so is_playing stays True long enough for
        # a barge-in to be detected.
        for _ in range(5):
            yield None
            await asyncio.sleep(0.2)


class ClientAudioTrack(MediaStreamTrack):
    """Sends tone frames then silence frames over RTP, paced at ~20ms.

    Cycles tone → silence → tone → silence so barge-in can be detected
    during playback (the second tone burst arrives while is_playing)."""

    kind = "audio"

    def __init__(self, frames, silence_frames=40, cycles=2):
        super().__init__()
        self._frames = list(frames)
        self._silence_frames = silence_frames
        self._cycles = cycles
        self._idx = 0
        self._sr = frames[0].sample_rate
        self._dur = 0.02

    def _make_silence(self, i):
        f = AudioFrame.from_ndarray(
            np.zeros((1, int(self._sr * self._dur)), dtype=np.int16),
            format="s16",
            layout="mono",
        )
        f.sample_rate = self._sr
        f.pts = int(i * self._sr * self._dur)
        f.time_base = fractions.Fraction(1, self._sr)
        return f

    async def recv(self):
        cycle_len = len(self._frames) + self._silence_frames
        total = cycle_len * self._cycles
        if self._idx >= total:
            await asyncio.sleep(0.1)
            return self._make_silence(total - 1)
        pos = self._idx % cycle_len
        self._idx += 1
        await asyncio.sleep(0.02)
        if pos < len(self._frames):
            return self._frames[pos]
        return self._make_silence(self._idx - 1)


def make_tone_frames(n=20, sr=48000, dur=0.02):
    """Build n 20ms 440Hz tone frames at 48kHz (default Opus rate)."""
    frames = []
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = (tone * 32767).astype(np.int16)
    for i in range(n):
        f = AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        f.sample_rate = sr
        f.pts = int(i * sr * dur)
        f.time_base = fractions.Fraction(1, sr)
        frames.append(f)
    return frames


# ── Server fixture ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def webrtc_server():
    """Start a real uvicorn server in a background thread with fakes injected."""
    import uvicorn

    from src.server.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to bind a port.
    for _ in range(100):
        if server.started and server.servers:
            port = server.servers[0].sockets[0].getsockname()[1]
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("uvicorn server failed to start")

    # Inject fakes AFTER the lifespan has run (it replaces app_state.vad /
    # app_state.pipeline with real instances at startup).
    fake_vad = FakeVAD()
    fake_pipeline = FakePipeline()
    app_state.vad = fake_vad
    app_state.pipeline = fake_pipeline

    yield {"port": port, "vad": fake_vad, "pipeline": fake_pipeline}

    server.should_exit = True
    thread.join(timeout=10)


async def _connect_client(port, frames, on_track=None):
    """Create a real aiortc client, exchange SDP, and return the PC."""
    pc = RTCPeerConnection()
    pc.addTrack(ClientAudioTrack(frames))
    if on_track:
        pc.on("track")(on_track)

    # The client creates the data channel; the server's on_datachannel fires.
    dc = pc.createDataChannel("chat")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"http://127.0.0.1:{port}/api/webrtc/offer",
            json={"sdp": pc.localDescription.sdp},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="answer"))
    return pc, dc


async def _wait_for_datachannel(dc, timeout=5.0):
    """Wait for the data channel to open."""
    for _ in range(int(timeout / 0.1)):
        if dc.readyState == "open":
            return dc
        await asyncio.sleep(0.1)
    return None


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webrtc_offer_creates_answer(webrtc_server):
    """A real aiortc client can complete the SDP exchange."""
    pc, _ = await _connect_client(webrtc_server["port"], make_tone_frames())
    await asyncio.sleep(1)
    assert pc.connectionState in ("connected", "connecting")
    await pc.close()


@pytest.mark.asyncio
async def test_webrtc_vad_auto_endpointing_over_rtp(webrtc_server):
    """VAD auto-endpointing: speech_end over real RTP triggers pipeline
    dispatch WITHOUT an explicit stop_listening (issue #22 acceptance)."""
    pipeline = webrtc_server["pipeline"]
    pipeline.dispatches = 0

    pc, dc = await _connect_client(webrtc_server["port"], make_tone_frames())
    dc = await _wait_for_datachannel(dc)
    assert dc is not None

    dc.send('{"type": "start_listening"}')
    # Let RTP frames flow through the collector + VAD endpointing.
    await asyncio.sleep(3)

    assert pipeline.dispatches >= 1, "pipeline should dispatch on VAD speech_end"
    await pc.close()


@pytest.mark.asyncio
async def test_webrtc_barge_in_over_rtp(webrtc_server):
    """Barge-in: speech during playback cancels the in-flight pipeline task
    and resumes listening (issue #22 acceptance).

    Flow over real RTP:
    1. start_listening → is_listening=True
    2. tone burst → VAD speech_end → pipeline dispatch (is_playing=True)
    3. stop_listening → is_listening=False (the first pipeline task keeps
       running, so is_playing stays True)
    4. second tone burst → barge-in branch → speech_start → cancels the
       in-flight task → listening_started"""
    pipeline = webrtc_server["pipeline"]
    pipeline.dispatches = 0

    pc, dc = await _connect_client(webrtc_server["port"], make_tone_frames())
    dc = await _wait_for_datachannel(dc)
    assert dc is not None

    received = []

    @dc.on("message")
    def on_msg(msg):
        received.append(msg)

    dc.send('{"type": "start_listening"}')
    await asyncio.sleep(2)  # tone burst → speech_end → dispatch

    assert pipeline.dispatches >= 1, "first dispatch should come from VAD speech_end"

    dc.send('{"type": "stop_listening"}')
    await asyncio.sleep(1)  # is_listening=False; first pipeline still playing

    # Second tone burst arrives while playing → barge-in.
    await asyncio.sleep(2)

    listening_started = [m for m in received if '"listening_started"' in m]
    assert len(listening_started) >= 2, (
        f"expected listening_started on start + barge-in, got {len(listening_started)}"
    )
    await pc.close()
