"""
Server integration tests for OpenClaw Voice.
Uses FastAPI TestClient instead of launching a real subprocess.
"""

import base64
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure test env before importing app
os.environ.setdefault("OPENCLAW_STT_MODEL", "tiny")
os.environ.setdefault("OPENCLAW_STT_DEVICE", "cpu")
os.environ.setdefault("OPENCLAW_TTS_MODEL", "mock")

from src.server.main import app


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient connected to the FastAPI app (in-process)."""
    with TestClient(app) as c:
        yield c


class TestServerHTTP:
    """Test HTTP endpoints."""

    def test_index_page(self, client):
        """Test that index page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Voice" in response.text

    def test_index_page_contains_voice_button(self, client):
        """Test that the voice button is in the page."""
        response = client.get("/voice")
        assert response.status_code == 200
        assert "voice-button" in response.text

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "stt" in data
        assert "tts" in data
        assert "backend" in data

    def test_health_has_uptime(self, client):
        """Test health includes uptime."""
        response = client.get("/health")
        data = response.json()
        assert data["uptime_seconds"] >= 0


class TestServerWebSocket:
    """Test WebSocket functionality."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as ws:
            assert ws is not None

    def test_ping_pong(self, client):
        """Test ping/pong."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            response = ws.receive_json()
            assert response["type"] == "pong"

    def test_start_stop_listening(self, client):
        """Test start/stop listening cycle."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start_listening"})
            response = ws.receive_json()
            assert response["type"] == "listening_started"

            ws.send_json({"type": "stop_listening"})
            response = ws.receive_json()
            assert response["type"] == "listening_stopped"

    def test_audio_flow(self, client):
        """Test sending audio and getting response."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start_listening"})
            ws.receive_json()  # listening_started

            audio = np.zeros(16000, dtype=np.float32)
            audio_b64 = base64.b64encode(audio.tobytes()).decode()

            ws.send_json({"type": "audio", "data": audio_b64})

            ws.send_json({"type": "stop_listening"})

            messages = []
            for _ in range(10):
                try:
                    response = ws.receive_json()
                    messages.append(response["type"])
                    if response["type"] == "listening_stopped":
                        break
                except Exception:
                    break

            assert "transcript" in messages or "listening_stopped" in messages
