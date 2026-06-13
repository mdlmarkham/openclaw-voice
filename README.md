# OpenClaw Voice

**Self-hosted, browser-based voice interface for AI agents.**

Talk to your AI like you talk to Alexa — but private, local, and connected to your own agent with full memory continuity.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)

🌐 **Website:** [openclawvoice.com](https://openclawvoice.com)

## Features

| Feature | Description |
|---------|-------------|
| 🎤 **Local STT** | faster-whisper runs locally. Your voice never leaves your machine. |
| 🔊 **Streaming TTS** | Supertonic (local CPU, 31 languages) or ElevenLabs (cloud). Sentence-by-sentence streaming. |
| 🎯 **Voice Activity Detection** | Silero VAD filters background noise. Works in noisy environments. |
| 🧹 **Smart Text Cleaning** | Strips markdown, hashtags, URLs before TTS. No more "hash hash". |
| 🔌 **OpenClaw Gateway** | Full agent context — same memory, persona, and tools as text chat. Seamless cross-channel continuity. |
| 🎭 **Per-Agent Voices** | Each agent gets a distinct Supertonic voice. Métis → F2, Atlas → M2, Hephaestus → M4. |
| 🌐 **Browser-Based** | No app install. Works on desktop and mobile. HTTPS via Tailscale. |
| 🚗 **Continuous Mode** | Hands-free conversation. Auto-listens after each response. |
| 🔒 **API Key Auth** | Optional per-client authentication with tiered rate limits. |

## Quick Start

```bash
# Clone
git clone https://github.com/Purple-Horizons/openclaw-voice.git
cd openclaw-voice

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
PYTHONPATH=. python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8765

# Open http://localhost:8765
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENCLAW_GATEWAY_URL` | No | — | OpenClaw gateway URL (enables full agent mode) |
| `OPENCLAW_GATEWAY_TOKEN` | No | — | Gateway auth token |
| `OPENCLAW_BACKEND_TYPE` | No | `openai` | Backend: `openclaw` or `openai` |
| `OPENCLAW_PORT` | No | `8765` | Server port |
| `OPENCLAW_STT_MODEL` | No | `small` | Whisper model size (see table below) |
| `OPENCLAW_STT_DEVICE` | No | `auto` | Device: `auto`, `cpu`, `cuda` |
| `OPENCLAW_TTS_MODEL` | No | `supertonic` | TTS backend (see table below) |
| `SUPERTONIC_VOICE` | No | `F2` | Supertonic voice (F1-F5 female, M1-M5 male) |
| `SUPERTONIC_MODEL` | No | `supertonic-2` | Supertonic model version |
| `OPENCLAW_REQUIRE_AUTH` | No | `false` | Require API keys for WebSocket clients |
| `OPENCLAW_MASTER_KEY` | No | — | Master key for API key management |

> **OpenClaw gateway mode (recommended):** Set `OPENCLAW_GATEWAY_URL` and `OPENCLAW_GATEWAY_TOKEN`. The server sends only the user message to the gateway — no conversation history duplication. The gateway maintains full agent persona, memory, and workspace context, giving you seamless continuity between voice and text channels.
>
> **Direct OpenAI mode:** Set `OPENAI_API_KEY`. The server manages its own conversation history (last 10 turns). Note: conversation state is global — shared across all connected clients. For multi-user deployments, use the OpenClaw gateway.

### Whisper Model Sizes

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| `tiny` | Fastest | Fair | ~400MB | Quick testing |
| `base` | Fast | Good | ~1GB | Light deployments |
| `small` | Medium | Better | ~2GB | **Default. Good balance.** |
| `medium` | Slower | Great | ~5GB | Accuracy priority |
| `large-v3-turbo` | Slow | Best | ~6GB | Maximum accuracy |

### TTS Backends

| Backend | Type | Quality | Latency | Languages | Notes |
|---------|------|---------|---------|-----------|-------|
| **Supertonic** | Local CPU | Very Good | ~200ms | 31 | **Default.** ONNX, no GPU needed. |
| **ElevenLabs** | Cloud | Excellent | ~500ms | Multi | Streaming supported. Requires API key. |
| Edge TTS | Cloud | Good | ~300ms | Multi | Microsoft voices. Free. |
| Chatterbox | Local | Very Good | ~1s | English | MIT license, voice cloning |
| Mock | Local | None | 0ms | — | For testing (silence) |

Backend priority: ElevenLabs (if key provided) → Supertonic → Edge TTS → Chatterbox → Mock.

### Per-Agent Voice Mapping

When using the OpenClaw gateway, each agent automatically gets a distinct voice:

| Agent | Voice | Character |
|-------|-------|-----------|
| Métis | F2 | Warm, clear female |
| Atlas | M2 | Steady, authoritative male |
| Hephaestus | M4 | Deeper, precise male |
| Clio | F4 | Thoughtful, measured female |
| Deepthought | M1 | Professional, clear male |
| Mara | F5 | Gentle, warm female |

Override with `SUPERTONIC_VOICE` env var, or extend `ChatterboxTTS.AGENT_VOICE_MAP`.

## Architecture

```
┌─────────────┐   WebSocket   ┌──────────────────────────────────────────┐
│   Browser   │◄────────────►│            Voice Server                  │
│  (mic/spk)  │               │                                          │
└─────────────┘               │  ┌───────────┐   ┌─────────────────┐    │
                              │  │ Whisper   │   │  OpenClaw       │    │
                              │  │ (STT)     │   │  Gateway        │    │
                              │  └─────┬─────┘   │  or OpenAI      │    │
                              │        │         └────────┬────────┘    │
                              │        ▼                  ▼              │
                              │  ┌─────────────────────────────────┐    │
                              │  │     Supertonic / ElevenLabs    │    │
                              │  │          (TTS)                  │    │
                              │  └─────────────────────────────────┘    │
                              │        ▲                                │
                              │    [VAD]  [Audio buffer cap: 30s]     │
                              └──────────────────────────────────────────┘
```

**Streaming Flow:**
1. User speaks → faster-whisper transcribes locally
2. Text sent to OpenClaw gateway (voice-modality hint + user message only)
3. Gateway responds with full agent context (persona, memory, tools)
4. AI response streamed back → buffer completed sentences
5. First sentence → TTS starts immediately (Supertonic ~200ms)
6. Audio streams to browser while AI continues generating
7. Result: ~50% faster perceived response than waiting for full completion

**Cross-Channel Continuity:**

When connected to the OpenClaw gateway, the voice server sends **only** the user's message and a voice-modality system hint. The gateway manages the full conversation — persona, workspace context, long-term memory. This means a conversation started in Telegram continues seamlessly in voice, and vice versa. No history duplication.

## HTTPS for Mobile

Mobile browsers require HTTPS for microphone access.

**Tailscale (recommended for self-hosting):**
```bash
# The systemd unit uses Tailscale certificates
# Place certs at /var/lib/tailscale/certs/your-host.ts.net.{crt,key}
```

**nginx + Let's Encrypt:**
```nginx
server {
    listen 443 ssl;
    server_name voice.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Systemd Service

```ini
[Unit]
Description=OpenClaw Voice
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/openclaw-voice
Environment=PYTHONPATH=.
Environment=OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789
Environment=OPENCLAW_GATEWAY_TOKEN=your-token
Environment=OPENCLAW_BACKEND_TYPE=openclaw
Environment=OPENCLAW_PORT=8443
Environment=OPENCLAW_STT_MODEL=small
Environment=OPENCLAW_STT_DEVICE=cpu
Environment=OPENCLAW_TTS_MODEL=supertonic
Environment=SUPERTONIC_VOICE=F2
Environment=SUPERTONIC_MODEL=supertonic-2
Environment=OPENCLAW_REQUIRE_AUTH=false
ExecStart=/path/to/openclaw-voice/.venv/bin/python3 -m uvicorn src.server.main:app \
    --host 0.0.0.0 --port 8443 \
    --ssl-keyfile /var/lib/tailscale/certs/host.ts.net.key \
    --ssl-certfile /var/lib/tailscale/certs/host.ts.net.crt

[Install]
WantedBy=multi-user.target
```

## API

### WebSocket Protocol

Connect to `ws://localhost:8765/ws` (or `wss://` for HTTPS):

```javascript
// Start recording (optionally select agent)
{ "type": "start_listening", "agent": "metis" }  // agent optional

// Send audio (base64 PCM float32, 16kHz)
{ "type": "audio", "data": "base64..." }

// Stop recording
{ "type": "stop_listening" }

// Receive events:
{ "type": "transcript", "text": "...", "final": true }
{ "type": "response_chunk", "text": "..." }        // Streaming text
{ "type": "sentence_start", "seq": 0 }             // Sentence boundary
{ "type": "audio_chunk", "data": "...", "sample_rate": 44100, "seq": 0 }
{ "type": "sentence_end", "seq": 0 }               // Sentence boundary
{ "type": "response_complete", "text": "..." }      // Full response
{ "type": "vad_status", "speech_detected": true }   // VAD feedback
{ "type": "listening_started" }
{ "type": "listening_stopped" }
{ "type": "history_cleared" }                        // After clear_history
```

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, model info, agent voice map |
| `/` | GET | Voice UI (index.html) |
| `/api/keys` | POST | Create API key (auth required) |
| `/api/keys/{key}` | DELETE | Revoke API key (auth required) |

### Health Check Response

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "stt": { "backend": "faster-whisper", "model": "small", "device": "cpu" },
  "tts": {
    "backend": "supertonic",
    "model": "supertonic-2",
    "voice": "F2",
    "sample_rate": 44100,
    "agent_voice_map": { "metis": "F2", "atlas": "M2", ... }
  },
  "backend": "openclaw",
  "vad": "loaded",
  "config": { "stt_model": "small", "tts_model": "supertonic", ... }
}
```

## Security

- **XSS protection:** All user and assistant text is HTML-escaped before rendering. Markdown transforms run on escaped content.
- **CORS:** `allow_origins=["*"]` with `allow_credentials=False` — no credential leakage.
- **Audio buffer cap:** 30-second maximum prevents unbounded memory growth from misbehaving clients.
- **API key auth:** Optional per-client authentication with tiered rate limits (see `auth.py`).
- **No runtime package installation:** Dependencies are declared in `pyproject.toml`, not installed at runtime.

> **Note:** When `OPENCLAW_REQUIRE_AUTH=false` (the default), the `/api/keys` endpoint is accessible without authentication. This is intentional for local development. Enable auth for production deployments.

## Roadmap

- [x] WebSocket voice gateway
- [x] Whisper STT (local, faster-whisper)
- [x] Supertonic TTS (local CPU, 31 languages)
- [x] ElevenLabs TTS (cloud, streaming)
- [x] Streaming TTS (sentence-by-sentence)
- [x] Voice Activity Detection (Silero)
- [x] Text cleaning (markdown/hashtags/URLs)
- [x] Continuous conversation mode
- [x] OpenClaw gateway integration (cross-channel continuity)
- [x] Per-agent voice mapping
- [x] API key authentication
- [x] XSS protection (HTML escaping)
- [x] Audio buffer cap (DoS prevention)
- [ ] Server-side VAD endpointing
- [ ] Barge-in (interrupt AI when user speaks)
- [ ] WebRTC for lower latency
- [ ] Per-connection conversation state (direct OpenAI mode)
- [ ] Docker containerization
- [ ] Mobile-optimized UI
- [ ] Latency metrics dashboard

## Project Structure

```
openclaw-voice/
├── src/
│   └── server/
│       ├── main.py          # FastAPI app, WebSocket handler, health endpoint
│       ├── backend.py       # AI backend (OpenClaw gateway or direct OpenAI)
│       ├── tts.py           # TTS engine (Supertonic > ElevenLabs > Edge > Chatterbox)
│       ├── stt.py           # STT engine (faster-whisper)
│       ├── vad.py           # Voice Activity Detection (Silero)
│       ├── auth.py          # API key management with tiered rate limits
│       └── text_utils.py    # Markdown/URL stripping for TTS
├── src/
│   └── client/
│       └── index.html       # Single-file browser UI
├── .env.example             # Configuration template
├── pyproject.toml           # Dependencies and metadata
├── Dockerfile               # GPU-enabled Docker image
└── .beads/                  # Issue tracking (Beads)
```

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Local STT
- [Supertonic](https://github.com/nicolabottura/supertonic) — Local TTS, 31 languages
- [ElevenLabs](https://elevenlabs.io) — Cloud TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection
- Built for [OpenClaw](https://openclaw.ai)

---

**Made with 🦞 by [Purple Horizons](https://purplehorizons.io)**
