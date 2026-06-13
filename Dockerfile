# OpenClaw Voice — production Docker image
#
# Supports NVIDIA GPUs for fast Whisper + TTS inference.
# Multi-stage build keeps the final image lean.
#
# Build:
#   docker build -t openclaw-voice .
#
# Run (CPU):
#   docker run -p 8765:8765 openclaw-voice
#
# Run (GPU):
#   docker run --gpus all -p 8765:8765 openclaw-voice
#
# Mount volumes for persistent models and voice samples:
#   docker run -v models:/app/models -v voices:/app/voices ... openclaw-voice

# ── Build stage ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[stt]" && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Runtime stage ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r openclaw && useradd -r -g openclaw -d /app -s /sbin/nologin openclaw

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/src/ ./src/

# Directories for mounted volumes
RUN mkdir -p /app/models /app/voices && chown -R openclaw:openclaw /app

USER openclaw

ARG STT_MODEL=large-v3-turbo

# Pre-download STT model during build
RUN python -c "
from faster_whisper import WhisperModel
WhisperModel('${STT_MODEL}', device='cpu', compute_type='int8')
print(f'✅ STT model {STT_MODEL} cached')
"

# Environment
ENV OPENCLAW_HOST=0.0.0.0
ENV OPENCLAW_PORT=8765
ENV OPENCLAW_STT_MODEL=large-v3-turbo
ENV OPENCLAW_STT_DEVICE=cuda
ENV OPENCLAW_REQUIRE_AUTH=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8765/health || exit 1

CMD ["python", "-m", "uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8765"]
