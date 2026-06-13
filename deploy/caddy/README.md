# Caddy HTTPS Proxy for OpenClaw Voice

One-command HTTPS for mobile microphone access.

## Quick Start

```bash
# Point your domain to this server, then:
export DOMAIN=voice.yourdomain.com
caddy run --config deploy/caddy/Caddyfile
```

Requirements: [Caddy](https://caddyserver.com/download) installed on the server.

## What It Does

- Reverse-proxies port 8765 with TLS
- Auto-provisions Let's Encrypt certificates (no certbot needed)
- Passes WebSocket upgrades through
- Logs to `/var/log/caddy/openclaw-voice.log`

## Custom Port

If the voice server runs on a different port:

```bash
OPENCLAW_PORT=8766 caddy run --config deploy/caddy/Caddyfile
```

## Production Use

For production, run as a systemd service:

```bash
sudo caddy run --config /opt/openclaw-voice/deploy/caddy/Caddyfile
```

Or package it with the Docker Compose stack by adding a caddy service.
