# OpenClaw Voice — Agent Instructions

## Run the server

```bash
PYTHONPATH=. python -m src.server.main
```

Env loaded from `.env` via pydantic-settings with `OPENCLAW_` prefix. At minimum, set one of `OPENAI_API_KEY` or `OPENCLAW_GATEWAY_URL`+`OPENCLAW_GATEWAY_TOKEN`, plus a TTS key (`ELEVENLABS_API_KEY` recommended).

## Key commands

| Command | What |
|---|---|
| `uv pip install -e ".[dev,stt]"` | Dev install (preferred, CI uses uv) |
| `pip install -r requirements.txt` | Pip fallback |
| `pytest tests/test_modules.py -v` | Unit tests (fast, no server) |
| `pytest tests/test_server.py -v` | Integration (spawns real uvicorn on port 8799) |
| `ruff check src/ tests/` | Lint (line-length 100) |
| `black --check src/ tests/` | Format check (line-length 100) |
| `docker compose up` | GPU container |
| `docker compose --profile cpu up` | CPU-only container |

## Architecture

- **Entrypoint**: `src/server/main.py` — FastAPI app with one WebSocket route (`/ws`).
- **Client**: `src/client/index.html` — single-page browser UI with gapless audio scheduling.
- **STT** (`stt.py`): faster-whisper → openai-whisper → mock (auto GPU/CUDA/MPS/CPU detection).
- **TTS** (`tts.py`): ElevenLabs → Supertonic (local ONNX) → Edge TTS → Chatterbox → XTTS → mock. Has fallback (`_EdgeFallback` when Supertonic fails).
- **VAD** (`vad.py`): Silero VAD via torch.hub.
- **Auth** (`auth.py`): In-memory, keys prefixed `ocv_`. Optional `OPENCLAW_REQUIRE_AUTH=true` + `OPENCLAW_MASTER_KEY`.
- **System prompt**: "Métis, a wisdom companion" — hardcoded in `main.py:69` and `backend.py:27`.

## Testing quirks

- All test files insert `sys.path.insert(0, ..)` to resolve src imports.
- Server integration test (`test_server.py`) starts uvicorn as subprocess — requires port 8799 free.
- Some tests skip if `OPENAI_API_KEY` env var absent (`test_modules.py:93-96`).
- asyncio_mode = "auto" in `pyproject.toml`.

## Wire protocol

Browser → Server (base64 PCM float32 at 16kHz). Server → Browser (PCM int16 at 24kHz). Text cleaned for TTS via `text_utils.py` (strips markdown, URLs, hashtags, emojis).

## Misc

- STT model cache: `~/.cache/huggingface` (also used in Docker volumes).
- Pre-download models with `python scripts/download_models.py base`.
- Generate auth keys with `python scripts/generate_master_key.py`.
- `SKILL.md` is a separate OpenClaw skill doc, not agent instructions.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


