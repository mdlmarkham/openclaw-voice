# Feature Spec: Configurable Per-Agent Voice Hints

**Status:** Proposed
**Type:** Enhancement / Architecture
**Related:** #23 (per-agent voice model resolution), `AGENT_VOICE_CONFIG` in `src/server/backend.py`

## Problem

Per-agent voice hints (the `hint` strings in `AGENT_VOICE_CONFIG`) are hardcoded
in `src/server/backend.py`. Tuning them — e.g. tightening the "assess interest"
gate, adjusting word budgets per agent, or experimenting with new phrasing —
requires a code change, a commit, and a redeploy. This makes iteration slow and
forces every deployment to carry the full hint set even when only one agent
needs a tweak.

The hints are also the single most impactful lever on voice UX (they shape how
every agent speaks), so they deserve first-class, runtime-tunable configuration
rather than being buried in source.

## Goals

1. **Runtime-tunable** — hints editable without a code change or redeploy.
2. **Per-agent** — each agent keeps its own hint, personality, and word budget.
3. **Backward compatible** — existing hardcoded hints remain the default when no
   config is provided; no behavior change out of the box.
4. **Validated** — malformed config fails fast with a clear error, not silently.
5. **Documented** — the config surface is discoverable and self-explanatory.

## Non-Goals

- Not changing how hints are *injected* (the gateway-mode `system` message
  approach stays — it correctly avoids overriding agent personas).
- Not adding a UI. A config file / env surface is sufficient for this stage.
- Not making hints per-*session* or per-*user* (out of scope; see Future Work).

## Proposed Design

### 1. Config source (layered, env-first)

Resolve the hint for an agent in this order:

1. **Environment variable** — `VOICE_HINT_<AGENT>` (e.g. `VOICE_HINT_METIS`),
   for quick overrides without touching files. Uppercase agent id, `-` → `_`.
2. **Config file** — a JSON/YAML file (path via `VOICE_HINT_CONFIG` env, default
   `./voice_hints.json` if present) with a top-level map of agent id → hint.
3. **Built-in default** — the existing `AGENT_VOICE_CONFIG` hints.

This gives three tiers: ephemeral override, persistent file, and code default.

### 2. Config file schema

```json
{
  "metis": {
    "hint": "You are speaking through a voice interface...",
    "word_budget": 50
  },
  "atlas": {
    "hint": "...",
    "word_budget": 40
  }
}
```

- `hint`: the full system-message text (replaces the built-in for that agent).
- `word_budget`: optional; if present, used to inject a "keep it under N words"
  clause into the hint at build time (so budgets stay tunable without editing
  the prose). If absent, the hint is used verbatim.

### 3. Resolution logic

Add a `resolve_voice_hint(agent_id) -> str` function (or method on `AIBackend`)
that:

1. Checks `os.environ.get(f"VOICE_HINT_{agent_id.upper().replace('-','_')}")`.
2. Falls back to the config file map.
3. Falls back to `AGENT_VOICE_CONFIG[agent_id]["hint"]`.
4. Falls back to `DEFAULT_VOICE_HINT`.

`_build_messages()` calls this instead of reading `AGENT_VOICE_CONFIG` directly.

### 4. Validation & failure mode

- On startup, if `VOICE_HINT_CONFIG` points to a file, parse and validate it:
  - Must be a JSON object mapping string agent ids to objects with a string
    `hint` and optional positive-int `word_budget`.
  - Unknown keys → warn, don't fail (forward-compatible).
  - Malformed JSON / wrong types → **fail fast** with a clear error naming the
    file and the offending key, so a bad config can't silently ship a broken
    voice experience.
- Env overrides are validated at call time (cheap string check).

### 5. Docs

- Add a `docs/voice-hints.md` section to the README documenting the three
  tiers, the schema, and an example.
- Note the interaction with the "assess interest" gate (see #<PR for Option A>)
  so future editors know the intended voice behavior.

## Testing

- Unit tests for `resolve_voice_hint`:
  - env override wins over file and default
  - file wins over default
  - default used when nothing configured
  - unknown agent → `DEFAULT_VOICE_HINT`
  - `word_budget` injects the budget clause correctly
  - malformed config file raises a clear error
- Existing `_build_messages` tests updated to assert the resolved hint is used.

## Future Work (explicitly out of scope)

- Per-session / per-user hint overrides (e.g. "always give me the short version").
- A small admin UI or API endpoint to hot-reload hints without restart.
- Hint templating with variables (agent name, current word budget).

## Acceptance Criteria

- [ ] Hints resolvable from env, config file, or built-in default, in that order.
- [ ] No behavior change when no config is provided (defaults identical).
- [ ] Malformed config fails fast with a clear error.
- [ ] `word_budget` supported as a tunable without editing prose.
- [ ] README documents the config surface.
- [ ] Unit tests cover resolution order, validation, and budget injection.
