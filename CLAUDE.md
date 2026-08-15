# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Issue Tracking

Issue tracking for this project lives in **GitHub Issues** on
`mdlmarkham/openclaw-voice`. Do **not** use the local `bd`/Beads tracker
(removed 2026-08-15). Use the `gh` CLI for all task tracking:

```bash
gh issue list --repo mdlmarkham/openclaw-voice --state open   # view open work
gh issue view <num> --repo mdlmarkham/openclaw-voice          # issue details
gh issue create --repo mdlmarkham/openclaw-voice ...          # new issue
gh issue close <num> --repo mdlmarkham/openclaw-voice         # close work
```

- File a GitHub issue for any bug or improvement you discover while working.
- Keep the issue backlog accurate: close work you finish, update in-progress items.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File GitHub issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_
