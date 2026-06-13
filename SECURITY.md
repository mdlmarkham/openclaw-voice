# Security Policy

## Reporting a Vulnerability

If you discover a security issue in OpenClaw Voice, please report it privately:

- **Email:** hello@purplehorizons.io
- **Do not** open a public GitHub issue.

You should receive a response within 48 hours. If not, follow up to ensure receipt.

## Scope

- API key handling and authentication
- Prompt injection via AI responses (reflected in the browser UI)
- Remote code execution through model loading
- Audio buffer memory safety

## Out of Scope

- Dependency CVEs (tracked via Dependabot)
- Attacks requiring physical access to the server
- Social engineering of project maintainers

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | ✅ |
| < main  | ❌ |
