#!/usr/bin/env python3
"""
Browser-based verification of the AudioWorklet mic-capture migration (issue #16).

Drives a real browser (Chromium / Firefox via Playwright) against a real
uvicorn server running the demo client, and asserts the full voice round-trip
works over the AudioWorklet path with NO ScriptProcessorNode deprecation
warnings.

Usage:
    uv pip install playwright
    playwright install chromium firefox
    python scripts/verify_audio_worklet.py [--browser chromium] [--browser firefox]

Requires Playwright + a browser install. The server runs in mock STT/TTS mode
so no API keys or heavy model downloads are needed.
"""

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OPENCLAW_STT_MODEL"] = "tiny"
os.environ["OPENCLAW_STT_DEVICE"] = "cpu"
os.environ["OPENCLAW_TTS_MODEL"] = "mock"


def _start_server(port: int) -> threading.Thread:
    """Start the FastAPI app with uvicorn in a background thread."""
    import uvicorn

    from src.server.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started and server.servers:
            return server, thread
        time.sleep(0.1)
    raise RuntimeError("uvicorn server failed to start")


def _run_browser_check(browser_name: str, port: int, fallback: bool = False) -> dict:
    from playwright.sync_api import sync_playwright

    result = {"browser": browser_name, "passed": False, "notes": []}

    with sync_playwright() as p:
        if browser_name == "chromium":
            browser = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required",
                ]
            )
        elif browser_name == "firefox":
            browser = p.firefox.launch(
                firefox_user_prefs={
                    "media.navigator.streams.fake": True,
                    "media.navigator.permission.disabled": True,
                }
            )
        else:
            raise ValueError(f"Unsupported browser: {browser_name}")

        context = browser.new_context()
        page = context.new_page()
        console_msgs = []
        page.add_init_script(
            """
            window.__wsRecv = [];
            const OrigWS = window.WebSocket;
            window.WebSocket = class extends OrigWS {
                constructor(...args) {
                    super(...args);
                    this.addEventListener('message', (e) => {
                        window.__wsRecv.push('recv:' + e.data);
                    });
                    this.addEventListener('open', () => {
                        window.__wsRecv.push('open');
                    });
                }
            };

            // Instrument AudioWorkletNode to prove the worklet path (not the
            // ScriptProcessor fallback) is actually exercised.
            window.__awWorkletUsed = false;
            const OrigAudioWorkletNode = window.AudioWorkletNode;
            if (OrigAudioWorkletNode) {
                window.AudioWorkletNode = class extends OrigAudioWorkletNode {
                    constructor(ctx, name, opts) {
                        if (name === 'audio-capture-processor') {
                            window.__awWorkletUsed = true;
                        }
                        super(ctx, name, opts);
                    }
                };
            }
            """
        )
        if fallback:
            # Force the ScriptProcessorNode fallback path: make AudioWorklet
            # appear unsupported so the client's feature detection takes the
            # createScriptProcessor branch.
            page.add_init_script(
                """
                window.__awWorkletUsed = false;
                Object.defineProperty(window, 'AudioWorkletNode', { value: undefined });
                if (window.AudioContext) {
                    Object.defineProperty(window.AudioContext.prototype, 'audioWorklet', {
                        get: function () { return undefined; },
                    });
                }
                if (window.webkitAudioContext) {
                    Object.defineProperty(window.webkitAudioContext.prototype, 'audioWorklet', {
                        get: function () { return undefined; },
                    });
                }
                """
            )
        page.on("console", lambda m: console_msgs.append((m.type, m.text)))
        page.on("pageerror", lambda e: console_msgs.append(("pageerror", str(e))))

        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_selector("#voiceBtn")
        page.wait_for_timeout(2000)

        # Hold-to-talk round-trip.
        btn = page.locator("#voiceBtn")
        bb = btn.bounding_box()
        page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.mouse.down()
        page.wait_for_timeout(2500)
        page.mouse.up()
        page.wait_for_timeout(4000)

        ws_recv = page.evaluate("window.__wsRecv")

        # Collect the distinct message types we observed.
        types = set()
        for m in ws_recv:
            if m.startswith("recv:{"):
                try:
                    import json

                    types.add(json.loads(m[len("recv:"):]).get("type"))
                except ValueError:
                    pass
        result["types"] = sorted(types)

        # Assert the round-trip happened.
        required = {"listening_started", "listening_stopped", "transcript", "audio_chunk"}
        missing = required - types
        result["notes"].append(f"message types observed: {sorted(types)}")
        result["notes"].append(f"missing: {missing or 'none'}")

        # Assert no ScriptProcessorNode deprecation warning.
        deprecations = [
            txt for typ, txt in console_msgs if "ScriptProcessorNode" in txt
        ]
        result["notes"].append(f"ScriptProcessorNode deprecations: {deprecations or 'none'}")

        # AudioWorkletNode must be supported & used (unless forcing fallback).
        result["audio_worklet_supported"] = page.evaluate(
            "typeof AudioWorkletNode !== 'undefined'"
        )
        result["audio_worklet_used"] = page.evaluate("window.__awWorkletUsed")

        if fallback:
            # In fallback mode the round-trip must still work, and the worklet
            # must NOT have been used.
            result["passed"] = (
                not missing
                and not result["audio_worklet_used"]
            )
        else:
            result["passed"] = (
                not missing
                and not deprecations
                and bool(result["audio_worklet_supported"])
                and bool(result["audio_worklet_used"])
            )
        browser.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AudioWorklet capture in browser.")
    parser.add_argument(
        "--browser",
        action="append",
        default=None,
        choices=["chromium", "firefox"],
        help="Browser to test (repeatable). Default: chromium",
    )
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force the ScriptProcessorNode fallback path (AudioWorklet disabled) "
        "and assert the round-trip still works.",
    )
    args = parser.parse_args()
    browsers = args.browser or ["chromium"]

    server, thread = _start_server(args.port)
    try:
        failures = 0
        for browser_name in browsers:
            mode = "fallback" if args.fallback else "worklet"
            print(f"\n=== Testing {browser_name} ({mode} path) ===")
            result = _run_browser_check(browser_name, args.port, fallback=args.fallback)
            for note in result["notes"]:
                print(f"  {note}")
            print(f"  AudioWorkletNode supported: {result['audio_worklet_supported']}")
            print(f"  AudioWorklet capture node used: {result['audio_worklet_used']}")
            print(f"  PASSED: {result['passed']}")
            if not result["passed"]:
                failures += 1
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
