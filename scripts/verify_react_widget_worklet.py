#!/usr/bin/env python3
"""
Browser verification of the React widget's AudioWorklet capture path (issue #16).

Loads the built React widget (packages/react/dist) in a real Chromium browser
against a real uvicorn server, performs a hold-to-talk round-trip with a fake
microphone, and asserts:
  - the widget mounts and opens the WebSocket,
  - audio is captured and sent (server replies listening_started → transcript
    → response_chunk → audio_chunk → response_complete),
  - NO ScriptProcessorNode deprecation warnings,
  - AudioWorkletNode is used.

Usage:
    cd packages/react && npm install && npm run build
    cd ../.. && python scripts/verify_react_widget_worklet.py
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OPENCLAW_STT_MODEL"] = "tiny"
os.environ["OPENCLAW_STT_DEVICE"] = "cpu"
os.environ["OPENCLAW_TTS_MODEL"] = "mock"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _start_server(port: int):
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


def main() -> int:
    import argparse
    import http.server
    import socketserver

    parser = argparse.ArgumentParser(
        description="Verify the React widget's AudioWorklet capture path."
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force the ScriptProcessorNode fallback path (AudioWorklet disabled) "
        "and assert the round-trip still works.",
    )
    args = parser.parse_args()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    server, thread = _start_server(8896)

    # Serve the react package dir (dist + node_modules UMD builds).
    react_dir = os.path.join(ROOT, "packages", "react")
    from functools import partial

    httpd = socketserver.TCPServer(
        ("127.0.0.1", 8897), partial(Handler, directory=react_dir)
    )
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    # Write a test page that loads the built CJS widget with a require stub.
    test_html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<div id="root"></div>
<script src="node_modules/react/umd/react.development.js"></script>
<script src="node_modules/react-dom/umd/react-dom.development.js"></script>
<script>
window.__module = { exports: {} };
window.module = window.__module;
window.require = function (name) {
  if (name === 'react') return window.React;
  if (name === 'react-dom') return window.ReactDOM;
  if (name === 'react/jsx-runtime' || name === 'react/jsx-dev-runtime') {
    return {
      jsx: window.React.createElement,
      jsxs: window.React.createElement,
      Fragment: window.React.Fragment,
    };
  }
  return undefined;
};
</script>
<script src="dist/index.js"></script>
<script>
window.__VoiceWidget = window.__module.exports.VoiceWidget || window.__module.exports;
document.title = 'widget=' + (typeof window.__VoiceWidget);
</script>
</body>
</html>
"""
    test_path = os.path.join(react_dir, "_widget_test.html")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_html)

    try:
        from playwright.sync_api import sync_playwright

        failures = 0
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required",
                ]
            )
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

                // Instrument AudioWorkletNode to prove the worklet path (not
                // the ScriptProcessor fallback) is actually exercised.
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
            if args.fallback:
                # Force the ScriptProcessorNode fallback path.
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

            page.goto("http://127.0.0.1:8897/_widget_test.html")
            page.wait_for_timeout(3000)

            title = page.title()
            print(f"widget loaded: {title}")
            assert "widget=function" in title or "widget=object" in title

            # Mount the widget.
            page.evaluate(
                """() => {
                    const VoiceWidget = window.__VoiceWidget.default || window.__VoiceWidget;
                    const root = document.getElementById('root');
                    window.ReactDOM.render(
                        window.React.createElement(VoiceWidget, {
                            serverUrl: 'ws://127.0.0.1:8896/ws',
                            size: 100,
                        }),
                        root
                    );
                    window.__mounted = true;
                }"""
            )
            page.wait_for_timeout(3000)
            print(f"mounted: {page.evaluate('window.__mounted')}")
            print(f"buttons rendered: {page.locator('button').count()}")

            # Hold-to-talk via the widget button (mousedown → wait → mouseup).
            btn = page.locator("button").first
            bb = btn.bounding_box()
            page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
            page.mouse.down()
            page.wait_for_timeout(2500)
            page.mouse.up()
            page.wait_for_timeout(4000)

            ws_recv = page.evaluate("window.__wsRecv")
            types = set()
            for m in ws_recv:
                if m.startswith("recv:{"):
                    try:
                        types.add(json.loads(m[len("recv:"):]).get("type"))
                    except ValueError:
                        pass

            required = {"listening_started", "listening_stopped", "transcript", "audio_chunk"}
            missing = required - types
            print(f"message types observed: {sorted(types)}")
            print(f"missing: {missing or 'none'}")

            deprecations = [t for _, t in console_msgs if "ScriptProcessorNode" in t]
            print(f"ScriptProcessorNode deprecations: {deprecations or 'none'}")

            aw = page.evaluate("typeof AudioWorkletNode !== 'undefined'")
            aw_used = page.evaluate("window.__awWorkletUsed")
            print(f"AudioWorkletNode supported: {aw}")
            print(f"AudioWorklet capture node used: {aw_used}")

            if args.fallback:
                # Fallback mode: round-trip must work and the worklet must NOT
                # have been used.
                if missing or aw_used:
                    failures += 1
            else:
                if missing or deprecations or not aw or not aw_used:
                    failures += 1

            browser.close()

        print(f"\nRESULT: {'PASSED' if failures == 0 else 'FAILED'}")
        return 1 if failures else 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        httpd.shutdown()
        http_thread.join(timeout=5)
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    raise SystemExit(main())
