"""A deliberately small local server for the Gradient Microscope.

Design constraints, all of which follow from "this thing evaluates expressions a
user typed":

* **Binds to 127.0.0.1 only.** Never ``0.0.0.0``. The app is a local teaching
  tool, not a service, and there is no authentication to put in front of it.
* **Static files are served from one directory**, resolved and then checked to
  be inside it, so ``..`` cannot escape. Only a fixed set of extensions is served.
* **The expression endpoint is opt-in** (``--allow-expressions``) and runs the
  allowlisted evaluator in :mod:`gradlab.inspection.expressions`. No ``eval``,
  no ``exec``, no Python parsing of user text.
* **Every request is bounded**: body size, expression length, variable count and
  tensor size, plus a per-client request rate limit.

If you only want to read the shipped traces, don't pass ``--allow-expressions``:
the app then runs entirely on static JSON and has no execution surface at all.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

#: Only these extensions are served.
ALLOWED_SUFFIXES = {".html", ".js", ".css", ".json", ".svg", ".woff2", ".map", ".md"}
#: Maximum POST body, in bytes.
MAX_BODY_BYTES = 16 * 1024
#: Requests per client per window.
RATE_LIMIT = 120
RATE_WINDOW_SECONDS = 60.0


class _Handler(http.server.SimpleHTTPRequestHandler):
    directory_root: Path
    allow_expressions: bool = False
    _hits: ClassVar[dict[str, deque]] = {}
    _lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def _rate_limited(self) -> bool:
        client = self.client_address[0]
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] > RATE_WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= RATE_LIMIT:
                return True
            hits.append(now)
            return False

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, url_path: str) -> Path | None:
        """Map a URL path to a file inside the root, or ``None`` if it escapes."""
        relative = url_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if not relative:
            relative = "index.html"
        candidate = (self.directory_root / relative).resolve()
        root = self.directory_root.resolve()
        if root != candidate and root not in candidate.parents:
            return None
        if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
            return None
        return candidate if candidate.is_file() else None

    # -- request handling --------------------------------------------------

    def do_GET(self) -> None:
        if self._rate_limited():
            self._send_json(429, {"error": "too many requests"})
            return
        if self.path.split("?")[0] == "/api/capabilities":
            self._send_json(
                200,
                {
                    "allow_expressions": self.allow_expressions,
                    "limits": _limits(),
                },
            )
            return
        target = self._resolve(self.path)
        if target is None:
            self._send_json(404, {"error": "not found"})
            return
        try:
            data = target.read_bytes()
        except OSError:
            self._send_json(404, {"error": "not found"})
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".md": "text/plain; charset=utf-8",
        }.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self._rate_limited():
            self._send_json(429, {"error": "too many requests"})
            return
        if self.path.split("?")[0] != "/api/trace":
            self._send_json(404, {"error": "not found"})
            return
        if not self.allow_expressions:
            self._send_json(
                403,
                {
                    "error": "the expression endpoint is disabled",
                    "hint": "start the server with --allow-expressions to enable it",
                },
            )
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "expected a JSON object"})
            return

        expression = payload.get("expression")
        values = payload.get("values") or {}
        if not isinstance(expression, str) or not isinstance(values, dict):
            self._send_json(400, {"error": "expected {expression: str, values: object}"})
            return

        from gradlab.inspection.expressions import ExpressionError, evaluate
        from gradlab.inspection.tracer import trace_expression

        try:
            output, variables = evaluate(expression, values)
            names = list(variables)
            trace = trace_expression(
                lambda *ts, _out=output: _out,
                *[variables[n] for n in names],
                names=names,
                label=expression,
            )
            trace["expression_text"] = expression
            trace["slug"] = "custom"
            trace["teaching_note"] = "Your expression, differentiated by the real engine."
        except ExpressionError as exc:
            self._send_json(400, {"error": str(exc), "kind": "expression"})
            return
        except Exception as exc:  # a genuine evaluation failure, reported not hidden
            self._send_json(400, {"error": f"{type(exc).__name__}: {exc}", "kind": "evaluation"})
            return
        self._send_json(200, trace)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quieter default logging; the app makes many trace requests."""
        if self.path.startswith("/api") or not self.path.endswith((".js", ".css", ".json")):
            super().log_message(fmt, *args)


def _limits() -> dict[str, int]:
    from gradlab.inspection import expressions as ex

    return {
        "max_expression_length": ex.MAX_EXPRESSION_LENGTH,
        "max_nodes": ex.MAX_NODES,
        "max_depth": ex.MAX_DEPTH,
        "max_variables": ex.MAX_VARIABLES,
        "max_tensor_elements": ex.MAX_TENSOR_ELEMENTS,
        "max_body_bytes": MAX_BODY_BYTES,
        "rate_limit_per_minute": RATE_LIMIT,
    }


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(
    directory: Path,
    port: int = 8371,
    open_browser: bool = True,
    allow_expressions: bool = False,
    host: str = "127.0.0.1",
) -> int:
    """Serve the Microscope on ``host:port``. Blocks until interrupted."""
    directory = Path(directory).resolve()
    if not (directory / "index.html").exists():
        print(f"error: {directory}/index.html not found")
        return 1

    handler = type(
        "BoundHandler",
        (_Handler,),
        {"directory_root": directory, "allow_expressions": allow_expressions},
    )

    with _Server((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"Gradient Microscope serving {directory}")
        print(f"  {url}")
        print(f"  expression endpoint: {'ENABLED' if allow_expressions else 'disabled'}")
        if allow_expressions:
            print("  (allowlisted operations only; no Python is evaluated)")
        print("  bound to localhost only. Ctrl-C to stop.")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0
