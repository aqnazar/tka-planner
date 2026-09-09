"""The HTTP adapter: sockets in, :mod:`tka_planner.server.api` out.

Deliberately the standard library. The shipped dependency set is numpy and manifold3d,
and a local planner serving one surgeon on one machine gains nothing from an ASGI stack
that it does not gain from ``ThreadingHTTPServer``. What it loses is worth avoiding: a
clinical tool whose install can fail on a hospital machine because a transitive
dependency changed.

Threaded rather than single-threaded for one specific reason. A commit runs booleans
that can take seconds, and during that time the browser must still be able to fetch the
meshes it is already drawing. A single-threaded server would make the page appear to
hang for the duration of the cut.

Bound to the loopback interface only. A planner holding patient geometry has no business
listening on a network interface, and making that the default rather than an option
means it cannot be forgotten.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import Api, Binary

__all__ = ["PlannerServer", "WEB_ROOT"]

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"

# Serving the wrong type for a JavaScript module makes the browser refuse it outright,
# and Windows registers .js from the registry, where it is frequently wrong.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("text/css", ".css")

MAX_BODY_BYTES = 4 << 20


class PlannerServer:
    """The application: a session manager, the routes, and a socket in front."""

    def __init__(self, manager, *, host: str = "127.0.0.1", port: int = 8731,
                 web_root=None):
        self.api = Api(manager)
        self.web_root = Path(web_root or WEB_ROOT)
        handler = _make_handler(self.api, self.web_root)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = None

    @property
    def address(self) -> tuple:
        return self._server.server_address

    @property
    def url(self) -> str:
        host, port = self.address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> str:
        """Serve in a background thread and return the URL."""
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="tka-planner", daemon=True
        )
        self._thread.start()
        return self.url

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _make_handler(api: Api, web_root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tka-planner"
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - the base class names it
            if self.path.split("?")[0].startswith("/api/"):
                self._api("GET", None)
            else:
                self._static()

        def do_POST(self):  # noqa: N802
            self._api("POST", self._body())

        def do_DELETE(self):  # noqa: N802
            self._api("DELETE", None)

        # -- helpers ---------------------------------------------------

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}

        def _api(self, method, body):
            status, payload = api.dispatch(method, self.path, body)
            if isinstance(payload, Binary):
                headers = {"Content-Type": payload.content_type}
                if payload.immutable:
                    # The path is a content hash, so a cached copy can never be stale.
                    headers["Cache-Control"] = "public, max-age=31536000, immutable"
                self._send(status, payload.data, headers)
                return
            encoded = json.dumps(payload).encode("utf-8")
            self._send(status, encoded, {
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            })

        def _static(self):
            relative = self.path.split("?")[0].lstrip("/") or "index.html"
            target = (web_root / relative).resolve()
            try:
                target.relative_to(web_root.resolve())
            except ValueError:
                # A path that escapes the web root, whether by traversal or by symlink.
                self._send(403, b"forbidden", {"Content-Type": "text/plain"})
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                self._send(404, b"not found", {"Content-Type": "text/plain"})
                return

            kind, _ = mimetypes.guess_type(target.name)
            self._send(200, target.read_bytes(), {
                "Content-Type": kind or "application/octet-stream",
                # The viewer is edited while it is being used; a cached copy of it is
                # more confusing than a re-read of a few kilobytes is slow.
                "Cache-Control": "no-cache",
            })

        def _send(self, status, data: bytes, headers: dict):
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # The browser navigated away mid-transfer. Normal, and not worth a
                # traceback on the console a surgeon is looking at.
                pass

        def log_message(self, fmt, *args):
            # The default handler writes every request to stderr, which buries the one
            # line that matters: the URL to open.
            pass

    return Handler


def free_port() -> int:
    """An unused localhost port, for tests and for a second instance."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
