"""Tiny mock OpenAI-compatible ASR server used by the e2e suite.

Responds to `POST /v1/audio/transcriptions` with
`{"text": "mock-<N>"}` where N is a per-server incrementing counter.
This is enough to verify the pipeline end-to-end without needing a real
ASR model.

Usage as a library (preferred):

    from tests_e2e.mock_asr.server import start_mock_asr
    with start_mock_asr() as (host, port, server):
        ...

Usage as a standalone script:

    python -m tests_e2e.mock_asr.server 0.0.0.0 9123
"""

from __future__ import annotations

import contextlib
import json
import socketserver
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    server: "MockASRServer"  # type: ignore[assignment]

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/v1/audio/transcriptions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        # Drain the request body so the client doesn't see a broken pipe.
        if length:
            self.rfile.read(length)

        with self.server.lock:
            self.server.call_count += 1
            idx = self.server.call_count

        body = json.dumps({"text": f"mock-{idx}"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Silence access logs unless explicitly enabled.
        return


class MockASRServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _Handler)
        self.call_count = 0
        self.lock = threading.Lock()


@contextlib.contextmanager
def start_mock_asr(
    host: str = "0.0.0.0", port: int = 0
) -> Iterator[tuple[str, int, MockASRServer]]:
    """Run a MockASRServer in a background thread.

    Yields `(host, port, server)`. `port` is the actually-bound port (useful
    when `port=0` was passed).
    """
    server = MockASRServer((host, port))
    bound_host, bound_port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield bound_host, bound_port, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def main(argv: list[str]) -> int:
    host = argv[1] if len(argv) > 1 else "0.0.0.0"
    port = int(argv[2]) if len(argv) > 2 else 9123
    with start_mock_asr(host, port) as (h, p, server):
        print(f"mock ASR listening on {h}:{p}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
