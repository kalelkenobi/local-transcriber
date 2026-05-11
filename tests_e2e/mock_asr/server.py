"""
Minimal mock ASR server for e2e testing.

Returns a canned transcription response for any audio file posted to
/v1/audio/transcriptions (mimicking the vLLM OpenAI-compatible endpoint).
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json


class MockASRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            # Read and discard the body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length:
                self.rfile.read(content_length)

            response = {"text": "mock transcription output"}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress logs


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9000), MockASRHandler)
    print("Mock ASR server listening on :9000", flush=True)
    server.serve_forever()
