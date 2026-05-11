"""Tests for RemoteASRBackend HTTP behavior."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import httpx

from local_transcriber.backend import ASRRequestError, RemoteASRBackend


def _backend_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = None,
    max_retries: int = 0,
    backoff_base: float = 0.0,
    model: str = "mock-model",
) -> RemoteASRBackend:
    return RemoteASRBackend(
        base_url="http://asr.test",
        model=model,
        api_key=api_key,
        timeout=5.0,
        max_retries=max_retries,
        backoff_base=backoff_base,
        transport=httpx.MockTransport(handler),
    )


class TestBackendBasics(unittest.IsolatedAsyncioTestCase):
    async def test_init_requires_url_and_model(self):
        with self.assertRaises(ValueError):
            RemoteASRBackend(base_url="", model="m")
        with self.assertRaises(ValueError):
            RemoteASRBackend(base_url="http://x", model="")

    async def test_url_property(self):
        b = _backend_with_handler(lambda r: httpx.Response(200, json={"text": ""}))
        try:
            self.assertEqual(b.url, "http://asr.test/v1/audio/transcriptions")
            self.assertEqual(b.model, "mock-model")
        finally:
            await b.close()

    async def test_successful_transcription(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(200, json={"text": "  hello world  "})

        b = _backend_with_handler(handler, api_key="secret")
        try:
            text = await b.transcribe(b"RIFF....fake", "en")
        finally:
            await b.close()

        self.assertEqual(text, "hello world")
        req = captured["request"]
        self.assertEqual(req.method, "POST")
        self.assertTrue(str(req.url).endswith("/v1/audio/transcriptions"))
        self.assertEqual(req.headers.get("authorization"), "Bearer secret")
        # The request body is multipart; verify model + language tokens appear.
        body = req.content.decode("utf-8", errors="replace")
        self.assertIn("mock-model", body)
        self.assertIn('name="model"', body)
        self.assertIn('name="language"', body)
        self.assertIn("en", body)
        self.assertIn('name="file"', body)
        self.assertIn("segment.wav", body)

    async def test_no_auth_header_when_no_key(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(200, json={"text": "ok"})

        b = _backend_with_handler(handler)
        try:
            await b.transcribe(b"x" * 16, "en")
        finally:
            await b.close()
        self.assertNotIn("authorization", captured["request"].headers)


class TestBackendErrors(unittest.IsolatedAsyncioTestCase):
    async def test_4xx_raises_immediately(self):
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(400, text="bad request")

        b = _backend_with_handler(handler, max_retries=3)
        try:
            with self.assertRaises(ASRRequestError):
                await b.transcribe(b"x" * 16, "en")
        finally:
            await b.close()
        self.assertEqual(call_count["n"], 1)

    async def test_5xx_retries_then_raises(self):
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(503, text="service unavailable")

        b = _backend_with_handler(handler, max_retries=2)
        try:
            with self.assertRaises(ASRRequestError):
                await b.transcribe(b"x" * 16, "en")
        finally:
            await b.close()
        self.assertEqual(call_count["n"], 3)  # 1 initial + 2 retries

    async def test_5xx_recovers_on_retry(self):
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] < 2:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(200, json={"text": "recovered"})

        b = _backend_with_handler(handler, max_retries=2)
        try:
            text = await b.transcribe(b"x" * 16, "en")
        finally:
            await b.close()
        self.assertEqual(text, "recovered")
        self.assertEqual(call_count["n"], 2)

    async def test_transport_error_retries(self):
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise httpx.ConnectError("conn refused")
            return httpx.Response(200, json={"text": "ok"})

        b = _backend_with_handler(handler, max_retries=2)
        try:
            text = await b.transcribe(b"x" * 16, "en")
        finally:
            await b.close()
        self.assertEqual(text, "ok")
        self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
