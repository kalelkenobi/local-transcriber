"""
vLLM ASR backend — sends audio to a remote vLLM server via the
OpenAI-compatible /v1/audio/transcriptions endpoint.
"""

import logging

import httpx

logger = logging.getLogger(__name__)


class VLLMBackend:
    """ASR backend that calls a vLLM server over HTTP."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        """
        Transcribe WAV audio by POSTing to vLLM.

        Args:
            wav_bytes: Complete WAV file bytes (with header).
            language: Language code (e.g. "en", "fr").

        Returns:
            Transcribed text.
        """
        client = self._get_client()
        url = f"{self._base_url}/v1/audio/transcriptions"

        files = {"file": ("segment.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "CohereLabs/cohere-transcribe-03-2026",
            "language": language,
        }

        response = await client.post(url, files=files, data=data)
        response.raise_for_status()

        result = response.json()
        return result.get("text", "").strip()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
