"""Remote ASR backend — OpenAI-compatible /v1/audio/transcriptions client.

The local-transcriber container never bundles a model. Users point it at
any HTTP server that implements the OpenAI-compatible transcriptions API,
local or remote (vLLM, mlx-whisper-server, etc.).
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class ASRRequestError(RuntimeError):
    """Raised when the ASR backend returns a non-recoverable error."""


class RemoteASRBackend:
    """HTTP client for OpenAI-compatible /v1/audio/transcriptions endpoints.

    Retries on transport errors and HTTP 5xx with exponential backoff.
    4xx responses are surfaced immediately (caller has a bad request).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = float(backoff_base)

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client_kwargs: dict[str, object] = {
            "timeout": timeout,
            "headers": headers,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**client_kwargs)

    @property
    def model(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        return f"{self._base_url}/v1/audio/transcriptions"

    async def transcribe(self, wav_bytes: bytes, language: str) -> str:
        """POST a WAV segment and return the transcribed text (may be '')."""
        logger.debug(
            "ASR request: url=%s model=%s language=%s bytes=%d",
            self.url,
            self._model,
            language,
            len(wav_bytes),
        )
        files = {"file": ("segment.wav", wav_bytes, "audio/wav")}
        data = {"model": self._model, "language": language}

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self.url, files=files, data=data)
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "ASR transport error on attempt %d/%d: %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
            else:
                if response.status_code < 400:
                    payload = response.json()
                    text = str(payload.get("text", "")).strip()
                    logger.debug(
                        "ASR response: status=%d text_chars=%d",
                        response.status_code,
                        len(text),
                    )
                    return text
                if 500 <= response.status_code < 600:
                    last_error = ASRRequestError(
                        f"ASR server returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    logger.warning(
                        "ASR server %d on attempt %d/%d",
                        response.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                    )
                else:
                    # 4xx — bad request, don't retry.
                    raise ASRRequestError(
                        f"ASR request failed ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
            if attempt < self._max_retries:
                await asyncio.sleep(self._backoff_base * (2 ** attempt))

        raise ASRRequestError(
            f"ASR request failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
