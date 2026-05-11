"""local-transcriber CLI entry point."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click

from .backend import RemoteASRBackend
from .pipeline import SessionResult, transcribe_session
from .session import iter_sessions
from .vad import SileroVAD

logger = logging.getLogger(__name__)

_DEFAULT_SILERO_PATH = Path("/app/models/silero_vad.onnx")


def _resolve_silero_path(explicit: str | None) -> Path:
    """Resolve the Silero VAD ONNX path with a dev fallback."""
    if explicit:
        return Path(explicit)
    env_path = os.environ.get("SILERO_VAD_PATH")
    if env_path:
        return Path(env_path)
    if _DEFAULT_SILERO_PATH.exists():
        return _DEFAULT_SILERO_PATH
    return (
        Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
    )


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="When PATH is a parent dir, process every child containing manifest.json.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory for transcripts. Default: write alongside each session.",
)
@click.option("--language", "-l", default="en", show_default=True)
@click.option(
    "--api-url",
    envvar="TRANSCRIBE_URL",
    required=True,
    help="Base URL of an OpenAI-compatible ASR server (env TRANSCRIBE_URL).",
)
@click.option(
    "--api-key",
    envvar="TRANSCRIBE_API_KEY",
    default=None,
    help="Optional bearer token (env TRANSCRIBE_API_KEY).",
)
@click.option(
    "--model",
    envvar="TRANSCRIBE_MODEL",
    required=True,
    help="Model name passed to the ASR server (env TRANSCRIBE_MODEL).",
)
@click.option(
    "--vad-threshold",
    type=float,
    default=0.5,
    show_default=True,
    envvar="TRANSCRIBE_VAD_THRESHOLD",
)
@click.option("--vad-min-speech-ms", type=int, default=250, show_default=True)
@click.option("--vad-min-silence-ms", type=int, default=500, show_default=True)
@click.option("--concurrency", type=int, default=4, show_default=True)
@click.option("--timeout", type=float, default=300.0, show_default=True)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    envvar="LOG_LEVEL",
)
@click.option(
    "--silero-vad-path",
    default=None,
    envvar="SILERO_VAD_PATH",
    help="Path to Silero VAD ONNX model (env SILERO_VAD_PATH).",
)
def main(
    path: Path,
    recursive: bool,
    output_dir: Path | None,
    language: str,
    api_url: str,
    api_key: str | None,
    model: str,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    concurrency: int,
    timeout: float,
    log_level: str,
    silero_vad_path: str | None,
) -> None:
    """Transcribe a recording session (or all sessions under PATH).

    PATH is either a single session directory (containing manifest.json)
    or, with --recursive, a parent directory whose immediate children are
    session directories.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sessions = list(iter_sessions(path, recursive))
    if not sessions:
        click.echo(
            f"No sessions found under {path}. "
            f"Expected manifest.json in PATH, or pass --recursive to scan "
            f"its immediate subdirectories.",
            err=True,
        )
        sys.exit(1)

    silero_path = _resolve_silero_path(silero_vad_path)
    try:
        vad = SileroVAD(
            silero_path,
            threshold=vad_threshold,
            min_speech_ms=vad_min_speech_ms,
            min_silence_ms=vad_min_silence_ms,
        )
    except FileNotFoundError as exc:
        click.echo(f"Silero VAD model not available: {exc}", err=True)
        sys.exit(1)

    results = asyncio.run(
        _run_all(
            sessions=sessions,
            vad=vad,
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            language=language,
            output_dir=output_dir,
            concurrency=concurrency,
        )
    )

    for r in results:
        status = "OK" if r.ok else f"FAIL ({r.error})"
        click.echo(
            f"[{status}] {r.session_id}: {r.num_segments} segments, "
            f"{r.num_speakers} speakers -> {r.output_dir}"
        )

    if any(not r.ok for r in results):
        sys.exit(2)


async def _run_all(
    *,
    sessions: list[Path],
    vad: SileroVAD,
    api_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
    language: str,
    output_dir: Path | None,
    concurrency: int,
) -> list[SessionResult]:
    """Drive `transcribe_session()` for each discovered session.

    The backend is created and closed inside this coroutine so its
    underlying httpx.AsyncClient lives entirely within one event loop.
    """
    backend = RemoteASRBackend(
        base_url=api_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    try:
        results: list[SessionResult] = []
        for session_dir in sessions:
            out = output_dir / session_dir.name if output_dir else None
            result = await transcribe_session(
                session_dir,
                backend=backend,
                vad=vad,
                language=language,
                output_dir=out,
                concurrency=concurrency,
            )
            results.append(result)
        return results
    finally:
        await backend.close()


if __name__ == "__main__":
    main()
