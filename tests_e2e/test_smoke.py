"""End-to-end smoke test: run the container against example_recording/."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from tests_e2e.conftest import (
    EXAMPLE_RECORDING,
    _container_host_address,
)
from tests_e2e.mock_asr.server import MockASRServer


def _run_container(
    runtime: str,
    image: str,
    name: str,
    mounts: list[tuple[Path, str]],
    env: dict[str, str],
    extra_args: list[str],
    cli_args: list[str],
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = [runtime, "run", "--rm", "--name", name]
    for host_path, container_path in mounts:
        cmd += ["-v", f"{host_path}:{container_path}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += extra_args
    cmd += [image]
    cmd += cli_args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


TIMESTAMP_LINE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{2} \S+$", re.MULTILINE)


def test_smoke_single_session(
    runtime: str,
    container_image: str,
    mock_asr: tuple[str, int, MockASRServer],
    tmp_path: Path,
) -> None:
    """Transcribe example_recording/ through the container + mock ASR.

    Asserts:
      - container exits 0,
      - transcript.json + transcript.txt are written,
      - segments are non-empty and sorted by absolute start time,
      - speaker matches the manifest participant,
      - mock ASR was called at least once.
    """
    _, port, server = mock_asr

    # Copy the example so the container can write transcript files into it.
    session = tmp_path / "example_recording"
    shutil.copytree(EXAMPLE_RECORDING, session)
    # Be permissive — the container runs as root by default and writes back
    # to the host mount.
    session.chmod(0o777)

    host_addr, extra_args = _container_host_address(runtime)
    name = f"lt-e2e-smoke-{uuid.uuid4().hex[:8]}"

    result = _run_container(
        runtime=runtime,
        image=container_image,
        name=name,
        mounts=[(session, "/in")],
        env={
            "TRANSCRIBE_URL": f"http://{host_addr}:{port}",
            "TRANSCRIBE_MODEL": "mock-model",
            "LOG_LEVEL": "INFO",
        },
        extra_args=extra_args,
        cli_args=["/in"],
    )

    assert result.returncode == 0, (
        f"container exit={result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    transcript_json = session / "transcript.json"
    transcript_txt = session / "transcript.txt"
    assert transcript_json.exists(), "transcript.json was not written"
    assert transcript_txt.exists(), "transcript.txt was not written"

    data = json.loads(transcript_json.read_text())
    assert data["session_id"] == "2026-05-11_21-47-37"
    assert data["model"] == "mock-model"
    assert data["language"] == "en"
    assert len(data["segments"]) > 0, "no segments produced"

    starts = [s["start"] for s in data["segments"]]
    assert starts == sorted(starts), "segments not sorted by start time"

    speakers = {s["speaker"] for s in data["segments"]}
    assert speakers == {"Riccardo"}, f"unexpected speakers: {speakers}"

    for seg in data["segments"]:
        assert seg["text"].startswith("mock-"), seg
        assert seg["end"] > seg["start"]
        assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{2}$", seg["start_absolute"])

    txt = transcript_txt.read_text()
    assert TIMESTAMP_LINE.search(txt), f"no timestamp lines in:\n{txt}"
    assert "Riccardo" in txt
    # Each segment in JSON should appear in TXT.
    assert txt.count("Riccardo") == len(data["segments"])

    assert server.call_count >= len(data["segments"]), (
        "mock ASR was called fewer times than the number of returned segments"
    )


def test_smoke_recursive(
    runtime: str,
    container_image: str,
    mock_asr: tuple[str, int, MockASRServer],
    tmp_path: Path,
) -> None:
    """Two-session recursive run produces a transcript for each session."""
    _, port, server = mock_asr

    root = tmp_path / "recordings"
    root.mkdir()
    for suffix in ("a", "b"):
        target = root / f"session-{suffix}"
        shutil.copytree(EXAMPLE_RECORDING, target)
        target.chmod(0o777)

    host_addr, extra_args = _container_host_address(runtime)
    name = f"lt-e2e-recursive-{uuid.uuid4().hex[:8]}"

    result = _run_container(
        runtime=runtime,
        image=container_image,
        name=name,
        mounts=[(root, "/in")],
        env={
            "TRANSCRIBE_URL": f"http://{host_addr}:{port}",
            "TRANSCRIBE_MODEL": "mock-model",
        },
        extra_args=extra_args,
        cli_args=["/in", "--recursive"],
    )

    assert result.returncode == 0, (
        f"container exit={result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    for suffix in ("a", "b"):
        target = root / f"session-{suffix}"
        assert (target / "transcript.json").exists(), suffix
        assert (target / "transcript.txt").exists(), suffix


def test_smoke_exit_code_when_no_sessions(
    runtime: str,
    container_image: str,
    mock_asr: tuple[str, int, MockASRServer],
    tmp_path: Path,
) -> None:
    """A non-session dir with no manifest.json should exit 1."""
    _, port, _ = mock_asr
    empty = tmp_path / "empty"
    empty.mkdir()
    empty.chmod(0o777)

    host_addr, extra_args = _container_host_address(runtime)
    name = f"lt-e2e-empty-{uuid.uuid4().hex[:8]}"

    result = _run_container(
        runtime=runtime,
        image=container_image,
        name=name,
        mounts=[(empty, "/in")],
        env={
            "TRANSCRIBE_URL": f"http://{host_addr}:{port}",
            "TRANSCRIBE_MODEL": "mock-model",
        },
        extra_args=extra_args,
        cli_args=["/in"],
    )

    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "No sessions found" in (result.stdout + result.stderr)
