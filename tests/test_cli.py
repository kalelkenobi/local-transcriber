"""Tests for the click CLI plumbing.

`transcribe_session` is patched so these tests don't depend on ffmpeg,
the ONNX model, or a real ASR server. `SileroVAD` is also patched to
avoid the onnxruntime cost.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from local_transcriber import cli as cli_mod
from local_transcriber.pipeline import SessionResult


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "example_recording"


def _write_session(dirpath: Path, session_id: str) -> Path:
    session = dirpath / session_id
    session.mkdir()
    session.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "start_epoch": 1.0,
                "end_epoch": 2.0,
                "participants": [],
            }
        )
    )
    return session


class FakeVAD:
    def __init__(self, *args, **kwargs):
        pass


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_missing_required_flags(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.runner.invoke(cli_mod.main, [td])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--api-url", result.output + (result.stderr or ""))

    def test_path_with_no_manifest_no_recursive_exits(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.runner.invoke(
                cli_mod.main,
                [
                    td,
                    "--api-url", "http://x",
                    "--model", "m",
                ],
            )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("No sessions found", result.output)

    def test_single_session_invokes_pipeline(self):
        async def fake_transcribe(session_dir, **kwargs):
            return SessionResult(
                session_id=session_dir.name,
                ok=True,
                num_segments=2,
                num_speakers=1,
                output_dir=session_dir,
            )

        with tempfile.TemporaryDirectory() as td:
            session = _write_session(Path(td), "s1")
            with patch.object(cli_mod, "SileroVAD", FakeVAD), patch.object(
                cli_mod, "transcribe_session", side_effect=fake_transcribe
            ) as mock_pipeline:
                result = self.runner.invoke(
                    cli_mod.main,
                    [
                        str(session),
                        "--api-url", "http://x",
                        "--model", "m",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertEqual(mock_pipeline.call_count, 1)
            self.assertIn("[OK] s1", result.output)

    def test_recursive_runs_each_session(self):
        async def fake_transcribe(session_dir, **kwargs):
            return SessionResult(
                session_id=session_dir.name,
                ok=True,
                num_segments=1,
                num_speakers=1,
                output_dir=session_dir,
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_session(root, "s1")
            _write_session(root, "s2")
            # also a non-session sibling — should be ignored
            (root / "junk").mkdir()
            with patch.object(cli_mod, "SileroVAD", FakeVAD), patch.object(
                cli_mod, "transcribe_session", side_effect=fake_transcribe
            ) as mock_pipeline:
                result = self.runner.invoke(
                    cli_mod.main,
                    [
                        str(root),
                        "--recursive",
                        "--api-url", "http://x",
                        "--model", "m",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertEqual(mock_pipeline.call_count, 2)

    def test_partial_failure_exits_2(self):
        results_iter = iter(
            [
                SessionResult("s1", True, 1, 1, Path("/tmp/s1")),
                SessionResult(
                    "s2", False, 0, 0, Path("/tmp/s2"), error="boom"
                ),
            ]
        )

        async def fake_transcribe(session_dir, **kwargs):
            return next(results_iter)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_session(root, "s1")
            _write_session(root, "s2")
            with patch.object(cli_mod, "SileroVAD", FakeVAD), patch.object(
                cli_mod, "transcribe_session", side_effect=fake_transcribe
            ):
                result = self.runner.invoke(
                    cli_mod.main,
                    [
                        str(root),
                        "--recursive",
                        "--api-url", "http://x",
                        "--model", "m",
                    ],
                )
        self.assertEqual(result.exit_code, 2, msg=result.output)
        self.assertIn("[OK] s1", result.output)
        self.assertIn("[FAIL (boom)] s2", result.output)

    def test_env_var_substitution(self):
        async def fake_transcribe(session_dir, **kwargs):
            return SessionResult(
                session_id=session_dir.name,
                ok=True,
                num_segments=0,
                num_speakers=0,
                output_dir=session_dir,
            )

        with tempfile.TemporaryDirectory() as td:
            session = _write_session(Path(td), "s1")
            with patch.object(cli_mod, "SileroVAD", FakeVAD), patch.object(
                cli_mod, "transcribe_session", side_effect=fake_transcribe
            ):
                result = self.runner.invoke(
                    cli_mod.main,
                    [str(session)],
                    env={
                        "TRANSCRIBE_URL": "http://from-env",
                        "TRANSCRIBE_MODEL": "env-model",
                    },
                )
        self.assertEqual(result.exit_code, 0, msg=result.output)


class TestResolveSileroPath(unittest.TestCase):
    def test_explicit_arg_wins(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "explicit.onnx"
            p.touch()
            result = cli_mod._resolve_silero_path(str(p))
            self.assertEqual(result, p)

    def test_dev_fallback_to_repo_models(self):
        # When neither explicit nor env nor container path exists, the dev
        # fallback should point at the repo-local models/silero_vad.onnx.
        with patch.dict("os.environ", {}, clear=False) as _env:
            import os as _os
            _os.environ.pop("SILERO_VAD_PATH", None)
            with patch.object(
                cli_mod, "_DEFAULT_SILERO_PATH", Path("/nope/silero_vad.onnx")
            ):
                result = cli_mod._resolve_silero_path(None)
            self.assertEqual(
                result.name, "silero_vad.onnx"
            )


if __name__ == "__main__":
    unittest.main()
