"""Tests for the transcriber HTTP API."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestTranscriberAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch env before importing the app
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "JOBS_DIR": self.tmpdir,
                "TRANSCRIBE_BACKEND": "local",
                "LOG_LEVEL": "WARNING",
            },
        )
        self.env_patcher.start()

        # Import after env patch
        import transcriber.main as main_mod

        main_mod.JOBS_DIR = Path(self.tmpdir)
        main_mod.registry = main_mod.JobRegistry(Path(self.tmpdir))
        self.client = TestClient(main_mod.app, raise_server_exceptions=False)

    def tearDown(self):
        self.env_patcher.stop()

    def test_list_jobs_empty(self):
        resp = self.client.get("/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["jobs"], [])

    def test_submit_bad_extension(self):
        resp = self.client.post(
            "/transcribe",
            files={"file": ("test.mp3", b"fake", "audio/mpeg")},
            data={"language": "en"},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("transcriber.main._process_job", new_callable=AsyncMock)
    def test_submit_wav(self, mock_process):
        mock_process.return_value = None
        resp = self.client.post(
            "/transcribe",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"language": "en"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "queued")

    @patch("transcriber.main._process_job", new_callable=AsyncMock)
    def test_submit_zip(self, mock_process):
        mock_process.return_value = None
        resp = self.client.post(
            "/transcribe",
            files={"file": ("session.zip", b"PK" + b"\x00" * 100, "application/zip")},
            data={"language": "fr"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job_id", data)

    def test_get_job_not_found(self):
        resp = self.client.get("/transcribe/nonexistent")
        self.assertEqual(resp.status_code, 404)

    @patch("transcriber.main._process_job", new_callable=AsyncMock)
    def test_get_job_status(self, mock_process):
        mock_process.return_value = None
        # Create a job
        resp = self.client.post(
            "/transcribe",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"language": "en"},
        )
        job_id = resp.json()["job_id"]

        # Get status
        resp = self.client.get(f"/transcribe/{job_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["job_id"], job_id)
        self.assertIn(data["status"], ["queued", "processing", "completed", "failed"])

    def test_get_transcript_not_found(self):
        resp = self.client.get("/transcribe/nonexistent/transcript")
        self.assertEqual(resp.status_code, 404)

    @patch("transcriber.main._process_job", new_callable=AsyncMock)
    def test_get_transcript_not_ready(self, mock_process):
        mock_process.return_value = None
        resp = self.client.post(
            "/transcribe",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"language": "en"},
        )
        job_id = resp.json()["job_id"]

        resp = self.client.get(f"/transcribe/{job_id}/transcript")
        self.assertEqual(resp.status_code, 202)


if __name__ == "__main__":
    unittest.main()
