"""Tests for job registry."""

import json
import tempfile
import unittest
from pathlib import Path

from transcriber.jobs import Job, JobRegistry, JobStatus


class TestJobRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = JobRegistry(Path(self.tmpdir))

    def test_create_job(self):
        job = self.registry.create_job(
            source_type="wav", source_name="test.wav", language="en"
        )
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual(job.source_type, "wav")
        self.assertEqual(job.source_name, "test.wav")
        self.assertIsNotNone(job.job_id)
        # Job dir created
        self.assertTrue((Path(self.tmpdir) / job.job_id).is_dir())

    def test_get_job(self):
        job = self.registry.create_job("zip", "session.zip")
        got = self.registry.get_job(job.job_id)
        self.assertEqual(got.job_id, job.job_id)

    def test_get_nonexistent(self):
        self.assertIsNone(self.registry.get_job("nope"))

    def test_update_status_processing(self):
        job = self.registry.create_job("wav", "test.wav")
        self.registry.update_status(job.job_id, JobStatus.PROCESSING)
        got = self.registry.get_job(job.job_id)
        self.assertEqual(got.status, JobStatus.PROCESSING)
        self.assertIsNone(got.completed_at)

    def test_update_status_completed(self):
        job = self.registry.create_job("wav", "test.wav")
        # Write a transcript file so it gets loaded
        job_dir = self.registry.get_job_dir(job.job_id)
        transcript = {"segments": [{"text": "hello"}]}
        with open(job_dir / "transcript.json", "w") as f:
            json.dump(transcript, f)

        self.registry.update_status(job.job_id, JobStatus.COMPLETED)
        got = self.registry.get_job(job.job_id)
        self.assertEqual(got.status, JobStatus.COMPLETED)
        self.assertIsNotNone(got.completed_at)
        self.assertEqual(got.transcript, transcript)

    def test_update_status_failed(self):
        job = self.registry.create_job("wav", "test.wav")
        self.registry.update_status(job.job_id, JobStatus.FAILED, error="boom")
        got = self.registry.get_job(job.job_id)
        self.assertEqual(got.status, JobStatus.FAILED)
        self.assertEqual(got.error, "boom")
        self.assertIsNotNone(got.completed_at)

    def test_list_jobs(self):
        self.registry.create_job("wav", "a.wav")
        self.registry.create_job("wav", "b.wav")
        self.registry.create_job("zip", "c.zip")
        jobs = self.registry.list_jobs()
        self.assertEqual(len(jobs), 3)
        # Newest first
        self.assertEqual(jobs[0].source_name, "c.zip")

    def test_persistence_reload(self):
        job = self.registry.create_job("wav", "test.wav")
        self.registry.update_status(job.job_id, JobStatus.PROCESSING)

        # Create a new registry pointing at the same dir
        registry2 = JobRegistry(Path(self.tmpdir))
        got = registry2.get_job(job.job_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.status, JobStatus.PROCESSING)

    def test_to_dict(self):
        job = self.registry.create_job("wav", "test.wav", webhook_url="http://x")
        d = job.to_dict()
        self.assertEqual(d["job_id"], job.job_id)
        self.assertEqual(d["status"], "queued")
        self.assertEqual(d["webhook_url"], "http://x")
        self.assertNotIn("transcript", d)  # Not completed


if __name__ == "__main__":
    unittest.main()
