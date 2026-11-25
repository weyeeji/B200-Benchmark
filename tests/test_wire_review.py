import json
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import runner


class WireRequestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input.jsonl"
        self.raw = self.root / "profile_export_raw.jsonl"
        self.source.write_text(
            '{"session_id":"phase-0","text":"第一个 prompt","output_length":8}\n'
            '{"session_id":"phase-1","text":"第二个 prompt","output_length":16}\n',
            encoding="utf-8",
        )
        self.cfg = {
            "model": "m", "use_legacy_max_tokens": False, "per_chunk_usage": False,
            "chat_template_kwargs": {"thinking": True},
        }

    def frame(self, index=0, **payload_updates):
        text, length = [("第一个 prompt", 8), ("第二个 prompt", 16)][index]
        options = {"include_usage": True}
        if self.cfg["per_chunk_usage"]:
            options["continuous_usage_stats"] = True
        token_key = "max_tokens" if self.cfg["use_legacy_max_tokens"] else "max_completion_tokens"
        return {
            "metadata": {"session_num": index, "conversation_id": f"phase-{index}"},
            "payload": {
                "messages": [{"role": "user", "content": text}],
                "model": "m", "stream": True, "ignore_eos": True, "temperature": 0,
                token_key: length, "chat_template_kwargs": {"thinking": True}, "stream_options": options,
                **payload_updates,
            },
        }

    def verify(self, records):
        self.raw.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return runner.verify_payload(self.root, self.source, self.cfg)

    def test_raw_records_can_arrive_out_of_order(self):
        result = self.verify([self.frame(1), self.frame(0)])
        self.assertEqual(result, {"status": "verified", "recorded": 2, "verified": 2, "violations": []})

    def test_expected_token_field_and_continuous_usage(self):
        self.cfg.update(use_legacy_max_tokens=True, per_chunk_usage=True)
        self.assertEqual(self.verify([self.frame()])["status"], "verified")
        result = self.verify([self.frame(max_completion_tokens=8, stream_options={"include_usage": True})])
        self.assertEqual(result["violations"][0]["fields"], ["max_completion_tokens", "stream_options"])

    def test_records_wrong_prompt_settings_and_duplicate_index(self):
        result = self.verify([
            self.frame(
                messages=[{"role": "user", "content": "wrong"}],
                ignore_eos=False, max_completion_tokens=9, temperature=1, stop=["stop"],
            ),
            self.frame(0),
        ])
        self.assertEqual(result["status"], "payload_mismatch")
        self.assertEqual(result["recorded"], 2)
        self.assertEqual(result["verified"], 0)
        self.assertEqual(
            result["violations"][0]["fields"],
            ["messages", "ignore_eos", "max_completion_tokens", "stop", "temperature"],
        )
        self.assertEqual(result["violations"][1]["fields"], ["session_num"])

    def test_archive_hash_is_verified_after_planning(self):
        expected = runner.sha256(self.source)
        self.assertEqual(self.verify([self.frame()])["status"], "verified")
        original = self.source.read_text(encoding="utf-8")
        self.source.write_text(original + "\n", encoding="utf-8")
        result = runner.verify_payload(self.root, self.source, self.cfg, expected_sha256=expected)
        self.assertEqual(result["status"], "dataset_changed")
        self.assertEqual(result["dataset_sha256"], runner.sha256(self.source))
        self.assertEqual(result["expected_dataset_sha256"], expected)

    def test_missing_body_and_missing_export_are_not_verified(self):
        self.assertEqual(runner.verify_payload(self.root, self.source, self.cfg)["status"], "no_raw_payloads")
        self.assertEqual(self.verify([{"metadata": self.frame()["metadata"]}])["violations"][0]["fields"], ["payload"])


class CapacityAiperfCancellationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.event = threading.Event()
        self.cfg = {"api_key_env": None}

    def test_cancellation_before_worker_launch(self):
        self.event.set()
        with patch.object(runner, "aiperf_command", return_value=["test"]), patch.object(runner.subprocess, "Popen") as spawn:
            artifact = self.root / "phase"
            self.assertEqual(runner.run_aiperf(self.cfg, Path("input"), artifact, 1, 1, cancel_event=self.event), 130)
        spawn.assert_not_called()
        self.assertTrue(json.loads((artifact / "invocation.json").read_text())["interrupted"])

    def test_cancellation_stops_process_group_during_wait(self):
        child = Mock(pid=10)

        def wait(timeout=None):
            if timeout == 0.25:
                self.event.set()
                raise subprocess.TimeoutExpired("test", timeout)
            if timeout == 20:
                raise subprocess.TimeoutExpired("test", timeout)
            return -signal.SIGKILL

        child.wait.side_effect = wait
        with patch.object(runner, "aiperf_command", return_value=["test"]), \
             patch.object(runner.subprocess, "Popen", return_value=child), \
             patch.object(runner.os, "killpg") as kill:
            artifact = self.root / "phase"
            self.assertEqual(runner.run_aiperf(self.cfg, Path("input"), artifact, 1, 1, cancel_event=self.event), 130)
        self.assertEqual([item.args for item in kill.call_args_list], [(10, signal.SIGINT), (10, signal.SIGKILL)])
        self.assertTrue(json.loads((artifact / "invocation.json").read_text())["interrupted"])
