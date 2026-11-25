"""Offline regressions for the Benchmark run and reporting protocol."""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark
from observability import PhaseObserver
from report import summarize


PROJECT = benchmark.ROOT


def arguments(experiments=("P01", "P02", "P03"), concurrency=(16,), repeats=1):
    return SimpleNamespace(
        experiments=experiments, concurrency=concurrency, repeats=repeats, model="test-model",
        url="http://offline.invalid:8000", metrics_url=None, seed=42, dataset_nonce="test",
        max_context_length=None, request_timeout=600,
    )


def record():
    return {
        "metadata": {
            "benchmark_phase": "profiling", "request_start_ns": 1_000_000_000,
            "request_end_ns": 2_000_000_000,
        },
        "metrics": {
            "time_to_first_token": {"unit": "ms", "value": 100},
            "full_response_inter_token_latency": {"unit": "ms", "value": 2},
            "usage_prompt_tokens": {"unit": "tokens", "value": 16},
            "usage_completion_tokens": {"unit": "tokens", "value": 10},
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_repeat_paths_and_nonces_keep_p03_single(self):
        plan = benchmark.create_plan(arguments(repeats=3))
        pairs = {(point["experiment"], point["repeat"]): point for point in plan["points"]}
        self.assertEqual(len(pairs), 7)
        self.assertEqual(pairs["P01", 1]["dataset_nonce"], "test")
        self.assertNotEqual(pairs["P01", 1]["dataset_nonce"], pairs["P01", 2]["dataset_nonce"])
        self.assertEqual(pairs["P02", 3]["point_dir"], "P02/C16/R03")
        self.assertEqual(pairs["P03", 1]["point_dir"], "P03/C16")
        self.assertEqual(pairs["P03", 1]["repeat_count"], 1)
        flags = pairs["P03", 1]["phases"][0]["argv"]
        self.assertEqual(flags[flags.index("--benchmark-grace-period") + 1], "90")

    def test_false_native_submission_preserves_measured_sla(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "profile_export.jsonl").write_text(json.dumps(record()) + "\n")
            (artifacts / "profile_export.json").write_text(json.dumps({
                "metadata": {"submission_valid": False}, "benchmark_duration": {"unit": "s", "avg": 1},
            }))
            row = summarize(artifacts, "P03", 16)
            self.assertEqual(row["status"], "invalid")
            self.assertEqual(row["validity"], "invalid")
            self.assertEqual(row["sla_status"], "pass")
            self.assertEqual(row["completed_requests"], 1)

    def test_failed_child_retains_partial_records_and_phase_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "P03").mkdir()
            shutil.copyfile(PROJECT / "P03/workload.json", root / "P03/workload.json")
            with patch.object(benchmark, "ROOT", root):
                plan = benchmark.create_plan(arguments(experiments=("P03",)))
                point = plan["points"][0]
                phase = point["phases"][0]
                child = (
                    "import pathlib,sys\n"
                    "path = pathlib.Path(sys.argv[1]); path.mkdir(parents=True)\n"
                    "(path / 'profile_export.jsonl').write_text(sys.argv[2] + '\\n')\n"
                    "print('Phase profiling (profiling) started | target: 900s')\n"
                    "print('Phase profiling (profiling) sending complete | sent=1, completed=1, in_flight=0')\n"
                    "sys.exit(17)\n"
                )
                phase["argv"] = [sys.executable, "-c", child, phase["artifacts"], json.dumps(record())]
                logs = root / "logs" / plan["run_id"]
                results = root / "results" / plan["run_id"]
                row = benchmark.run_point(point, plan, logs, results)
            self.assertEqual((row["status"], row["validity"], row["exit_code"]),
                             ("error", "incomplete", 17))
            self.assertEqual(row["completed_requests"], 1)
            self.assertEqual(row["sla_status"], "pass")
            self.assertEqual(row["in_flight_at_cutoff"], 0)
            self.assertTrue((results / row["phase_summary"]).is_file())
            status = json.loads((logs / "P03/C16/status.json").read_text())
            self.assertEqual(status["state"], "error")

    def test_failed_repeat_completes_current_point_and_skips_higher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "P01").mkdir()
            shutil.copyfile(PROJECT / "P01/workload.json", root / "P01/workload.json")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}")
            called = []

            def fake_point(point, plan, logs, results):
                called.append((point["concurrency"], point["repeat"]))
                result = "fail" if point["repeat"] == 2 else "pass"
                return {
                    "experiment": "P01", "concurrency": point["concurrency"], "repeat": point["repeat"],
                    "repeat_count": 3, "status": result, "sla_status": result,
                    "validity": "valid", "detail": "fixture",
                }

            with patch.multiple(benchmark, ROOT=root, TOKENIZER=tokenizer), \
                    patch.object(benchmark, "run_point", side_effect=fake_point), \
                    patch("run_metadata.capture_run_metadata", return_value={"code_commit": "fixture"}):
                plan = benchmark.create_plan(arguments(experiments=("P01",), concurrency=(16, 32), repeats=3))
                self.assertEqual(benchmark.run_plan(plan), 0)
            self.assertEqual(called, [(16, 1), (16, 2), (16, 3)])
            result = root / "results" / plan["run_id"]
            with (result / "summary.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertEqual(json.loads((result / "manifest.json").read_text())["code_commit"], "fixture")

    def test_agentx_whole_run_metric_never_becomes_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            point = Path(directory)
            labels = (("model_name", "test-model"),)

            def snapshot(requests, tokens, cached):
                return {
                    ("vllm:request_success_total", labels): requests,
                    ("vllm:prompt_tokens_total", labels): tokens,
                    ("vllm:prompt_tokens_cached_total", labels): cached,
                }

            with patch("observability.capture", side_effect=(snapshot(0, 0, 0), snapshot(8, 200, 180))):
                with PhaseObserver(point, point, logical_phase="formal", experiment="P03",
                                   metrics_url="http://offline.invalid/metrics") as observer:
                    observer.returncode = 0
            row = {"completed_requests": 6, "duration_seconds": 10, "input_tpm": 720}
            observer.enrich(row, "test-model")
            self.assertEqual(row["whole_run_kv_hit_percent"], 90)
            self.assertIsNone(row.get("kv_hit_percent"))
            self.assertIsNone(row.get("window_kv_hit_percent"))
            self.assertEqual(observer.summary["whole_run_requests_seen"], 8)


if __name__ == "__main__":
    unittest.main()
