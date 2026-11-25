import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark
import measure_cache
import report
import runner


class RunnerTest(unittest.TestCase):
    def test_child_auth_does_not_redirect_the_benchmark(self):
        with patch.dict(os.environ, {
            "MY_API_KEY": "sample-key", "OPENAI_API_KEY": "ambient-key",
            "AIPERF_INJECTED_ENDPOINT_URLS": '["http://ambient.example:8000"]',
            "AIPERF_INJECTED_DATASET_URLS": '["http://ambient.example/input.jsonl"]',
            "AIPERF_INJECTED_HEADERS": '{"X-Session-ID":"ambient"}',
            "AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID": "true",
        }, clear=True):
            env = runner.aiperf_env("MY_API_KEY", {"x-session-affinity": "pinned"})
        self.assertEqual(env["OPENAI_API_KEY"], "sample-key")
        self.assertNotIn("AIPERF_INJECTED_API_KEY", env)
        self.assertEqual(env["AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID"], "false")
        for key in ("MY_API_KEY", "AIPERF_INJECTED_ENDPOINT_URLS",
                    "AIPERF_INJECTED_HEADERS", "AIPERF_INJECTED_DATASET_URLS"):
            self.assertNotIn(key, env)

    def test_dispatch_validates_credits_and_allows_wire_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text('{"session_id":"a"}\n{"session_id":"b"}\n')
            records = root / "profile_export.jsonl"
            for first, second, expected in (
                (2, 3, "ordered"),
                (3, 2, "dispatch_mismatch"),
                (None, 3, "unverified_dispatch"),
            ):
                rows = [
                    {"metadata": {"session_num": 1, "conversation_id": "b",
                                  "credit_issued_ns": second, "request_start_ns": 20}},
                    {"metadata": {"session_num": 0, "conversation_id": "a",
                                  "credit_issued_ns": first, "request_start_ns": 25}},
                ]
                records.write_text("".join(json.dumps(row) + "\n" for row in rows))
                self.assertEqual(runner.verify_dispatch(root, dataset)["status"], expected)


class BenchmarkTest(unittest.TestCase):
    def test_warmup_mismatched_usage_blocks_formal_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            data = {
                "manifest": manifest, "warmup": root / "warmup.jsonl", "formal": root / "formal.jsonl",
                "profile": "P01", "isl": 4096, "osl": 1024, "warmup_requests": 16, "formal_requests": 6,
                "prefix_groups": 0, "prefix_tokens": 0, "formal_full_cycles": None,
                "theoretical_shared_prefix_ratio": 0,
                "files": {"formal.jsonl": "formal", "warmup.jsonl": "warmup"},
            }
            cfg = benchmark.DEFAULTS | {"api_url": "http://localhost:8000"}
            counts = {"completed_requests": 16, "isl_matching_requests": 15, "osl_matching_requests": 16}
            with patch.object(benchmark, "run_aiperf", side_effect=lambda _, file, artifact, *args: artifact.mkdir() or 0) as run, \
                 patch.object(benchmark, "verify_dispatch", return_value={"status": "ordered", "recorded": 16}), \
                 patch.object(benchmark.metrics, "record_counts", return_value=counts), \
                 patch.object(benchmark.metrics, "summarize", return_value={
                     "completed_requests": 0, "steady_seconds": "N/A", "steady_completed_requests": "N/A",
                 }):
                row = benchmark.run_point(cfg, data, 2, 1, root / "point", "run", None)
            run.assert_called_once()
            self.assertEqual(row["status"], "incomplete")
            self.assertIn("Warmup server Usage", row["issues"])

    def test_monitoring_does_not_override_request_counters(self):
        with self.assertRaises(ValueError):
            benchmark.validate_prometheus({"urls": [], "rules": {
                "rps": {"kind": "gauge_mean", "metric": "foo"},
            }})
        with self.assertRaises(ValueError):
            benchmark.validate_prometheus({"urls": [], "rules": {
                "gpu_utilization_pct": {"kind": "gauge_mean", "metric": "foo", "label_regex": {"gpu": "["}},
            }})

    def test_report_escapes_user_values_and_flags_incompatible_datasets(self):
        row = {"model": "<img src=x>", "deployment": "<script>alert(1)</script>",
               "profile": "P01", "concurrency": 32, "repeat": 1, "status": "complete"}
        rows = [row | {"run_source": "a", "dataset_formal_sha256": "a"},
                row | {"run_source": "b", "dataset_formal_sha256": "b"}]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.html"
            report.write_report(target, rows, combined=True)
            result = target.read_text()
        self.assertIn("数据集或运行协议存在差异", result)
        self.assertIn("&lt;img src=x&gt;", result)
        self.assertNotIn("<script>", result)

    def test_capacity_labels_multi_engine_boundary_and_checks_pin_across_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = benchmark.DEFAULTS | {"api_url": "http://localhost:8000"}
            args = SimpleNamespace(calibration_osl=8, max_groups=2, start_groups=2,
                                   calibration_trials=1, hit_fraction=.95,
                                   cache_affinity_header="X-Session-ID", cache_affinity_key="one", pin_fill=False)

            def sample(options, _cfg, _url, path, groups, number, seed):
                path.mkdir()
                return {"status": "observed", "monitored_engines": ["0", "1"],
                        "prime_engine": str(groups - 1), "gpu_hit_tokens": 58880 if groups == 1 else 0}

            with patch.object(measure_cache, "measure_trial", side_effect=sample):
                url = cfg["api_url"] + "/metrics"
                routed = root / "routed"
                routed.mkdir()
                self.assertEqual(measure_cache.run_calibration(args, cfg, routed, url, 123), 2)
                routed_result = json.loads((routed / "capacity.json").read_text())
                self.assertEqual(routed_result["status"], "routed_boundary_only")
                self.assertIsNone(routed_result["cache_capacity_tokens"])

                args.pin_fill = True
                pinned = root / "pinned"
                pinned.mkdir()
                self.assertEqual(measure_cache.run_calibration(args, cfg, pinned, url, 123), 2)
                pinned_result = json.loads((pinned / "capacity.json").read_text())
                self.assertEqual(pinned_result["status"], "inconclusive")
                self.assertIn("pinned engine changed", pinned_result["last_error"])


if __name__ == "__main__":
    unittest.main()
