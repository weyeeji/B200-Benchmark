import json
import tempfile
import unittest
from pathlib import Path

from metrics import NA, evaluate_rule, select_samples, summarize


CLOCK = 1_800_000_000_000_000_000
SECOND = 10**9
ENDPOINT = "http://localhost:9100/metrics"
COUNTER = "node_network_receive_bytes"


def timestamp(seconds):
    return CLOCK + round(seconds * SECOND)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def snapshot(sent, received, metrics, endpoint=ENDPOINT):
    return {
        "endpoint_url": endpoint,
        "request_sent_ns": timestamp(sent),
        "timestamp_ns": timestamp(received),
        "metrics": metrics,
    }


def counter_sample(value, engine="a"):
    return {"value": value, "labels": {"engine": engine}}


def profile(index, start, end, ttft=50, error=None, cancellation=None, cache_usage=None):
    return {
        "metadata": {
            "session_num": index,
            "benchmark_phase": "profiling",
            "request_start_ns": timestamp(start) if start is not None else None,
            "request_end_ns": timestamp(end),
            "was_cancelled": cancellation is not None,
            "cancellation_time_ns": timestamp(cancellation) if cancellation is not None else None,
        },
        "metrics": {
            "time_to_first_token": {"value": ttft, "unit": "ms"},
            "full_decode_duration": {"value": max(0, (end - (start or 0)) * 1e3 - ttft), "unit": "ms"},
            "inter_token_latency": {"value": 4, "unit": "ms"},
            "full_response_inter_token_latency": {"value": 5, "unit": "ms"},
            "usage_prompt_tokens": {"value": 100, "unit": "tokens"},
            "usage_completion_tokens": {"value": 20, "unit": "tokens"},
            **{
                "usage_prompt_cache_" + field + "_tokens": {"value": value, "unit": "tokens"}
                for field, value in (cache_usage or {}).items()
            },
        } if error is None else {},
        "error": error,
    }


class MetricTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def timeseries(self, snapshots, families, start=0, end=10):
        write_jsonl(self.path / "profile_export_server_metrics.jsonl", snapshots)
        return select_samples(self.path, timestamp(start), timestamp(end), set(families))

    def test_counter_uses_request_send_order_not_response_order(self):
        counter = 2_400_000_000_000
        data = self.timeseries([
            snapshot(2, 2.2, {COUNTER: [counter_sample(counter + 130)]}),
            snapshot(1, 2.3, {COUNTER: [counter_sample(counter)]}),
            snapshot(3, 3.1, {COUNTER: [counter_sample(counter + 250)]}),
            snapshot(10.1, 10.2, {COUNTER: [counter_sample(999_999_999_999_999)]}),
        ], [COUNTER])
        rule = {"kind": "counter_rate", "metric": COUNTER, "endpoints": ["http://localhost:9100"]}
        self.assertEqual(evaluate_rule(data, rule), 125)

    def test_counter_decrease_or_partial_worker_invalidates_aggregate(self):
        rule = {"kind": "counter_delta", "metric": COUNTER}
        for case, samples in (
            ("one-byte drop", [[9, 99], [8, 100], [12, 110]]),
            ("reset", [[9, 99], [1, 100], [12, 110]]),
            ("missing worker", [[9, 99], [10, None], [12, 110]]),
            ("missing worker at every steady scrape", [[9, None], [10, None], [12, None]]),
        ):
            with self.subTest(case=case):
                snapshots = [snapshot(-1, -0.9, {COUNTER: [counter_sample(95, "b")]})]
                for step, (a, b) in enumerate(samples, start=1):
                    observations = [counter_sample(a)]
                    if b is not None:
                        observations.append(counter_sample(b, "b"))
                    snapshots.append(snapshot(step, step + 0.1, {COUNTER: observations}))
                self.assertEqual(evaluate_rule(self.timeseries(snapshots, [COUNTER]), rule), NA)

    def test_ratio_requires_same_worker_samples_and_compatible_endpoint_windows(self):
        hits, total = "cache_hits", "cache_queries"
        rate = {"kind": "ratio", "metric": hits, "denominator": total}
        first = {hits: [counter_sample(4)], total: [counter_sample(10)]}
        second = {hits: [counter_sample(8)], total: [counter_sample(18)]}
        data = self.timeseries([
            snapshot(1, 1.1, first),
            snapshot(2, 2.1, second),
        ], [hits, total])
        self.assertEqual(evaluate_rule(data, rate), 0.5)
        self.assertEqual(evaluate_rule(data, {**rate, "endpoints": [ENDPOINT, "http://missing"]}), NA)

        late = [
            snapshot(8, 8.1, first, "http://localhost:9200/metrics"),
            snapshot(9, 9.1, second, "http://localhost:9200/metrics"),
        ]
        data = self.timeseries([snapshot(1, 1.1, first), snapshot(2, 2.1, second), *late], [hits, total])
        self.assertEqual(evaluate_rule(data, rate), NA)

        data = self.timeseries([
            snapshot(1, 1.1, first),
            snapshot(2, 2.1, {hits: second[hits]}),
            snapshot(3, 3.1, {total: second[total]}),
        ], [hits, total])
        self.assertEqual(evaluate_rule(data, rate), NA)

    def test_cohort_metrics_and_cache_sources_remain_separate(self):
        results = [
            profile(0, 0, 1.0),
            profile(1, 0.1, 2.5, ttft=100),
            profile(2, 1.01, 1.01, error={"type": "TimeoutError"}),
            profile(3, 1.02, 3.0, ttft=350),
            profile(4, 2.51, 5.0, ttft=2000),
            profile(5, 3.01, 4.0, error={"type": "CancelledError"}, cancellation=5.5),
        ]
        write_jsonl(self.path / "profile_export.jsonl", results)
        all_tokens, cached_tokens = "vllm:prompt_tokens", "vllm:prompt_tokens_cached"
        write_jsonl(self.path / "profile_export_server_metrics.jsonl", [
            snapshot(0.2, 0.3, {all_tokens: [counter_sample(200)], cached_tokens: [counter_sample(20)]}),
            snapshot(2.7, 2.8, {all_tokens: [counter_sample(600)], cached_tokens: [counter_sample(20)]}),
        ])
        summary = summarize(
            self.path, concurrency=2, gpu_count=4, isl=100, osl=20, sla={"ttft_ms": 150},
        )
        self.assertEqual(summary["steady_attempts"], 4)
        self.assertEqual(summary["steady_completed_requests"], 3)
        self.assertEqual(summary["drain_attempts"], 2)
        self.assertEqual(summary["drain_completed_requests"], 1)
        self.assertEqual(summary["drain_failed_requests"], 1)
        self.assertEqual(summary["timeout_rate"], 1 / 4)
        self.assertEqual(summary["error_rate"], 1 / 4)
        self.assertEqual(summary["whole_run_error_rate"], 2 / 6)
        self.assertEqual(summary["sla_pass_rate"], 2 / 4)
        self.assertAlmostEqual(summary["rps"], 3 / 2.91)
        self.assertAlmostEqual(summary["goodput_rps"], 2 / 2.91)
        self.assertAlmostEqual(summary["actual_concurrency_mean"], (2 * 2.91 - 0.04) / 2.91)
        self.assertAlmostEqual(summary["steady_concurrency_at_target_fraction"], (2.91 - 0.04) / 2.91)
        self.assertEqual(summary["itl_statistic"], "percentiles_of_per_request_mean_token_intervals")
        self.assertEqual(summary["steady_itl_samples"], 3)
        self.assertEqual(summary["formal_itl_samples"], 4)
        self.assertEqual(summary["ttft_max_ms"], 350)
        self.assertEqual(summary["formal_ttft_max_ms"], 2000)
        self.assertEqual(summary["cached_input_tps"], NA)
        self.assertEqual(summary["uncached_input_tps"], NA)
        self.assertEqual(summary["actual_cache_hit_rate"], NA)
        self.assertEqual(summary["server_cached_input_tps"], 0)
        self.assertEqual(summary["server_uncached_input_tps"], 400 / 2.5)
        self.assertEqual(summary["server_prefix_cache_hit_rate"], 0)

    def test_consistent_usage_cache_counts_reconcile_and_invalid_counts_stay_unknown(self):
        for case, usage, expected in (
            ("cache-read only", {"read": 25}, 0.25),
            ("cache-miss only", {"miss": 75}, 0.25),
            ("contradictory", {"read": 25, "miss": 50}, NA),
            ("more than input", {"read": 200}, NA),
        ):
            with self.subTest(case=case):
                write_jsonl(self.path / "profile_export.jsonl", [
                    profile(0, 0, 1.0, cache_usage=usage),
                    profile(1, 1.1, 2.0, cache_usage=usage),
                ])
                summary = summarize(self.path, concurrency=1, gpu_count=4, isl=100, osl=20)
                self.assertEqual(summary["actual_cache_hit_rate"], expected)
                if expected != NA:
                    self.assertEqual(summary["cached_input_tokens"] + summary["uncached_input_tokens"],
                                     summary["input_tokens"])
                else:
                    self.assertEqual(summary["cached_input_tps"], NA)
                    self.assertEqual(summary["uncached_input_tps"], NA)

    def test_missing_timestamp_invalidates_steady_but_preserves_formal_latency(self):
        write_jsonl(self.path / "profile_export.jsonl", [
            profile(0, 0, 1.0), profile(1, 1.1, 2.0), profile(2, None, 3.0, ttft=600),
        ])
        summary = summarize(self.path, concurrency=1, gpu_count=4, isl=100, osl=20)
        self.assertEqual(summary["untimed_requests"], 1)
        self.assertEqual(summary["rps"], NA)
        self.assertEqual(summary["whole_run_request_success_rate"], 1)
        self.assertEqual(summary["formal_ttft_max_ms"], 600)
        self.assertEqual(summary["formal_ttft_samples"], 3)


if __name__ == "__main__":
    unittest.main()
