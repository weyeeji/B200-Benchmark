import json
import tempfile
import unittest
from pathlib import Path

from metrics import NA, evaluate_rule, select_samples, server_summary


TIME = 10**9
ENDPOINT = "http://localhost:9100/metrics"


def sample(value, **labels):
    return {"value": value, "labels": labels}


def scrape(second, metrics, endpoint=ENDPOINT):
    return {"endpoint_url": endpoint, "timestamp_ns": second * TIME, "metrics": metrics}


class MetricReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def record(self, observations):
        (self.root / "profile_export_server_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
        )

    def test_model_filter_keeps_unlabelled_host_metrics(self):
        hits, queries = "vllm:prefix_cache_hits", "vllm:prefix_cache_queries"
        device = "DCGM_FI_DEV_GPU_UTIL"
        self.record([
            scrape(1, {
                hits: [sample(0, engine="0", model_name="target"), sample(0, engine="1", model_name="other")],
                queries: [sample(0, engine="0", model_name="target"), sample(0, engine="1", model_name="other")],
                device: [sample(70, gpu="0")],
            }),
            scrape(2, {
                hits: [sample(3, engine="0", model_name="target"), sample(70, engine="1", model_name="other")],
                queries: [sample(10, engine="0", model_name="target"), sample(100, engine="1", model_name="other")],
                device: [sample(90, gpu="0")],
            }),
        ])
        result = server_summary(self.root, 0, 3 * TIME, {"urls": [ENDPOINT]}, model_name="target")
        self.assertEqual(result["gpu_cache_hit_rate"], 0.3)
        self.assertEqual(result["gpu_utilization_pct"], 80)

    def test_vllm_tier_hits_use_same_stage_scheduler_queries(self):
        local, external = "vllm:prefix_cache_queries", "vllm:external_prefix_cache_queries"
        tier = "vllm:prompt_tokens_cached_by_source"
        prompts = "vllm:prompt_tokens"
        self.record([
            scrape(1, {family: [sample(0, engine="0")] for family in (local, external, prompts)} | {
                tier: [sample(0, engine="0", source=label) for label in ("device", "host", "disk")],
            }),
            scrape(2, {
                local: [sample(60, engine="0")], external: [sample(40, engine="0")],
                prompts: [sample(100, engine="0")],
                tier: [sample(30, engine="0", source="device"), sample(10, engine="0", source="host"),
                       sample(5, engine="0", source="disk")],
            }),
        ])
        result = server_summary(self.root, 0, 3 * TIME, {"urls": [ENDPOINT]})
        self.assertEqual(result["gpu_cache_hit_rate"], 0.5)
        self.assertEqual(result["cpu_cache_hit_rate"], 0.25)
        self.assertEqual(result["ssd_cache_hit_rate"], 0.125)

    def test_inconsistent_worker_counts_cannot_cancel_each_other(self):
        hits, total = "hits", "total"
        self.record([
            scrape(1, {hits: [sample(0, engine="0"), sample(0, engine="1")],
                       total: [sample(0, engine="0"), sample(0, engine="1")]}),
            scrape(2, {hits: [sample(120, engine="0"), sample(10, engine="1")],
                       total: [sample(100, engine="0"), sample(200, engine="1")]}),
        ])
        series = select_samples(self.root, 0, 3 * TIME, {hits, total})
        self.assertEqual(evaluate_rule(series, {"kind": "ratio", "metric": hits, "denominator": total}), NA)
        self.assertEqual(evaluate_rule(series, {
            "kind": "counter_delta_difference", "metric": total, "denominator": hits,
        }), NA)

    def test_counter_windows_must_overlap_but_gauges_may_be_sampled_separately(self):
        observations = [
            scrape(1, {"family": [sample(0)]}),
            scrape(2, {"family": [sample(10)]}),
            scrape(2, {"family": [sample(0)]}, "http://localhost:9200/metrics"),
            scrape(3, {"family": [sample(20)]}, "http://localhost:9200/metrics"),
        ]
        self.record(observations)
        series = select_samples(self.root, 0, 4 * TIME, {"family"})
        self.assertEqual(evaluate_rule(series, {"kind": "counter_rate", "metric": "family"}), NA)
        self.assertEqual(evaluate_rule(series, {"kind": "gauge_sum", "metric": "family"}), 15)


if __name__ == "__main__":
    unittest.main()
