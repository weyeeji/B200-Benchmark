import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from balanced_cache import run_balanced, scoped_deltas
from measure_cache import PREFIX, check_cache_delta


class CapacityReviewTest(unittest.TestCase):
    def test_group_boundary_uses_the_weakest_engine_for_equal_placement(self):
        args = SimpleNamespace(engine_affinity=[("0", "a"), ("1", "b")], max_groups=3,
                               start_groups=2, calibration_trials=2, calibration_osl=8,
                               cache_affinity_header="X-Session-ID", baseline_min_fraction=.90,
                               hit_fraction=.95)
        cfg = {"api_url": "http://api", "gpu_count": 2, "model": "model"}

        def trial(_, __, url, root, size, number, rng, pool, cancelled):
            return {"status": "observed", "gpu_hit_tokens": {
                "0": PREFIX if size < 3 else 32_768, "1": PREFIX,
            }}

        with TemporaryDirectory() as temp, patch("balanced_cache.balanced_trial", side_effect=trial) as probe:
            directory = Path(temp)
            self.assertEqual(run_balanced(args, cfg, directory, "http://metrics", 42), 0)
            report = json.loads((directory / "capacity.json").read_text())
            self.assertEqual(report["measurement_scope"], "balanced_working_set")
            self.assertEqual(report["candidate_groups_per_engine"], {"pass": 2, "fail": 3})
            self.assertEqual(report["cache_capacity_tokens"], 4 * PREFIX)
            self.assertEqual(report["first_failed_working_set_tokens"], 6 * PREFIX)
            self.assertEqual(probe.call_count, 6)

    def test_small_baseline_cannot_produce_full_prefix_capacity(self):
        args = SimpleNamespace(engine_affinity=[("0", "a"), ("1", "b")], max_groups=3,
                               start_groups=2, calibration_trials=1, calibration_osl=8,
                               cache_affinity_header="X-Session-ID", baseline_min_fraction=.90,
                               hit_fraction=.95)
        cfg = {"api_url": "http://api", "gpu_count": 2, "model": "model"}
        result = {"status": "observed", "gpu_hit_tokens": {"0": 128, "1": PREFIX}}
        with TemporaryDirectory() as temp, patch("balanced_cache.balanced_trial", return_value=result):
            directory = Path(temp)
            self.assertEqual(run_balanced(args, cfg, directory, "http://metrics", 42), 2)
            report = json.loads((directory / "capacity.json").read_text())
            self.assertEqual(report["status"], "no_stable_gpu_hit_baseline")
            self.assertIsNone(report["cache_capacity_tokens"])

    def test_warm_fill_and_wrong_query_tokens_are_rejected(self):
        data = {"isl": 65_536, "validation": {"template_common_header_tokens": 28,
                                             "cross_group_prefix_max_extra_tokens": 3}}
        check_cache_delta({"queries": 65_536, "device": 31}, data, 1, "fill")
        with self.assertRaisesRegex(ValueError, "unexpected cache hits"):
            check_cache_delta({"queries": 65_536, "device": 16, "host": 16}, data, 1, "fill")
        with self.assertRaisesRegex(ValueError, "query token count"):
            check_cache_delta({"queries": 65_535, "device": 0}, data, 1, "probe")

    def test_same_total_on_wrong_engine_is_rejected(self):
        def engine(completed):
            return {"running": 0, "waiting": 0, "completed": completed,
                    "finish_reasons": {"length": completed}, "device": 0,
                    "queries": 65_536 * completed}

        before = {"0": engine(0), "1": engine(0)}
        after = {"0": engine(0), "1": engine(1)}
        with self.assertRaisesRegex(ValueError, "did not reach the configured engines"):
            scoped_deltas(before, after, {"0": 1, "1": 0}, {}, "prime")


if __name__ == "__main__":
    unittest.main()
