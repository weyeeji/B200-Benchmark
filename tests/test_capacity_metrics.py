import unittest

import capacity_metrics as cm


MODEL = "deepseek-test"


def exposition(engine, *, completed=0, hits=0, running=0, waiting=0, queries=0):
    labels = f'model_name="{MODEL}",engine="{engine}"'
    lines = [
        f'{cm.CACHE}{{{labels},source="device"}} {hits}',
        f'{cm.COMPLETIONS}{{{labels},finished_reason="length"}} {completed}',
    ]
    for name, value in [
        ("vllm:num_requests_running", running),
        ("vllm:num_requests_waiting", waiting),
        (cm.QUERIES, queries),
    ]:
        if value is not None:
            lines.append(f"{name}{{{labels}}} {value}")
    return "\n".join(lines) + "\n"


class CapacityMetricTests(unittest.TestCase):
    def test_idle_snapshots_support_per_engine_delta(self):
        before = cm.parse_snapshot(exposition(0) + exposition(1), MODEL)
        after = cm.parse_snapshot(exposition(0) + exposition(1, completed=1, hits=20, queries=30), MODEL)

        result = cm.phase_delta(before, after, expected=1)

        self.assertEqual(result["engine"], "1")
        self.assertEqual(result["device_tokens"], 20)
        self.assertEqual(result["queries"], 30)

    def test_rejects_partial_per_engine_metric_coverage(self):
        for metric in ("running", "waiting", "queries"):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, rf"Engine 1 lacks metric evidence: {metric}"):
                    cm.parse_snapshot(exposition(0) + exposition(1, **{metric: None}), MODEL)

    def test_missing_query_family_is_supported_when_absent_on_all_engines(self):
        before = cm.parse_snapshot(exposition(0, queries=None) + exposition(1, queries=None), MODEL)
        after = cm.parse_snapshot(exposition(0, queries=None, completed=1) + exposition(1, queries=None), MODEL)

        self.assertIsNone(cm.phase_delta(before, after, expected=1)["queries"])

    def test_rejects_invalid_activity_values(self):
        for value in ("NaN", "+Inf", "-1", "0.5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Invalid vllm:num_requests_running value"):
                    cm.parse_snapshot(exposition(0, running=value), MODEL)

    def test_invalid_archived_snapshot_cannot_assert_idle(self):
        for missing in ("running", "waiting"):
            with self.subTest(missing=missing):
                before = cm.parse_snapshot(exposition(0), MODEL)
                after = cm.parse_snapshot(exposition(0, completed=1), MODEL)
                del before["0"][missing]

                with self.assertRaisesRegex(ValueError, rf"Engine 0 lacks metric evidence: {missing}"):
                    cm.phase_delta(before, after, expected=1)

    def test_busy_engine_cannot_assert_idle(self):
        before = cm.parse_snapshot(exposition(0, waiting=1), MODEL)
        after = cm.parse_snapshot(exposition(0, completed=1), MODEL)

        with self.assertRaisesRegex(ValueError, "requests in flight"):
            cm.phase_delta(before, after, expected=1)

    def test_invalid_archived_counters_fail_closed(self):
        cases = [
            ({"device": float("nan")}, "Invalid device value"),
            ({"completed": 2}, "inconsistent request completion counters"),
            ({"finish_reasons": {"length": -1}}, "Invalid finish reason counter"),
        ]
        for corruption, message in cases:
            with self.subTest(corruption=corruption):
                before = cm.parse_snapshot(exposition(0), MODEL)
                after = cm.parse_snapshot(exposition(0, completed=1), MODEL)
                after["0"].update(corruption)

                with self.assertRaisesRegex(ValueError, message):
                    cm.phase_delta(before, after, expected=1)
