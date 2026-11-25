import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import benchmark
import report
from report_charts import profile_charts
from report_trends import output_trend


def measurement(concurrency, **overrides):
    return {
        "profile": "P01", "concurrency": concurrency, "repeat": 1, "status": "complete",
        "kind": "run", "deployment": "配置 A", "output_tps": 2380, "valid_for_comparison": True,
        **overrides,
    }


class ChartTests(unittest.TestCase):
    def test_profiles_and_deployment_labels_are_escaped(self):
        html = report.chart_section([
            measurement(32, profile="P<01>"),
            measurement(64, profile="P<01>", deployment="<script>"),
        ], combined=True)
        self.assertIn("P&lt;01&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("图表", html)

    def test_trend_connects_only_adjacent_levels_in_the_same_repeat(self):
        html = output_trend([
            measurement(32), measurement(64, output_tps=0),
            measurement(128, output_tps="N/A"),
            measurement(256),
            measurement(128, repeat=2, valid_for_comparison=False),
        ], combined=False)
        self.assertEqual(html.count('class="trend-segment"'), 1)
        self.assertEqual(html.count("<circle "), 4)
        self.assertIn("并发 64 · 0.00 tok/s", html)
        self.assertIn("空心点", html)

    def test_incomplete_level_breaks_the_line(self):
        html = output_trend([
            measurement(32), measurement(64, status="incomplete"), measurement(128),
        ], False)
        self.assertEqual(html.count("<circle "), 2)
        self.assertNotIn('class="trend-segment"', html)
        self.assertIn(">64</text>", html)

    def test_metric_panels_keep_measured_zero_separate_from_missing_values(self):
        html = profile_charts([
            measurement(32, rps=0, formal_ttft_p50_ms=1200, formal_ttft_p95_ms=2400,
                        formal_ttft_p99_ms=5000, formal_ttft_samples=96),
            measurement(64, rps="N/A"),
        ])
        self.assertIn("--chart-width:0.00000000%", html)
        self.assertIn("N/A", html)
        self.assertIn("样本 96", html)
        self.assertIn("P99 <b>5</b>", html)
        self.assertNotIn("GPU 利用率", html)

    def test_bar_axis_is_shared_across_scenarios(self):
        here = measurement(32, output_tps=1000)
        other = measurement(32, profile="P02", output_tps=5000)
        html = profile_charts([here], scale_rows=[here, other])
        self.assertIn("--chart-width:20.00000000%", html)
        self.assertIn("<span>5,000</span>", html)

    def test_same_directory_names_are_disambiguated_in_comparison(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / label / "same-name" for label in ("a", "b")]
            for path in paths:
                path.mkdir(parents=True)
                (path / "summary.json").write_text(json.dumps([measurement(32, report_schema=report.SCHEMA)]))
            with patch.object(report, "write_report") as capture:
                self.assertEqual(benchmark.compare_reports(paths, {"output_dir": root / "out"}), 0)
            self.assertEqual([row["run_source"] for row in capture.call_args.args[1]],
                             ["same-name", "same-name (2)"])

    def test_incomplete_or_single_level_does_not_create_a_trend(self):
        self.assertEqual(output_trend([
            measurement(32), measurement(64, status="incomplete"),
            measurement(256, output_tps="N/A"),
        ], False), "")


if __name__ == "__main__":
    unittest.main()
