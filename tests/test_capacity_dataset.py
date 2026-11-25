"""Regression tests for exact, independent capacity trial inputs."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import dataset


class CapacityDatasetTest(unittest.TestCase):
    def test_only_anchor_fits_a_reusable_prefix(self):
        pool = tuple(" token" + str(i) for i in range(1024))
        token_ids = {piece.strip(): i + 10 for i, piece in enumerate(pool)}

        def encode(prompt):
            return [1, 2, *(token_ids[piece] for piece in prompt.split()), 3]

        with (TemporaryDirectory() as directory,
              patch.dict(dataset.PROFILES, {"P03": (64, 8, 48)}),
              patch.object(dataset, "_tokenizer_hashes", return_value={"tokenizer.json": "test"}),
              patch.object(dataset, "_tokenizer", return_value=(encode, pool)),
              patch.object(dataset, "_fit", wraps=dataset._fit) as fit):
            groups = 11
            root = Path(directory)
            trial = dataset.prepare_capacity_trial({"tokenizer": root}, root, groups, 7, 8)
            prime = json.loads(trial["prime"].read_text())
            probe = json.loads(trial["probe"].read_text())
            fills = [json.loads(line) for line in trial["fill"].read_text().splitlines()]
            order = [json.loads(line) for line in trial["order"].read_text().splitlines()]
            records = [prime, *fills, probe]
            ids = [encode(row["text"]) for row in records]
            self.assertEqual(trial["requests"], {"prime": 1, "fill": 10, "probe": 1})
            self.assertEqual(trial["validation"]["calibration_fill_mode"], "single_use_independent")
            self.assertEqual([row["session_id"] for row in records],
                             [row["session_id"] for row in order])
            self.assertEqual(sorted(row["prefix_group"] for row in order[1:-1]), list(range(1, groups)))
            self.assertEqual(dataset._common_length(ids[0], ids[-1]), 48)
            self.assertTrue(all(len(row) == 64 for row in ids))
            self.assertTrue(all(row["output_length"] == 8 for row in records))
            self.assertEqual(len({tuple(row[2:6]) for row in ids[:-1]}), groups)
            self.assertEqual(sum("compare" in call.kwargs for call in fit.call_args_list), 1)
            self.assertEqual(fit.call_count, groups + 2)
            for name, digest in trial["files"].items():
                self.assertEqual(dataset.sha256(root / name), digest)


if __name__ == "__main__":
    unittest.main()
