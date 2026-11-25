"""Regression tests for frozen benchmark dataset archives."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import dataset


def stub_generate(directory, settings, requests, tokenizer_path, hashes, *, phase_orders=None):
    for phase, count in requests.items():
        (directory / f"{phase}.jsonl").write_text(f"{phase} {count}\n", encoding="utf-8")
    (directory / "order.jsonl").write_text("sample order\n", encoding="utf-8")
    return {}


class FrozenDatasetTest(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(TemporaryDirectory())
        self.root = Path(temporary)
        self.config = {"tokenizer": self.root / "tokenizer"}
        self.hashes = self.enterContext(
            patch.object(dataset, "_tokenizer_hashes", return_value={"tokenizer.json": "hash"})
        )
        self.generate = self.enterContext(patch.object(dataset, "_generate", side_effect=stub_generate))

    def test_plan_rejects_changed_manifest_and_never_creates_lock(self):
        result = dataset.prepare(self.config, self.root, "P01", 1)
        lock = self.root / "P01" / ".prepare.lock"
        lock.unlink()
        dataset.prepare(self.config, self.root, "P01", 1, create=False)
        self.assertFalse(lock.exists())
        manifest = result["manifest"]
        contents = json.loads(manifest.read_text())
        contents["osl"] = 7
        manifest.write_text(json.dumps(contents))
        with self.assertRaisesRegex(ValueError, "configuration mismatch for osl"):
            dataset.prepare(self.config, self.root, "P01", 1, create=False)

    def test_cached_siblings_share_one_capacity_even_when_created_separately(self):
        with_capacity = self.config | {"cache_capacity_tokens": 500000}
        p03 = dataset.prepare(with_capacity, self.root, "P03", 1)
        self.assertEqual(p03["prefix_groups"], 4)
        self.assertEqual(p03["settings"]["prefix_groups_source"], "capacity")
        with self.assertRaisesRegex(ValueError, "frozen baseline cache capacity 500000"):
            dataset.prepare(self.config | {"cache_capacity_tokens": 500001}, self.root, "P04", 1)
        p04 = dataset.prepare(self.config, self.root, "P04", 1)
        self.assertEqual(p04["prefix_groups"], 17)
        self.assertEqual(p04["settings"]["baseline_cache_capacity_tokens"], 500000)
        self.assertEqual(p04["formal_full_cycles"], 2)
        self.assertEqual(p04["warmup_requests"], 17)
        frozen = dataset.prepare(self.config, self.root, "P03", 1, create=False)
        self.assertEqual(frozen["files"], p03["files"])

    def test_explicit_sibling_groups_do_not_select_derivation_capacity(self):
        p03 = dataset.prepare(
            self.config | {"cache_capacity_tokens": 600000, "prefix_groups": {"P03": 3}},
            self.root, "P03", 1,
        )
        self.assertEqual(p03["settings"]["prefix_groups_source"], "explicit")
        self.assertIsNone(p03["settings"]["baseline_cache_capacity_tokens"])
        p04 = dataset.prepare(self.config | {"cache_capacity_tokens": 500000}, self.root, "P04", 1)
        self.assertEqual(p04["settings"]["baseline_cache_capacity_tokens"], 500000)
        with self.assertRaisesRegex(ValueError, "different frozen baseline"):
            dataset.prepare(self.config | {"prefix_groups": {"P03": 4}}, self.root, "P03", 1)

    def test_changed_tokenizer_is_not_published(self):
        self.hashes.side_effect = [{"tokenizer.json": "initial"}, {"tokenizer.json": "changed"}]
        with self.assertRaisesRegex(ValueError, "Tokenizer snapshot changed"):
            dataset.prepare(self.config, self.root, "P02", 1)
        self.assertFalse((self.root / "P02" / "c1").exists())
        self.assertFalse((self.root / "P02" / "baseline.json").exists())

    def test_capacity_trial_has_one_publisher(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            attempts = [
                executor.submit(dataset.prepare_capacity_trial, self.config, self.root, 1, 77, 4)
                for _ in range(2)
            ]
        successes = [attempt.result() for attempt in attempts if attempt.exception() is None]
        failures = [attempt.exception() for attempt in attempts if attempt.exception() is not None]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], FileExistsError)
        self.assertEqual(self.generate.call_count, 1)
        result = successes[0]
        for filename, checksum in result["files"].items():
            self.assertEqual(dataset.sha256(self.root / filename), checksum)
        self.assertEqual(result["requests"], {"prime": 1, "fill": 0, "probe": 1})


if __name__ == "__main__":
    unittest.main()
