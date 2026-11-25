"""Prepare the pinned AgentX corpus for AIPerf's offline public loader."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "cc-traces-weka-062126"
DATASET_REPO = f"semianalysisai/{DATASET_NAME}"
PUBLIC_ALIAS = "semianalysis_cc_traces_weka_062126"
REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
TRACE_SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
TRACE_BYTES = 1_847_151_435
TRACE_COUNT = 393
HF_HOME = ROOT / "datasets" / "huggingface"
DEFAULT_TRACE = ROOT / "datasets" / "source" / DATASET_NAME / "traces.jsonl"


def prepare(raw_trace: Path) -> dict:
    if raw_trace.stat().st_size != TRACE_BYTES:
        raise ValueError(f"Unexpected corpus size: {raw_trace}")
    with raw_trace.open("rb") as stream:
        observed_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed_hash != TRACE_SHA256:
        raise ValueError(f"Unexpected corpus SHA-256: {observed_hash}")

    # The AIPerf public loader resolves the dataset by repository name.
    cache = HF_HOME / "datasets"
    os.environ.update(
        HF_HOME=str(HF_HOME),
        HF_DATASETS_CACHE=str(cache),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    from datasets import load_dataset
    from datasets.packaged_modules.json.json import Json
    from filelock import FileLock

    repo_cache = cache / DATASET_REPO.replace("/", "___")
    repo_cache.mkdir(parents=True, exist_ok=True)
    target = repo_cache / "default" / "0.0.0" / REVISION
    with FileLock(str(repo_cache / ".prepare.lock")):
        if not target.is_dir():
            with tempfile.TemporaryDirectory(prefix="agentx-", dir=cache) as work:
                staging = Path(work)
                builder = Json(
                    cache_dir=str(staging / "build"),
                    dataset_name=DATASET_NAME,
                    config_name="default",
                    data_files={"train": [str(raw_trace)]},
                    hash=REVISION,
                )
                output = staging / "prepared"
                builder.download_and_prepare(output_dir=str(output))
                if builder.info.splits["train"].num_examples != TRACE_COUNT:
                    raise ValueError("The prepared corpus has an unexpected trace count.")
                target.parent.mkdir(parents=True, exist_ok=True)
                output.rename(target)

        dataset = load_dataset(DATASET_REPO, split="train", streaming=False)
        cache_files = [Path(item["filename"]).resolve() for item in dataset.cache_files]
        if len(dataset) != TRACE_COUNT or not cache_files or any(
            not path.is_relative_to(target) for path in cache_files
        ):
            raise ValueError("The public dataset alias did not resolve to the pinned cache.")

        manifest = {
            "dataset_repo": DATASET_REPO,
            "public_alias": PUBLIC_ALIAS,
            "revision": REVISION,
            "trace_bytes": TRACE_BYTES,
            "trace_sha256": TRACE_SHA256,
            "trace_count": TRACE_COUNT,
            "offline_cache": str(target.relative_to(ROOT)),
            "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        }
        manifest_path = ROOT / "datasets" / "agentx.json"
        temp_manifest = manifest_path.with_suffix(".json.part")
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces_jsonl", type=Path, nargs="?", default=DEFAULT_TRACE)
    args = parser.parse_args()
    manifest = prepare(args.traces_jsonl.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
