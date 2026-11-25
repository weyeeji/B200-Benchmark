#!/usr/bin/env python3
"""Plan or run the three Benchmark workloads against an OpenAI-compatible API."""

import argparse
import json
import os
import shutil
import sys
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("P01", "P02", "P03")
TOKENIZER = ROOT / "tokenizer"
HF_HOME = ROOT / "datasets/huggingface"
CHAT_TEMPLATE = {"thinking": True, "reasoning_effort": "high"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be in [0, 2**32)")
    return number


def http_url(value):
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise argparse.ArgumentTypeError("expected an http:// or https:// URL")
    if url.username is not None or url.query or url.fragment:
        raise argparse.ArgumentTypeError("URL credentials, query, and fragment are unsupported")
    return value.rstrip("/")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    common.add_argument("--concurrency", nargs="+", type=positive_int, default=(16, 32, 64, 128, 256))
    common.add_argument("--model", default="DeepSeek-V4.1-Flash-1017", help="served model ID")
    common.add_argument("--max-context-length", type=positive_int, help="P03 recorded peak context ceiling")
    common.add_argument("--seed", type=seed_value, default=42)
    common.add_argument("--dataset-nonce", help="optional reproducible P01/P02 dataset identifier")
    common.add_argument("--request-timeout", type=positive_int, default=600, help="seconds per HTTP request")
    common.add_argument("--metrics-url", type=http_url, help="optional Prometheus exposition URL")
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan", parents=[common], help="print parameters and AIPerf commands")
    plan.add_argument("--url", type=http_url, default="http://127.0.0.1:8000")
    run = actions.add_parser("run", parents=[common], help="run the selected experiments")
    run.add_argument("--url", type=http_url, required=True, help="inference server base URL")
    args = parser.parse_args()
    if list(args.concurrency) != sorted(set(args.concurrency)):
        parser.error("--concurrency must be strictly increasing")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("--experiments must not contain duplicates")
    if urlsplit(args.url).path.rstrip("/").endswith(("/v1", "/v1/chat/completions")):
        parser.error("--url must be the API base before /v1/chat/completions")
    return args


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def workload(experiment):
    return json.loads((ROOT / experiment / "workload.json").read_text(encoding="utf-8"))


def shared_command(args, artifacts):
    return [
        sys.executable, "-m", "aiperf", "profile",
        "--url", args.url, "--model", args.model, "--tokenizer", str(TOKENIZER),
        "--endpoint-type", "chat", "--streaming", "--use-server-token-count",
        "--request-timeout-seconds", str(args.request_timeout),
        "--random-seed", str(args.seed), "--artifact-dir", str(artifacts),
        "--export-level", "records", "--ui", "none", "--no-gpu-telemetry",
        "--no-server-metrics", "--no-auto-plot",
    ]


def synthetic_phase(args, settings, directory, phase, concurrency, count):
    artifacts = directory / phase
    dataset = ROOT / "datasets/synthetic" / directory.relative_to(ROOT / "results") / f"{phase}.jsonl"
    extra = {
        "ignore_eos": True,
        "min_tokens": settings["output_tokens"],
        "chat_template_kwargs": CHAT_TEMPLATE,
    }
    command = shared_command(args, artifacts) + [
        "--input-file", str(dataset), "--custom-dataset-type", "single_turn",
        "--dataset-sampling-strategy", "sequential", "--concurrency", str(concurrency),
        "--request-count", str(count), "--extra-inputs", json.dumps(extra, separators=(",", ":")),
    ]
    return {"name": phase, "artifacts": str(artifacts), "dataset": str(dataset), "requests": count,
            "concurrency": concurrency, "argv": command}


def point_plan(args, run_id, experiment, concurrency):
    settings = workload(experiment)
    directory = ROOT / "results" / run_id / experiment / f"C{concurrency}"
    if experiment != "P03":
        warmup = settings["warmup_requests"]
        phases = [
            synthetic_phase(args, settings, directory, "warmup", min(warmup, concurrency), warmup),
            synthetic_phase(args, settings, directory, "formal", concurrency,
                            settings["requests_per_concurrency"] * concurrency),
        ]
    else:
        artifacts = directory / "formal"
        command = shared_command(args, artifacts) + [
            "--scenario", settings["scenario"], "--public-dataset", settings["public_dataset"],
            "--concurrency", str(concurrency), "--benchmark-duration", str(settings["profile_seconds"]),
            "--benchmark-grace-period", str(settings["drain_seconds"]),
            "--warmup-requests-per-lane", str(settings["warmup_requests_per_lane"]),
            "--agentic-warmup-grace-period", str(settings["warmup_grace_seconds"]),
            "--cache-bust", settings["cache_bust"], "--extra-inputs", "ignore_eos:true",
        ]
        if args.max_context_length is not None:
            command.extend(("--max-context-length", str(args.max_context_length)))
        phases = [{"name": "formal", "artifacts": str(artifacts), "concurrency": concurrency,
                   "argv": command}]

    return {"experiment": experiment, "concurrency": concurrency, "settings": settings, "phases": phases}


def create_plan(args):
    run_id = timestamp()
    return {
        "run_id": run_id,
        "model": args.model,
        "url": args.url,
        "metrics_url": args.metrics_url,
        "seed": args.seed,
        "dataset_nonce": args.dataset_nonce if args.dataset_nonce is not None else run_id,
        "max_context_length": args.max_context_length,
        "request_timeout_seconds": args.request_timeout,
        "points": [point_plan(args, run_id, experiment, concurrency)
                   for experiment in args.experiments for concurrency in args.concurrency],
    }


def client_environment():
    return {
        **os.environ,
        "HF_HOME": str(HF_HOME),
        "HF_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_HOME / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def failed_point(point, status, detail):
    return {"experiment": point["experiment"], "concurrency": point["concurrency"],
            "status": status, "detail": detail}


def run_point(point, plan, logs):
    from report import summarize

    experiment = point["experiment"]
    concurrency = point["concurrency"]
    settings = point["settings"]
    point_log = logs / experiment / f"C{concurrency}"
    point_log.mkdir(parents=True)
    print(f"{experiment} C{concurrency}  {point_log}", flush=True)
    write_json(point_log / "plan.json", point)
    metrics_url = plan["metrics_url"]
    before = None
    for phase in point["phases"]:
        name = phase["name"]
        logfile = point_log / f"{name}.log"
        try:
            if "dataset" in phase:
                from synthetic import write_dataset

                dataset = Path(phase["dataset"])
                digest = write_dataset(
                    dataset, TOKENIZER, experiment, concurrency,
                    input_tokens=settings["input_tokens"], output_tokens=settings["output_tokens"],
                    count=phase["requests"], seed=plan["seed"], phase=name, nonce=plan["dataset_nonce"],
                )
                write_json(point_log / f"{name}.dataset.json",
                           {"path": str(dataset), "sha256": digest, "requests": phase["requests"]})
            if name == "formal" and metrics_url:
                from server_metrics import capture

                before = capture(metrics_url, point_log / "metrics-before.prom")
            artifacts = Path(phase["artifacts"])
            try:
                with logfile.open("wb") as stream:
                    process = subprocess.run(
                        phase["argv"], cwd=ROOT, env=client_environment(),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    )
            finally:
                native_logs = artifacts / "logs"
                if native_logs.is_dir():
                    shutil.move(native_logs, point_log / f"{name}.aiperf")
            if process.returncode != 0:
                return failed_point(point, "error", f"{name}: AIPerf exit {process.returncode}; {logfile}")
            row = summarize(
                artifacts, experiment, concurrency,
                expected_requests=phase.get("requests"), expected_input=settings.get("input_tokens"),
                expected_output=settings.get("output_tokens"),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return failed_point(point, "error", f"{name}: {exc}")
        if name == "warmup" and row["status"] in ("invalid", "error"):
            return failed_point(point, row["status"], f"warmup: {row['detail']}")
        if name == "formal" and metrics_url:
            from server_metrics import capture, enrich

            after = capture(metrics_url, point_log / "metrics-after.prom")
            enrich(row, before, after, plan["model"], includes_warmup=(experiment == "P03"))
    return row


def run_plan(plan):
    from report import write_reports

    if not (TOKENIZER / "tokenizer.json").is_file():
        raise FileNotFoundError(f"missing tokenizer at {TOKENIZER}")
    agentx_points = [point for point in plan["points"] if point["experiment"] == "P03"]
    if agentx_points:
        dataset_manifest = ROOT / "datasets/agentx.json"
        dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if not (ROOT / dataset["offline_cache"]).is_dir():
            raise FileNotFoundError(f"missing offline AgentX dataset cache at {HF_HOME}")
        if any(point["settings"]["public_dataset"] != dataset["public_alias"] for point in agentx_points):
            raise ValueError("P03 public_dataset does not match the prepared cache")
        plan["agentx_dataset"] = dataset
    plan["aiperf_version"] = version("aiperf")
    plan["python_version"] = sys.version.split()[0]
    logs = ROOT / "logs" / plan["run_id"]
    results = ROOT / "results" / plan["run_id"]
    logs.mkdir(parents=True)
    results.mkdir(parents=True)
    write_json(logs / "run.json", plan)
    rows = []
    stopped = set()
    for point in plan["points"]:
        experiment = point["experiment"]
        if experiment in stopped:
            continue
        row = run_point(point, plan, logs)
        rows.append(row)
        write_reports(results, rows)
        if row["status"] != "pass":
            stopped.add(experiment)
        print(f"{experiment} C{row['concurrency']} {row['status']}: {row['detail']}", flush=True)

    print(f"Results: {results}", flush=True)
    return int(any(row["status"] in ("invalid", "error") for row in rows))


def main():
    args = parse_args()
    plan = create_plan(args)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        return run_plan(plan)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
