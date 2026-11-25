#!/usr/bin/env python3
"""Plan or run the three Benchmark workloads against an OpenAI-compatible API."""

import argparse
import json
import os
import shutil
import signal
import sys
import time
import subprocess
from datetime import datetime, timezone
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
    common.add_argument("--repeats", type=positive_int, default=1, help="repeat each P01/P02 concurrency point")
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan", parents=[common], help="print parameters and AIPerf commands")
    plan.add_argument("--url", type=http_url, default="http://127.0.0.1:8000")
    run = actions.add_parser("run", parents=[common], help="run the selected experiments")
    run.add_argument("--url", type=http_url, required=True, help="inference server base URL")
    run.add_argument("--deployment-info", type=Path, help="optional deployment metadata JSON")
    run.add_argument("--backend-container", help="capture relevant launch metadata from a local Docker container")
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


def point_plan(args, run_id, experiment, concurrency, *, repeat, repeat_count, dataset_nonce):
    settings = workload(experiment)
    point_dir = Path(experiment) / f"C{concurrency}"
    if repeat_count > 1:
        point_dir /= f"R{repeat:02d}"
    directory = ROOT / "results" / run_id / point_dir
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

    return {
        "experiment": experiment, "concurrency": concurrency, "settings": settings, "phases": phases,
        "point_dir": str(point_dir), "repeat": repeat, "repeat_count": repeat_count,
        "dataset_nonce": dataset_nonce if repeat == 1 else f"{dataset_nonce}/repeat:{repeat}",
    }


def create_plan(args):
    run_id = timestamp()
    dataset_nonce = args.dataset_nonce if args.dataset_nonce is not None else run_id
    points = []
    for experiment in args.experiments:
        repeat_count = 1 if experiment == "P03" else args.repeats
        for concurrency in args.concurrency:
            points.extend(point_plan(args, run_id, experiment, concurrency, repeat=repeat,
                                     repeat_count=repeat_count, dataset_nonce=dataset_nonce)
                          for repeat in range(1, repeat_count + 1))
    return {
        "run_id": run_id,
        "model": args.model,
        "url": args.url,
        "metrics_url": args.metrics_url,
        "seed": args.seed,
        "dataset_nonce": dataset_nonce,
        "synthetic_repeats": args.repeats,
        "max_context_length": args.max_context_length,
        "request_timeout_seconds": args.request_timeout,
        "points": points,
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


def failed_point(status, detail):
    return {
        "status": status, "sla_status": "unknown", "validity": "incomplete",
        "detail": detail, "validity_reason": detail, "failure_reason": detail,
    }


def mark_bad_point(row, validity, detail):
    prior = row.get("validity_reason")
    row["validity"] = validity
    row["status"] = "error" if validity == "incomplete" else "invalid"
    row["validity_reason"] = "；".join(dict.fromkeys(filter(None, (prior, detail))))
    row["failure_reason"] = detail
    row["detail"] = row["validity_reason"]


def prepare_synthetic(phase, point, plan, point_log):
    from synthetic import write_dataset

    settings = point["settings"]
    dataset = Path(phase["dataset"])
    digest = write_dataset(
        dataset, TOKENIZER, point["experiment"], point["concurrency"],
        input_tokens=settings["input_tokens"], output_tokens=settings["output_tokens"],
        count=phase["requests"], seed=plan["seed"], phase=phase["name"], nonce=point["dataset_nonce"],
    )
    write_json(point_log / f"{phase['name']}.dataset.json",
               {"path": str(dataset), "sha256": digest, "requests": phase["requests"]})
    return digest


def wait_for_aiperf(argv, stream):
    with subprocess.Popen(
        argv, cwd=ROOT, env=client_environment(), start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
    ) as process:
        try:
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            return 130


def run_aiperf_phase(phase, point_log, experiment, metrics_url):
    from observability import PhaseObserver

    name = phase["name"]
    artifacts = Path(phase["artifacts"])
    logfile = point_log / f"{name}.log"
    detail = None
    observer = PhaseObserver(artifacts, point_log, logical_phase=name, experiment=experiment,
                             metrics_url=metrics_url if name == "formal" else None)
    try:
        with logfile.open("wb") as stream, observer:
            try:
                status_code = wait_for_aiperf(phase["argv"], stream)
                if status_code == 130:
                    detail = f"{name}: 运行被中断"
                elif status_code:
                    detail = f"{name}: AIPerf 退出码 {status_code}"
            except OSError as exc:
                status_code = 1
                detail = f"{name}: {exc}"
            observer.returncode = status_code
    finally:
        native_logs = artifacts / "logs"
        if native_logs.is_dir():
            shutil.move(native_logs, point_log / f"{name}.aiperf")
    return observer, status_code, detail


def phase_seconds(phase, start, end):
    first = phase.get(start)
    last = phase.get(end)
    if isinstance(first, int) and isinstance(last, int) and first <= last:
        return (last - first) / 1e9
    return None


def add_phase_evidence(row, summaries, point):
    formal = summaries.get("formal", {})
    warmup = summaries.get("warmup", {})
    for field in (
        "profile_start_ns", "send_end_ns", "requests_end_ns", "in_flight_at_cutoff",
        "grace_period_timeout_triggered", "final_requests_completed", "final_requests_cancelled",
        "final_request_errors", "drain_last_response_seconds",
    ):
        if (value := formal.get(field)) is not None:
            row[field] = value
    preparation = [formal.get(name) for name in ("dataset_seconds", "generation_seconds")]
    if any(duration is not None for duration in preparation):
        row["prepare_seconds"] = sum(duration or 0 for duration in preparation)
    row["warmup_seconds"] = formal.get("warmup_seconds")
    if row["warmup_seconds"] is None:
        row["warmup_seconds"] = phase_seconds(warmup, "profile_start_ns", "requests_end_ns")
    row["send_seconds"] = phase_seconds(formal, "profile_start_ns", "send_end_ns")
    row["drain_seconds"] = phase_seconds(formal, "send_end_ns", "requests_end_ns")
    if point["experiment"] == "P03":
        row["configured_send_seconds"] = point["settings"]["profile_seconds"]
        row["configured_drain_seconds"] = point["settings"]["drain_seconds"]


def run_point(point, plan, logs, results):
    from report import summarize

    experiment = point["experiment"]
    concurrency = point["concurrency"]
    settings = point["settings"]
    point_log = logs / point["point_dir"]
    point_results = results / point["point_dir"]
    point_log.mkdir(parents=True)
    point_results.mkdir(parents=True)
    repeat_note = f" R{point['repeat']}" if point["repeat_count"] > 1 else ""
    print(f"{experiment} C{concurrency}{repeat_note}  {point_log}", flush=True)
    write_json(point_log / "plan.json", point)

    digests = {}
    summaries = {}
    for phase in point["phases"]:
        name = phase["name"]
        try:
            generated_in = None
            if "dataset" in phase:
                started = time.monotonic()
                digests[name] = prepare_synthetic(phase, point, plan, point_log)
                generated_in = time.monotonic() - started
            observer, code, execution_error = run_aiperf_phase(
                phase, point_log, experiment, plan["metrics_url"],
            )
            summaries[name] = observer.summary
            if generated_in is not None:
                summaries[name]["generation_seconds"] = generated_in
            row = summarize(
                Path(phase["artifacts"]), experiment, concurrency,
                expected_requests=phase.get("requests"), expected_input=settings.get("input_tokens"),
                expected_output=settings.get("output_tokens"),
            )
            if name == "formal":
                try:
                    observer.enrich(row, plan["model"])
                except (OSError, ValueError, RuntimeError) as exc:
                    summaries[name]["telemetry_error"] = str(exc)
            if observer.summary.get("grace_period_timeout_triggered"):
                mark_bad_point(row, "invalid", f"{name}: 收尾达到上限，仍有请求未返回")
            elif observer.summary.get("final_requests_cancelled"):
                mark_bad_point(row, "invalid", f"{name}: 阶段日志记录了取消的请求")
            if execution_error:
                validity = "invalid" if row.get("validity") == "invalid" and code != 130 else "incomplete"
                mark_bad_point(row, validity, execution_error)
            row["exit_code"] = code
        except KeyboardInterrupt:
            row = failed_point("error", f"{name}: 运行被中断")
            row["exit_code"] = 130
        except (OSError, ValueError, RuntimeError) as exc:
            row = failed_point("error", f"{name}: {exc}")
        if name == "warmup" and row["status"] in ("invalid", "error"):
            summaries.setdefault(name, {})["warmup_result"] = row
            status, code = row["status"], row.get("exit_code")
            row = failed_point(status, f"预热未完成，正式阶段未开始。{row['detail']}")
            if status == "invalid":
                row["validity"] = "invalid"
            row["exit_code"] = code
            break

    row.update(
        experiment=experiment, concurrency=concurrency, repeat=point["repeat"], repeat_count=point["repeat_count"],
        dataset_sha256=digests.get("formal"), warmup_dataset_sha256=digests.get("warmup"),
        point_dir=point["point_dir"], artifact_dir=os.path.relpath(phase["artifacts"], results),
        log_dir=os.path.relpath(point_log, results),
    )
    add_phase_evidence(row, summaries, point)
    if summaries:
        summary = point_results / "phase-summary.json"
        write_json(summary, summaries)
        row["phase_summary"] = str(summary.relative_to(results))
    return row


def run_plan(plan, *, deployment_file=None, backend_container=None):
    from report import write_reports
    from run_metadata import capture_run_metadata

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
    plan.update(capture_run_metadata(ROOT, TOKENIZER, deployment_file=deployment_file,
                                     backend_container=backend_container))
    logs = ROOT / "logs" / plan["run_id"]
    results = ROOT / "results" / plan["run_id"]
    logs.mkdir(parents=True)
    results.mkdir(parents=True)
    write_json(logs / "run.json", plan)
    write_json(results / "manifest.json", plan)
    rows = []
    stop_above = {}
    for point in plan["points"]:
        experiment = point["experiment"]
        concurrency = point["concurrency"]
        if experiment in stop_above and concurrency > stop_above[experiment]:
            continue
        row = run_point(point, plan, logs, results)
        rows.append(row)
        write_reports(results, rows, manifest=plan)
        if row["status"] != "pass":
            stop_above[experiment] = concurrency
        print(f"{experiment} C{concurrency} R{row['repeat']} {row['status']}: {row['detail']}", flush=True)
        if row.get("exit_code") == 130:
            print(f"Results: {results}", flush=True)
            return 130

    print(f"Results: {results}", flush=True)
    return int(any(row["status"] in ("invalid", "error") for row in rows))


def main():
    args = parse_args()
    plan = create_plan(args)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        return run_plan(plan, deployment_file=args.deployment_info, backend_container=args.backend_container)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
