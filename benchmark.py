#!/usr/bin/env python3
"""Prepare and run reproducible DeepSeek serving benchmarks with NVIDIA AIPerf."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import dataset
import measure_cache
import metrics
import report as result_report
from runner import (SLA_TAGS, clear_cache, probe_eos, recommended_duration, run_aiperf,
                    sha256, utc_now, verify_dispatch, verify_payload, write_json)

HOME = Path(__file__).resolve().parent
VERSION = "0.14.0"
NA = "N/A"
DEFAULTS = {
    "api_url": None, "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "tokenizer": "tokenizer", "chat_encoder": "deepseek",
    "chat_template_kwargs": {"thinking": True}, "gpu_count": 6,
    "scenes": list(dataset.PROFILES), "concurrencies": [32, 64, 128, 256],
    "duration_s": 300, "repeats": 1, "seed": 20261125,
    "request_timeout_s": 300, "cache_capacity_tokens": None, "prefix_groups": {},
    "dataset_dir": "datasets", "output_dir": "results",
    "api_key_env": "OPENAI_API_KEY", "eos_probe_file": None,
    "use_legacy_max_tokens": False, "per_chunk_usage": False,
    "cache_hook": {"command": [], "timeout_s": 60, "require_confirmed": False},
    "prometheus": {"urls": [], "rules": {}}, "sla": {},
    "deployment": {"name": None, "details": {},
                   "speculative": {"mode": "unknown", "algorithm": None,
                                   "draft_length": None, "simulated_acceptance_length": None}},
}


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "plan", "probe", "measure-cache", "pilot", "run", "report"])
    parser.add_argument("--from-run", nargs="+", type=Path, help="Run directories for a combined HTML report")
    parser.add_argument("--config", type=Path, default=HOME / "config.example.json")
    parser.add_argument("--api-url")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--gpus", dest="gpu_count", type=int)
    parser.add_argument("--scenes", nargs="+", choices=dataset.PROFILES)
    parser.add_argument("--concurrencies", nargs="+", type=int)
    parser.add_argument("--duration", dest="duration_s", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--cache-capacity", dest="cache_capacity_tokens", type=int)
    parser.add_argument("--p03-groups", type=int)
    parser.add_argument("--p04-groups", type=int)
    parser.add_argument("--data-dir", dest="dataset_dir")
    parser.add_argument("--out-dir", dest="output_dir")
    parser.add_argument("--eos-probe-file")
    parser.add_argument("--eos-token-id", type=int, help="EOS ID override for probe")
    measure_cache.add_arguments(parser)
    return parser.parse_args()


def merge(defaults, custom):
    result = defaults.copy()
    for key, value in custom.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(defaults[key], dict) and defaults[key] and key != "chat_template_kwargs":
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object for {key}")
            value = merge(defaults[key], value)
        result[key] = value
    return result


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_prometheus(config):
    urls, rules = config["urls"], config["rules"]
    if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
        raise ValueError("prometheus.urls must be a list of HTTP(S) URLs")
    if len(set(urls)) != len(urls):
        raise ValueError("prometheus.urls must not repeat addresses")
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.query or parsed.fragment):
            raise ValueError("Use explicit Prometheus HTTP(S) URLs without embedded credentials")
    if not isinstance(rules, dict):
        raise ValueError("prometheus.rules must be an object")
    denominators = {
        "ratio", "ratio_plus_one", "histogram_ratio", "gauge_difference",
        "counter_rate_difference", "counter_delta_difference",
    }
    for name, configured in rules.items():
        if name not in metrics.MONITORING_DEFAULTS and not re.fullmatch(r"prom_[a-z0-9_]+", name):
            raise ValueError(f"Custom Prometheus fields must start with prom_: {name}")
        choices = configured if isinstance(configured, list) else [configured]
        if not choices:
            raise ValueError(f"Prometheus rule list is empty: {name}")
        for rule in choices:
            if (not isinstance(rule, dict) or rule.get("kind") not in metrics.RULE_KINDS
                    or not isinstance(rule.get("metric"), str) or not rule["metric"]):
                raise ValueError(f"Invalid Prometheus rule kind or metric: {name}")
            denominator = rule.get("denominator")
            if denominator is not None or rule["kind"] in denominators:
                if not isinstance(denominator, str) or not denominator:
                    raise ValueError(f"Invalid Prometheus denominator: {name}")
            if not positive(rule.get("scale", 1)):
                raise ValueError(f"Prometheus scale must be positive: {name}")
            for key in ("labels", "label_regex", "denominator_labels"):
                mapping = rule.get(key, {})
                if not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values()):
                    raise ValueError(f"Prometheus {key} must map labels to strings: {name}")
            for pattern in rule.get("label_regex", {}).values():
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise ValueError(f"Invalid Prometheus label pattern: {name}") from error
            for key in ("endpoints", "ignore_labels"):
                items = rule.get(key, [])
                if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                    raise ValueError(f"Prometheus {key} must be a list of strings: {name}")
            if set(rule.get("endpoints", [])) - set(urls):
                raise ValueError(f"Prometheus rule endpoints must be listed in prometheus.urls: {name}")


def load_config(args):
    path = args.config.expanduser().resolve()
    custom = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(custom, dict):
        raise ValueError("The configuration file must contain a JSON object")
    cfg = merge(DEFAULTS, custom)
    for key in ("prefix_groups", "sla"):
        if not isinstance(cfg[key], dict):
            raise ValueError(f"{key} must be an object")
    for key in ("use_legacy_max_tokens", "per_chunk_usage"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    cfg.update({key: getattr(args, key) for key in vars(args)
                if key in cfg and getattr(args, key) is not None})
    cfg["prefix_groups"] = cfg["prefix_groups"].copy()
    for key in ("p03", "p04"):
        value = getattr(args, f"{key}_groups")
        if value is not None:
            cfg["prefix_groups"][key.upper()] = value
    for key in ("tokenizer", "dataset_dir", "output_dir", "eos_probe_file"):
        if cfg[key] is not None:
            cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    for key in ("duration_s", "request_timeout_s"):
        if not positive(cfg[key]):
            raise ValueError(f"{key} must be positive and finite")
    for key in ("gpu_count", "repeats"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("scenes", "concurrencies"):
        values = cfg[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list of unique values")
    if any(scene not in dataset.PROFILES for scene in cfg["scenes"]):
        raise ValueError("Unknown scene")
    if any(type(value) is not int or value < 1 for value in cfg["concurrencies"]):
        raise ValueError("concurrencies must be positive integers")
    if set(cfg["prefix_groups"]) - {"P03", "P04"}:
        raise ValueError("Only P03 and P04 have prefix groups")
    if set(cfg["sla"]) - SLA_TAGS.keys() or any(not positive(v) for v in cfg["sla"].values()):
        raise ValueError("Use positive ttft_ms, tpot_ms, itl_ms and/or e2el_ms for SLA")
    if not isinstance(cfg["chat_template_kwargs"], dict) or cfg["chat_encoder"] not in ("deepseek", "hf"):
        raise ValueError("Set chat_encoder and chat_template_kwargs for the serving template")
    validate_prometheus(cfg["prometheus"])
    hook = cfg["cache_hook"]
    if not isinstance(hook["command"], list) or any(not isinstance(x, str) or not x for x in hook["command"]):
        raise ValueError("cache_hook.command must be an argv list of nonempty strings")
    if type(hook["require_confirmed"]) is not bool:
        raise ValueError("cache_hook.require_confirmed must be true or false")
    if not positive(hook["timeout_s"]):
        raise ValueError("cache_hook.timeout_s must be positive and finite")
    spec = cfg["deployment"]["speculative"]
    if spec["mode"] not in ("unknown", "off", "real", "simulated"):
        raise ValueError("Speculative mode must be unknown, off, real or simulated")
    simulated = spec["simulated_acceptance_length"]
    if spec["mode"] == "simulated":
        if not positive(simulated):
            raise ValueError("Simulated mode requires a positive simulated_acceptance_length")
    elif simulated is not None:
        raise ValueError("simulated_acceptance_length requires speculative mode simulated")
    if spec["draft_length"] is not None and (type(spec["draft_length"]) is not int or spec["draft_length"] < 1):
        raise ValueError("Speculative draft length must be a positive integer")
    if cfg["api_url"] is not None:
        url = cfg["api_url"].rstrip("/").removesuffix("/v1/chat/completions").removesuffix("/v1")
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path
                or parsed.username is not None or parsed.query or parsed.fragment):
            raise ValueError("api_url must be an HTTP(S) origin, optionally ending in /v1/chat/completions or /v1")
        cfg["api_url"] = url
    elif args.action in ("probe", "measure-cache", "pilot", "run"):
        raise ValueError("Pass --api-url or set api_url in the configuration")
    return cfg


def probe_signature(report, cfg):
    keys = ("api_url", "model", "chat_template_kwargs", "use_legacy_max_tokens")
    return all(report.get(key) == cfg[key] for key in keys)


def run_probe(args, cfg):
    eos_id = args.eos_token_id
    if eos_id is None:
        config = Path(cfg["tokenizer"]) / "config.json"
        eos_id = json.loads(config.read_text(encoding="utf-8")).get("eos_token_id")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("Set --eos-token-id to the model's actual integer EOS ID")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (HOME / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = probe_eos(cfg, eos_id)
    report["tokenizer_json_sha256"] = sha256(Path(cfg["tokenizer"]) / "tokenizer.json")
    target = output / f"eos-probe-{uuid.uuid4().hex[:8]}.json"
    write_json(target, report)
    print(f"{report['status']}  {target}", flush=True)
    return int(report["status"] == "failed_forced_eos")


def metadata(directory, cfg, kind):
    write_json(directory / "config.json", cfg)
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    paths = [HOME / name for name in ("benchmark.py", "runner.py", "dataset.py", "metrics.py",
                                     "measure_cache.py", "balanced_cache.py", "capacity_metrics.py", "report.py",
                                     "report_charts.py", "report_trends.py", "requirements.txt")]
    write_json(directory / "experiment.json", {
        "created_utc": utc_now(), "kind": kind, "python": sys.version,
        "platform": platform.platform(), "host": platform.node(),
        "packages": packages, "sha256": {path.name: sha256(path) for path in paths if path.is_file()},
    })


def point_row(cfg, data, concurrency, repeat, directory, kind):
    spec = cfg["deployment"]["speculative"]
    return {
        "report_schema": result_report.SCHEMA,
        "server_metrics_semantics": 2,
        "deployment": cfg["deployment"]["name"] or NA, "kind": kind,
        "model": cfg["model"], "api_url": cfg["api_url"], "profile": data["profile"],
        "concurrency": concurrency, "gpu_count": cfg["gpu_count"], "repeat": repeat,
        "duration_cap_s": cfg["duration_s"] if kind == "run" else NA,
        "request_timeout_s": cfg["request_timeout_s"], "per_chunk_usage": cfg["per_chunk_usage"],
        "sla_signature": json.dumps(cfg["sla"], sort_keys=True, separators=(",", ":")),
        "prefix_groups": data["prefix_groups"], "prefix_tokens": data["prefix_tokens"],
        "prefix_routing": "managed_by_serving_endpoint" if data["prefix_groups"] else NA,
        "theoretical_shared_prefix_ratio": data["theoretical_shared_prefix_ratio"],
        "warmup_target_requests": data["warmup_requests"],
        "formal_target_requests": data["formal_requests"],
        "formal_planned_full_cycles": data["formal_full_cycles"] or NA,
        "dataset_manifest_sha256": sha256(data["manifest"]),
        "dataset_formal_sha256": data["files"]["formal.jsonl"],
        "dataset_warmup_sha256": data["files"]["warmup.jsonl"],
        "speculative_mode": spec["mode"], "speculative_algorithm": spec["algorithm"] or NA,
        "speculative_draft_length": spec["draft_length"] if spec["draft_length"] is not None else NA,
        "simulated_acceptance_length": (
            spec["simulated_acceptance_length"] if spec["simulated_acceptance_length"] is not None else NA
        ),
        "artifact_dir": str(directory), "started_utc": utc_now(),
    }


def run_point(cfg, data, concurrency, repeat, directory, kind, eos):
    directory.mkdir()
    write_json(directory / "config.json", cfg)
    shutil.copyfile(data["manifest"], directory / "dataset-manifest.json")
    if eos:
        write_json(directory / "eos-probe.json", eos)
    row = point_row(cfg, data, concurrency, repeat, directory, kind)
    isl, osl = data["isl"], data["osl"]
    requested_warmup, requested_formal = data["warmup_requests"], data["formal_requests"]
    formal = directory / "formal"
    issues = []
    try:
        cache = clear_cache(cfg, directory, data["profile"], concurrency, repeat)
        write_json(directory / "cache-clear.json", cache)
        for key in ("status", "gpu", "cpu", "ssd"):
            row[f"cache_clear_{key}"] = cache[key]
        if cfg["cache_hook"]["require_confirmed"] and cache["status"] != "confirmed":
            raise RuntimeError("Cache-clear tiers were not confirmed")
        if eos and eos["status"] == "failed_forced_eos":
            raise RuntimeError("Configured forced-EOS probe failed")

        warmup = directory / "warmup"
        warmup_rc = run_aiperf(cfg, data["warmup"], warmup, requested_warmup,
                               min(concurrency, 16, requested_warmup))
        row["warmup_returncode"] = warmup_rc
        warmup_counts = metrics.record_counts(warmup, isl=isl, osl=osl)
        warmup_order = verify_dispatch(warmup, data["warmup"])
        warmup_payload = verify_payload(warmup, data["warmup"], cfg,
                                        expected_sha256=data["files"]["warmup.jsonl"])
        write_json(warmup / "validation.json", {"counts": warmup_counts, "dispatch": warmup_order,
                                                "wire_payload": warmup_payload})
        row["warmup_wire_payload_status"] = warmup_payload["status"]
        row["warmup_completed_requests"] = warmup_counts["completed_requests"]
        if (warmup_rc or warmup_order["status"] != "ordered"
                or warmup_order["recorded"] != requested_warmup
                or warmup_counts["completed_requests"] != requested_warmup):
            raise RuntimeError("Warmup requests failed or their dispatch order was not verified")
        if (warmup_counts["isl_matching_requests"] != requested_warmup
                or warmup_counts["osl_matching_requests"] != requested_warmup):
            raise RuntimeError("Warmup server Usage did not confirm all target token lengths")
        if warmup_payload["status"] != "verified" or warmup_payload["verified"] != requested_warmup:
            raise RuntimeError("Warmup wire payloads did not match the frozen workload")

        rc = run_aiperf(cfg, data["formal"], formal, requested_formal, concurrency,
                        cfg["duration_s"] if kind == "run" else None, monitor=True)
        row["formal_returncode"] = rc
        dispatch = verify_dispatch(formal, data["formal"])
        write_json(formal / "dispatch.json", dispatch)
        formal_payload = verify_payload(formal, data["formal"], cfg,
                                        expected_sha256=data["files"]["formal.jsonl"])
        write_json(formal / "payload-validation.json", formal_payload)
        row["formal_wire_payload_status"] = formal_payload["status"]
        row["dispatch_order_status"] = dispatch["status"]
        row["last_dispatch_s"] = dispatch["last_dispatch_s"] if dispatch["last_dispatch_s"] is not None else NA
        if rc:
            issues.append(f"AIPerf exited with status {rc}")
        if dispatch["status"] != "ordered" or dispatch["recorded"] != requested_formal:
            issues.append(f"Ordered request quota incomplete: {dispatch['recorded']}/{requested_formal}")
        if formal_payload["status"] != "verified" or formal_payload["verified"] != dispatch["recorded"]:
            issues.append("Formal wire payloads did not match the frozen workload")
    except KeyboardInterrupt:
        issues.append("Run interrupted")
        row["interrupted"] = True
    except Exception as error:
        issues.append(f"{type(error).__name__}: {error}")
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")

    try:
        row.update(metrics.summarize(formal, concurrency=concurrency, gpu_count=cfg["gpu_count"],
                                     isl=isl, osl=osl, sla=cfg["sla"], prometheus=cfg["prometheus"],
                                     tokenizer_path=cfg["tokenizer"], model_name=cfg["model"],
                                     input_token_count=dataset.input_token_counter(cfg)))
        if row["completed_requests"] != requested_formal and formal.exists():
            issues.append("Formal request successes are below quota")
        if row["completed_requests"] and (row["isl_matching_requests"] != row["completed_requests"]
                                             or row["osl_matching_requests"] != row["completed_requests"]):
            issues.append("Server Usage did not confirm all target token lengths")
        if (row["steady_seconds"] == NA or not isinstance(row["steady_completed_requests"], int)
                or not row["steady_completed_requests"]):
            issues.append("No successful requests completed in the steady window")
        peak = row.get("actual_concurrency_peak")
        if type(peak) is int and peak > concurrency:
            issues.append(f"Observed request concurrency exceeded target: {peak}/{concurrency}")
    except Exception as error:
        with (directory / "error.log").open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        issues.append(f"Could not reduce AIPerf artifacts: {error}")

    row["forced_eos_probe_status"] = eos["status"] if eos else "unverified"
    comparison_notes = []
    if row.get("cache_clear_status") != "confirmed":
        comparison_notes.append("缓存清理未确认")
    if row["forced_eos_probe_status"] != "verified_by_forced_eos":
        comparison_notes.append("没有已通过的强制 EOS 对照")
    row["comparison_notes"] = "；".join(comparison_notes) or NA
    row["valid_for_comparison"] = not issues and not comparison_notes and kind == "run"
    row["status"] = "incomplete" if issues else ("pilot_only" if kind == "pilot" else "complete")
    row["issues"] = " | ".join(issues) or NA
    row["finished_utc"] = utc_now()
    write_json(directory / "point.json", row)
    return row


def run_experiment(cfg, points, kind):
    version = importlib.metadata.version("aiperf")
    if version != VERSION:
        raise ValueError(f"Expected AIPerf {VERSION}, found {version}")
    eos = None
    if cfg["eos_probe_file"]:
        eos = json.loads(Path(cfg["eos_probe_file"]).read_text(encoding="utf-8"))
        if (not probe_signature(eos, cfg)
                or eos.get("tokenizer_json_sha256") != sha256(Path(cfg["tokenizer"]) / "tokenizer.json")):
            raise ValueError("Forced-EOS probe metadata does not match this endpoint and tokenizer")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (HOME / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = utc_now()[:19].replace(":", "")
        directory = output / f"{kind}-{run_id}-{uuid.uuid4().hex[:8]}"
        directory.mkdir()
        metadata(directory, cfg, kind)
        print(directory, flush=True)
        rows = []
        for profile, concurrency, data in points:
            for repeat in range(1, cfg["repeats"] + 1):
                point = directory / f"{profile}-c{concurrency}-r{repeat}"
                print(f"{kind} {profile} concurrency={concurrency} repeat={repeat}", flush=True)
                row = run_point(cfg, data, concurrency, repeat, point, kind, eos)
                rows.append(row)
                metrics.write_summaries(directory, rows)
                result_report.write_report(directory / "report.html", rows)
                print(f"{row['status']}  {row['issues']}", flush=True)
                if row.get("interrupted"):
                    return 130
        if kind == "pilot":
            completed = [row["last_dispatch_s"] for row in rows
                         if row["status"] == "pilot_only" and isinstance(row["last_dispatch_s"], (int, float))]
            recommendation = (recommended_duration(cfg["duration_s"], max(completed))
                              if completed and len(completed) == len(rows) else None)
            write_json(directory / "pilot-duration.json", {
                "baseline_duration_s": cfg["duration_s"], "recommended_duration_s": recommendation,
                "status": "all_selected_points_verified" if recommendation is not None else "pilot_incomplete",
            })
            print(f"Recommended unified dispatch cap: {recommendation if recommendation is not None else NA}", flush=True)
        return int(any(row["status"] == "incomplete" for row in rows))


def run_cache_measurement(args, cfg):
    version = importlib.metadata.version("aiperf")
    if version != VERSION:
        raise ValueError(f"Expected AIPerf {VERSION}, found {version}")
    url = measure_cache.validate_options(args, cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (HOME / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = utc_now()[:19].replace(":", "")
        nonce = uuid.uuid4()
        directory = output / f"cache-capacity-{run_id}-{nonce.hex[:8]}"
        directory.mkdir()
        metadata(directory, cfg, "measure-cache")
        return measure_cache.run_calibration(args, cfg, directory, url, cfg["seed"] ^ nonce.int)


def compare_reports(paths, cfg):
    if not paths:
        raise ValueError("Pass --from-run followed by one or more run directories")
    rows = []
    seen = set()
    labels = set()
    for supplied in paths:
        directory = supplied.expanduser().resolve()
        if directory in seen:
            raise ValueError(f"Duplicate run directory: {directory}")
        seen.add(directory)
        raw = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw or any(
            not isinstance(row, dict) or row.get("report_schema") != result_report.SCHEMA for row in raw
        ):
            raise ValueError(f"Unsupported summary metric schema: {directory}")
        label = directory.name
        suffix = 2
        while label in labels:
            label = f"{directory.name} ({suffix})"
            suffix += 1
        labels.add(label)
        rows.extend(row | {"run_source": label} for row in raw)
    rows.sort(key=lambda row: (row["profile"], row["concurrency"], row["run_source"], row["repeat"]))
    directory = Path(cfg["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    stamp = utc_now()[:19].replace(":", "")
    target = directory / f"comparison-{stamp}-{uuid.uuid4().hex[:8]}.html"
    result_report.write_report(target, rows, combined=True)
    print(target, flush=True)
    return 0


def main():
    args = options()
    try:
        cfg = load_config(args)
        if args.action == "report":
            return compare_reports(args.from_run, cfg)
        if args.from_run:
            raise ValueError("--from-run is only used by report")
        if args.action == "probe":
            return run_probe(args, cfg)
        if args.action == "measure-cache":
            return run_cache_measurement(args, cfg)
        points = []
        for profile in cfg["scenes"]:
            for concurrency in cfg["concurrencies"]:
                data = dataset.prepare(cfg, Path(cfg["dataset_dir"]), profile, concurrency,
                                       create=args.action == "prepare")
                points.append((profile, concurrency, data))
                print(f"{profile} c{concurrency}: warmup={data['warmup_requests']} formal={data['formal_requests']} "
                      f"groups={data['prefix_groups']}  {data['manifest']}", flush=True)
        if args.action in ("run", "pilot"):
            return run_experiment(cfg, points, args.action)
        return 0
    except BlockingIOError:
        print("Another benchmark is already active in this Benchmark directory.", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
