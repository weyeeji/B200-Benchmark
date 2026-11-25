#!/usr/bin/env python3
"""Prepare and run reproducible DeepSeek serving benchmarks with NVIDIA AIPerf."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import dataset
import measure_cache
import metrics
from runner import (SLA_TAGS, clear_cache, probe_eos, recommended_duration, run_aiperf,
                    sha256, utc_now, verify_dispatch, write_json)

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
    parser.add_argument("action", choices=["prepare", "plan", "probe", "measure-cache", "pilot", "run"])
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


def load_config(args):
    path = args.config.expanduser().resolve()
    cfg = merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))
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
    hook = cfg["cache_hook"]
    if not isinstance(hook["command"], list) or any(not isinstance(x, str) or not x for x in hook["command"]):
        raise ValueError("cache_hook.command must be an argv list of nonempty strings")
    if not positive(hook["timeout_s"]):
        raise ValueError("cache_hook.timeout_s must be positive and finite")
    spec = cfg["deployment"]["speculative"]
    if spec["mode"] not in ("unknown", "off", "real", "simulated"):
        raise ValueError("Speculative mode must be unknown, off, real or simulated")
    if spec["mode"] in ("real", "off") and spec["simulated_acceptance_length"] is not None:
        raise ValueError("Simulated acceptance length conflicts with speculative mode")
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
                                     "measure_cache.py", "capacity_metrics.py", "requirements.txt")]
    write_json(directory / "experiment.json", {
        "created_utc": utc_now(), "kind": kind, "python": sys.version,
        "platform": platform.platform(), "host": platform.node(),
        "packages": packages, "sha256": {path.name: sha256(path) for path in paths if path.is_file()},
    })


def point_row(cfg, data, concurrency, repeat, directory, kind):
    spec = cfg["deployment"]["speculative"]
    return {
        "deployment": cfg["deployment"]["name"] or NA, "kind": kind,
        "model": cfg["model"], "api_url": cfg["api_url"], "profile": data["profile"],
        "concurrency": concurrency, "gpu_count": cfg["gpu_count"], "repeat": repeat,
        "duration_cap_s": cfg["duration_s"] if kind == "run" else NA,
        "prefix_groups": data["prefix_groups"], "prefix_tokens": data["prefix_tokens"],
        "theoretical_shared_prefix_ratio": data["theoretical_shared_prefix_ratio"],
        "warmup_target_requests": data["warmup_requests"],
        "formal_target_requests": data["formal_requests"],
        "formal_planned_full_cycles": data["formal_full_cycles"] or NA,
        "dataset_manifest_sha256": sha256(data["manifest"]),
        "dataset_formal_sha256": sha256(data["formal"]),
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
        write_json(warmup / "validation.json", {"counts": warmup_counts, "dispatch": warmup_order})
        row["warmup_completed_requests"] = warmup_counts["completed_requests"]
        if (warmup_rc or warmup_order["status"] != "ordered"
                or warmup_counts["completed_requests"] != requested_warmup):
            raise RuntimeError("Warmup requests failed or their dispatch order was not verified")

        rc = run_aiperf(cfg, data["formal"], formal, requested_formal, concurrency,
                        cfg["duration_s"] if kind == "run" else None, monitor=True)
        row["formal_returncode"] = rc
        dispatch = verify_dispatch(formal, data["formal"])
        write_json(formal / "dispatch.json", dispatch)
        row["dispatch_order_status"] = dispatch["status"]
        row["last_dispatch_s"] = dispatch["last_dispatch_s"] if dispatch["last_dispatch_s"] is not None else NA
        if rc:
            issues.append(f"AIPerf exited with status {rc}")
        if dispatch["status"] != "ordered" or dispatch["recorded"] != requested_formal:
            issues.append(f"Ordered request quota incomplete: {dispatch['recorded']}/{requested_formal}")
    except Exception as error:
        issues.append(f"{type(error).__name__}: {error}")
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")

    try:
        row.update(metrics.summarize(formal, concurrency=concurrency, gpu_count=cfg["gpu_count"],
                                     isl=isl, osl=osl, sla=cfg["sla"], prometheus=cfg["prometheus"],
                                     tokenizer_path=cfg["tokenizer"],
                                     input_token_count=dataset.input_token_counter(cfg)))
        if row["completed_requests"] != requested_formal and formal.exists():
            issues.append("Formal request successes are below quota")
        if row["completed_requests"] and (row["isl_matching_requests"] != row["completed_requests"]
                                             or row["osl_matching_requests"] != row["completed_requests"]):
            issues.append("Server Usage did not confirm all target token lengths")
        if (row["steady_seconds"] == NA or not isinstance(row["steady_completed_requests"], int)
                or not row["steady_completed_requests"]):
            issues.append("No successful requests completed in the steady window")
    except Exception as error:
        with (directory / "error.log").open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        issues.append(f"Could not reduce AIPerf artifacts: {error}")

    row["forced_eos_probe_status"] = eos["status"] if eos else "unverified"
    row["valid_for_comparison"] = not issues and kind == "run"
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
        run_id = utc_now().split(".", 1)[0].replace(":", "")
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
                print(f"{row['status']}  {row['issues']}", flush=True)
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
        run_id = utc_now().split(".", 1)[0].replace(":", "")
        nonce = uuid.uuid4()
        directory = output / f"cache-capacity-{run_id}-{nonce.hex[:8]}"
        directory.mkdir()
        metadata(directory, cfg, "measure-cache")
        return measure_cache.run_calibration(args, cfg, directory, url, cfg["seed"] ^ nonce.int)


def main():
    args = options()
    try:
        cfg = load_config(args)
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
