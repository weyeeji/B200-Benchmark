"""Measure vLLM GPU prefix-cache capacity using AIPerf reuse-distance trials."""

import math
import random
import re
import statistics
import time
from urllib.parse import urlsplit

import capacity_metrics
import dataset
from metrics import record_counts
from runner import clear_cache, run_aiperf, utc_now, verify_dispatch, write_json

ISL, _, PREFIX = dataset.PROFILES["P03"]


def add_arguments(parser):
    group = parser.add_argument_group("measure-cache options")
    group.add_argument("--metrics-url", help="vLLM Prometheus URL (default: API origin /metrics)")
    group.add_argument("--max-groups", type=int, default=32, help="Largest working set to test")
    group.add_argument("--start-groups", type=int, default=2, help="First size to test after the baseline")
    group.add_argument("--cache-affinity-header", help="Send a stable routing key on each trial's anchor requests")
    group.add_argument("--cache-affinity-key", help="Override the generated routing key")
    group.add_argument("--pin-fill", action="store_true", help="Also route the fill to the anchor engine")
    group.add_argument("--calibration-trials", type=int, default=3, help="Independent trials per size")
    group.add_argument("--hit-fraction", type=float, default=0.95, help="Fraction of single-group GPU hits required")
    group.add_argument("--calibration-osl", type=int, default=8, help="Short fixed output for each request")


def validate_options(args, cfg):
    if args.max_groups < 2 or args.calibration_trials < 1 or args.calibration_osl < 1:
        raise ValueError("Use max-groups >= 2, calibration-trials >= 1 and calibration-osl >= 1")
    if not (0 < args.hit_fraction <= 1):
        raise ValueError("hit-fraction must be in (0, 1]")
    if not 2 <= args.start_groups <= args.max_groups:
        raise ValueError("start-groups must be between 2 and max-groups")
    if args.cache_affinity_header and not re.fullmatch(r"[A-Za-z0-9-]+", args.cache_affinity_header):
        raise ValueError("Invalid cache affinity header name")
    if (args.cache_affinity_key is not None or args.pin_fill) and not args.cache_affinity_header:
        raise ValueError("cache-affinity-key and pin-fill require cache-affinity-header")
    if args.cache_affinity_key is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.cache_affinity_key):
        raise ValueError("Invalid cache affinity key")
    url = args.metrics_url or cfg["api_url"] + "/metrics"
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.query or parsed.fragment):
        raise ValueError("metrics-url must be an HTTP(S) endpoint with no embedded credentials or query")
    return url


def measure_phase(cfg, url, data, name, trial_dir, previous=None, *, headers=None):
    count = data["requests"][name]
    artifact = trial_dir / name
    path = trial_dir / f"{name}-metrics.json"
    evidence = {"started_utc": utc_now(), "expected_requests": count}
    try:
        before = capacity_metrics.snapshot(url, cfg["model"], timeout=5)
        evidence["before"] = before
        if previous is not None:
            capacity_metrics.phase_delta(previous, before, expected=0)
        rc = run_aiperf(cfg, data[name], artifact, count, concurrency=1, headers=headers)
        evidence["aiperf_returncode"] = rc
        after = capacity_metrics.snapshot(url, cfg["model"], timeout=5)
        evidence["after"] = after
        counts = record_counts(artifact, isl=data["isl"], osl=data["osl"])
        dispatch = verify_dispatch(artifact, data[name])
        evidence.update(counts=counts, dispatch=dispatch)
        if (rc or dispatch["status"] != "ordered" or dispatch["recorded"] != count
                or counts["completed_requests"] != count or counts["isl_matching_requests"] != count
                or counts["osl_matching_requests"] != count):
            raise ValueError(f"{name}: AIPerf order, success or server Usage validation failed")
        evidence["delta"] = capacity_metrics.phase_delta(before, after, expected=count)
        return after, evidence["delta"]
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise ValueError(f"{name}: {error}") from error
    finally:
        evidence["finished_utc"] = utc_now()
        write_json(path, evidence)


def measure_trial(args, cfg, url, directory, groups, trial_number, seed):
    directory.mkdir()
    result = {"trial": trial_number, "seed": seed, "status": "inconclusive", "artifact_dir": str(directory)}
    try:
        data = dataset.prepare_capacity_trial(cfg, directory, groups, seed, args.calibration_osl)
        headers = None
        if args.cache_affinity_header:
            key = args.cache_affinity_key or f"capacity-{seed:016x}"
            headers = {args.cache_affinity_header: key}
        cache = clear_cache(cfg, directory, "capacity", 1, trial_number)
        result["cache_clear"] = cache["status"]
        write_json(directory / "cache-clear.json", cache)
        if cfg["cache_hook"]["require_confirmed"] and cache["status"] != "confirmed":
            raise ValueError("Cache-clear tiers were not confirmed")

        previous, prime = measure_phase(cfg, url, data, "prime", directory, headers=headers)
        primed_at = time.monotonic()
        result["prime_engine"] = prime["engine"]
        if groups > 1:
            previous, fill = measure_phase(cfg, url, data, "fill", directory, previous,
                                           headers=headers if args.pin_fill else None)
            result["fill_requests_by_engine"] = {engine: row["completed"]
                                                 for engine, row in fill["per_engine"].items()}
            if args.pin_fill and result["fill_requests_by_engine"].get(prime["engine"]) != groups - 1:
                raise ValueError("The fill phase did not stay on the anchor engine")
        else:
            result["fill_requests_by_engine"] = {}
        _, probe = measure_phase(cfg, url, data, "probe", directory, previous, headers=headers)
        result.update(probe_engine=probe["engine"], gpu_hit_tokens=probe["device_tokens"],
                      post_prime_through_probe_s=time.monotonic() - primed_at)
        if (prime["engine"] is None or prime["engine"] != probe["engine"]):
            raise ValueError("Prime and probe reached different engines")
        if probe["device_tokens"] > data["prefix_tokens"]:
            raise ValueError("GPU hit tokens exceeded the known shared prefix")
        result["status"] = "observed"
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "trial.json", result)
    return result


def scan(assess, max_groups, start_groups=2):
    """Return adjacent passing and failing sizes, or the tested lower bound."""
    low, high = 1, None
    while low < max_groups:
        candidate = min(start_groups if low == 1 else 2 * low, max_groups)
        outcome = assess(candidate)
        if outcome == "fail":
            high = candidate
            break
        if outcome != "pass":
            return low, None, outcome
        low = candidate
    if high is None:
        return low, None, "lower_bound_only"
    while high - low > 1:
        candidate = (high + low) // 2
        outcome = assess(candidate)
        if outcome == "pass":
            low = candidate
        elif outcome == "fail":
            high = candidate
        else:
            return low, high, outcome
    return low, high, "measured"


def run_calibration(args, cfg, directory, url, run_seed):
    report = {
        "status": "running", "created_utc": utc_now(), "api_url": cfg["api_url"], "model": cfg["model"],
        "metrics_url": url, "gpu_count_declared": cfg["gpu_count"], "concurrency": 1,
        "target_isl_tokens": ISL, "target_osl_tokens": args.calibration_osl,
        "prefix_tokens": PREFIX, "max_groups": args.max_groups, "trials_per_group_count": args.calibration_trials,
        "hit_fraction": args.hit_fraction, "run_seed": run_seed, "gpu_hit_metric":
        'vllm:prompt_tokens_cached_by_source_total{source="device"}',
        "cache_capacity_tokens": None, "tests": {}, "start_groups": args.start_groups,
        "cache_affinity_header": args.cache_affinity_header, "cache_affinity_key": args.cache_affinity_key,
        "measurement_scope": "single_engine_per_trial" if args.pin_fill else "serving_endpoint",
    }
    path = directory / "capacity.json"
    write_json(path, report)
    print(path, flush=True)
    rng = random.Random(run_seed)

    def collect(groups):
        print(f"cache capacity: testing {groups} prefix group(s)", flush=True)
        samples = []
        for trial in range(1, args.calibration_trials + 1):
            result = measure_trial(args, cfg, url, directory / f"g{groups}-t{trial}", groups, trial,
                                   rng.getrandbits(63))
            samples.append(result)
            if result["status"] != "observed":
                break
        point = {"trials": samples}
        report["tests"][str(groups)] = point
        write_json(path, report)
        if any(item["status"] != "observed" for item in samples):
            point["outcome"] = "inconclusive"
            report["last_error"] = samples[-1]["error"]
            return point, []
        return point, [item["gpu_hit_tokens"] for item in samples]

    try:
        baseline, hits = collect(1)
        if not hits:
            report["status"] = "inconclusive"
        else:
            reference = statistics.median(hits)
            threshold = math.ceil(reference * args.hit_fraction)
            report.update(single_group_gpu_hit_tokens=reference, gpu_hit_pass_threshold_tokens=threshold)
            if not threshold or min(hits) < threshold:
                baseline["outcome"] = report["status"] = "no_stable_gpu_hit_baseline"
            else:
                baseline["outcome"] = "pass"

                def assess(groups):
                    point, values = collect(groups)
                    if values:
                        point["outcome"] = ("pass" if min(values) >= threshold else
                                            "fail" if max(values) < threshold else "unstable_boundary")
                    write_json(path, report)
                    return point["outcome"]

                low, high, status = scan(assess, args.max_groups, args.start_groups)
                report.update(status=status, largest_passing_groups=low, lower_bound_tokens=low * PREFIX,
                              smallest_observed_failing_groups=high,
                              first_failed_working_set_tokens=None if high is None else high * PREFIX)
                if status == "measured":
                    report["cache_capacity_tokens"] = low * PREFIX
                    if not args.pin_fill:
                        report["derived_prefix_groups"] = {"P03": low // 2, "P04": 2 * low}
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    finally:
        report["finished_utc"] = utc_now()
        write_json(path, report)

    print(f"status={report['status']} cache_capacity_tokens={report['cache_capacity_tokens']}", flush=True)
    if report.get("last_error"):
        print(report["last_error"], flush=True)
    return 0 if report["status"] == "measured" else 2
