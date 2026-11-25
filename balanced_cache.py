"""Calibrate equal per-engine GPU working sets using verified routing affinity."""

import math
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import capacity_metrics
import dataset
from measure_cache import ISL, PREFIX, check_cache_delta, execute_phase, scan
from runner import clear_cache, utc_now, write_json


def scoped_deltas(before, after, counts, data, phase):
    delta = capacity_metrics.phase_delta(before, after, expected=sum(counts.values()))
    if {engine: row["completed"] for engine, row in delta["per_engine"].items()} != counts:
        raise ValueError(f"{phase}: requests did not reach the configured engines")
    for engine, row in delta["per_engine"].items():
        check_cache_delta(row, data[engine], counts[engine], phase)
    return delta


def balanced_phase(cfg, url, group_dir, phase, data, headers, pool, cancelled, previous):
    counts = {engine: values["requests"][phase] for engine, values in data.items()}
    evidence = {"expected_by_engine": counts, "started_utc": utc_now(),
                "aiperf": {engine: {} for engine in data}}

    def send(engine):
        execute_phase(cfg, data[engine], phase, data[engine][phase].with_suffix(""),
                      evidence["aiperf"][engine], headers=headers[engine], cancel_event=cancelled)

    try:
        before = capacity_metrics.snapshot(url, cfg["model"], 5)
        evidence["before"] = before
        if before.keys() != counts.keys():
            raise ValueError("Configured engine labels must exactly match the monitored model's engine labels")
        if previous is not None:
            capacity_metrics.phase_delta(previous, before, expected=0)
        else:
            capacity_metrics.phase_delta(before, before, expected=0)
        if phase == "fill":
            list(pool.map(send, counts))
            after = capacity_metrics.snapshot(url, cfg["model"], 5)
        else:
            # One request per scrape interval proves each affinity key reached its
            # declared engine. Equal aggregate counts cannot detect swapped keys.
            after = before
            evidence["routing_checks"] = {}
            for engine in counts:
                first = after
                send(engine)
                after = capacity_metrics.snapshot(url, cfg["model"], 5)
                expected = dict.fromkeys(counts, 0) | {engine: 1}
                evidence["routing_checks"][engine] = scoped_deltas(first, after, expected, data, phase)
        evidence["after"] = after
        evidence["delta"] = scoped_deltas(before, after, counts, data, phase)
        return after, evidence["delta"]
    except BaseException as error:
        cancelled.set()
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["finished_utc"] = utc_now()
        # Copy completed results so optional cancellation cannot mutate the object being written.
        saved = evidence | {"aiperf": {e: dict(row) for e, row in evidence["aiperf"].items()}}
        write_json(group_dir / f"{phase}-metrics.json", saved)


def balanced_trial(args, cfg, url, root, size, number, rng, pool, cancelled):
    group_dir = root / f"g{size}-t{number}"
    group_dir.mkdir()
    headers = {engine: {args.cache_affinity_header: key} for engine, key in args.engine_affinity}
    seeds = {engine: rng.getrandbits(63) for engine in headers}
    row = {"groups_per_engine": size, "trial": number, "seeds": seeds, "artifact_dir": str(group_dir)}

    def prepare(item):
        position, engine = item
        return engine, dataset.prepare_capacity_trial(
            cfg, group_dir / f"engine-{position}", size, seeds[engine], args.calibration_osl,
        )

    try:
        data = dict(pool.map(prepare, enumerate(headers)))
        cache = clear_cache(cfg, group_dir, "capacity", len(headers), number)
        write_json(group_dir / "cache-clear.json", cache)
        row["cache_clear"] = cache["status"]
        if cfg["cache_hook"]["require_confirmed"] and cache["status"] != "confirmed":
            raise ValueError("Cache-clear tiers were not confirmed")

        previous = None
        for phase in ("prime", "fill", "probe"):
            if phase == "fill" and size == 1:
                continue
            previous, delta = balanced_phase(cfg, url, group_dir, phase, data, headers, pool, cancelled, previous)
        hits = {engine: record["device"] for engine, record in delta["per_engine"].items()}
        if any(value > PREFIX for value in hits.values()):
            raise ValueError("GPU hit tokens exceeded the known shared prefix")
        row.update(status="observed", gpu_hit_tokens=hits)
    except (OSError, RuntimeError, ValueError) as error:
        row.update(status="inconclusive", error=f"{type(error).__name__}: {error}")
    write_json(group_dir / "trial.json", row)
    return row


def assess_rows(rows, thresholds):
    if any(row["status"] != "observed" for row in rows):
        return "inconclusive"
    passes = [all(row["gpu_hit_tokens"][engine] >= needed for engine, needed in thresholds.items())
              for row in rows]
    return "pass" if all(passes) else "unstable_boundary" if any(passes) else "fail"


def run_balanced(args, cfg, directory, url, run_seed):
    routes = dict(args.engine_affinity)
    limit, trials = args.max_groups, args.calibration_trials
    report = {"status": "running", "measurement_scope": "balanced_working_set",
              "created_utc": utc_now(), "api_url": cfg["api_url"], "metrics_url": url, "model": cfg["model"],
              "gpu_count_declared": cfg["gpu_count"], "engine_routes": routes,
              "cache_affinity_header": args.cache_affinity_header, "concurrency_per_engine": 1,
              "target_isl_tokens": ISL, "target_osl_tokens": args.calibration_osl, "prefix_tokens": PREFIX,
              "max_groups_per_engine": limit, "start_groups_per_engine": args.start_groups,
              "baseline_min_fraction": args.baseline_min_fraction, "hit_fraction": args.hit_fraction,
              "confirmation_trials": trials, "run_seed": run_seed, "tests": {}, "cache_capacity_tokens": None}
    path = directory / "capacity.json"
    rng = random.Random(run_seed)
    cancelled = Event()
    write_json(path, report)
    print(path, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(routes)) as pool:
            try:
                def collect(size, total):
                    point = report["tests"].setdefault(str(size), {"trials": []})
                    rows = point["trials"]
                    while len(rows) < total and not any(row["status"] != "observed" for row in rows):
                        print(f"cache capacity: {size} groups per engine, trial {len(rows) + 1}", flush=True)
                        row = balanced_trial(args, cfg, url, directory, size, len(rows) + 1, rng, pool, cancelled)
                        rows.append(row)
                        if row["status"] != "observed":
                            report["last_error"] = row["error"]
                        write_json(path, report)
                    return rows

                baseline = collect(1, trials)
                if any(row["status"] != "observed" for row in baseline):
                    report["status"] = "inconclusive"
                else:
                    refs = {engine: statistics.median(row["gpu_hit_tokens"][engine] for row in baseline)
                            for engine in routes}
                    thresholds = {engine: math.ceil(hit * args.hit_fraction) for engine, hit in refs.items()}
                    report.update(single_group_gpu_hit_tokens=refs, gpu_hit_pass_threshold_tokens=thresholds)
                    if (min(refs.values()) < PREFIX * args.baseline_min_fraction
                            or assess_rows(baseline, thresholds) != "pass"):
                        report["status"] = "no_stable_gpu_hit_baseline"
                    else:
                        def outcome(size, total=1):
                            result = assess_rows(collect(size, total), thresholds)
                            report["tests"][str(size)]["outcome"] = result
                            write_json(path, report)
                            return result

                        low, high, status = scan(outcome, limit, args.start_groups)
                        if status in ("measured", "lower_bound_only"):
                            verdicts = [(outcome(low, trials), "pass")]
                            if high is not None and verdicts[0][0] == "pass":
                                verdicts.append((outcome(high, trials), "fail"))
                            if any(actual != expected for actual, expected in verdicts):
                                status = ("inconclusive" if any(actual == "inconclusive" for actual, _ in verdicts)
                                          else "unstable_boundary")
                        report.update(status=status, candidate_groups_per_engine={"pass": low, "fail": high})
                        if status in ("measured", "lower_bound_only"):
                            floor = low * len(routes) * PREFIX
                            report["lower_bound_tokens"] = floor
                            if status == "measured":
                                report.update(cache_capacity_tokens=floor,
                                              first_failed_working_set_tokens=high * len(routes) * PREFIX)
                                if low * len(routes) >= 2:
                                    report["derived_prefix_groups"] = {"P03": low * len(routes) // 2,
                                                                       "P04": 2 * low * len(routes)}
            except BaseException:
                cancelled.set()
                raise
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except (OSError, RuntimeError, ValueError) as error:
        report.update(status="inconclusive", last_error=f"{type(error).__name__}: {error}")
    finally:
        report["finished_utc"] = utc_now()
        write_json(path, report)
    print(f"status={report['status']} cache_capacity_tokens={report['cache_capacity_tokens']}", flush=True)
    if report.get("last_error"):
        print(report["last_error"], flush=True)
    return 0 if report["status"] == "measured" else 2
