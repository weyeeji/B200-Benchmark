"""Read vLLM cache counters around isolated capacity-calibration phases."""

from urllib.request import Request, urlopen

from prometheus_client.parser import text_string_to_metric_families

CACHE = "vllm:prompt_tokens_cached_by_source_total"
COMPLETIONS = "vllm:request_success_total"
QUERIES = "vllm:prefix_cache_queries_total"
GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting"}
TIERS = ("device", "host", "disk", "p2p", "external_unspecified")
COUNTERS = {"completed", "queries", *TIERS}
IDLE_GAUGES = tuple(GAUGES.values())


def is_count(value):
    return type(value) in (int, float) and value >= 0 and (
        type(value) is int or value.is_integer()
    )


def validate_snapshot(engines):
    """Require independent idleness and counter evidence for each engine."""
    required = {"device", "completed", *IDLE_GAUGES}
    if any("queries" in values for values in engines.values()):
        required.add("queries")

    for engine, values in engines.items():
        missing = required - values.keys()
        if missing:
            raise ValueError(f"Engine {engine} lacks metric evidence: {', '.join(sorted(missing))}")

        finish_reasons = values.get("finish_reasons")
        if not isinstance(finish_reasons, dict) or not finish_reasons:
            raise ValueError(f"Engine {engine} lacks request completion counters by finish reason")

        for field in values.keys() & (COUNTERS | set(IDLE_GAUGES)):
            if not is_count(values[field]):
                raise ValueError(f"Invalid {field} value for engine {engine}: {values[field]}")
        if any(not is_count(value) for value in finish_reasons.values()):
            raise ValueError(f"Invalid finish reason counter for engine {engine}")
        if values["completed"] != sum(finish_reasons.values()):
            raise ValueError(f"Engine {engine} has inconsistent request completion counters")


def parse_snapshot(exposition, model):
    """Extract per-engine token counters and request activity for one model."""
    engines = {}
    relevant = {CACHE, COMPLETIONS, QUERIES, *GAUGES}
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            name, labels = sample.name, sample.labels
            if name not in relevant or labels.get("model_name") != model:
                continue
            engine = labels.get("engine")
            if not engine:
                raise ValueError(f"{name} for {model} lacks the vLLM engine label")
            metric = GAUGES.get(name)
            if name == CACHE:
                metric = labels.get("source")
                if metric not in TIERS:
                    continue
            if name == QUERIES:
                metric = "queries"
            values = engines.setdefault(engine, {"finish_reasons": {}})
            if name == COMPLETIONS:
                metric = labels.get("finished_reason")
                if not metric:
                    raise ValueError(f"{name} for engine {engine} lacks a finish reason")
                values = values["finish_reasons"]
            value = sample.value
            if not is_count(value):
                raise ValueError(f"Invalid {name} value for engine {engine}: {value}")
            if metric in values:
                raise ValueError(f"Duplicate {name} values for engine {engine}")
            values[metric] = int(value)

    if not engines:
        raise ValueError(f"No vLLM cache counters match model {model!r}")
    for values in engines.values():
        values["completed"] = sum(values["finish_reasons"].values())
    validate_snapshot(engines)
    return engines


def snapshot(url, model, timeout):
    request = Request(url, headers={"Accept": "text/plain; version=0.0.4"})
    with urlopen(request, timeout=timeout) as response:
        exposition = response.read().decode("utf-8")
    return parse_snapshot(exposition, model)


def phase_delta(before, after, *, expected):
    """Check isolation and return counter deltas for a completed phase."""
    if not before or before.keys() != after.keys():
        raise ValueError("vLLM engine topology changed between metrics scrapes")
    validate_snapshot(before)
    validate_snapshot(after)
    per_engine = {}
    for engine, first in before.items():
        last = after[engine]
        if any(values[field] for values in (first, last) for field in IDLE_GAUGES):
            raise ValueError(f"Engine {engine} had requests in flight at a phase boundary")
        old = {key: value for key, value in first.items() if key in COUNTERS}
        new = {key: value for key, value in last.items() if key in COUNTERS}
        if old.keys() != new.keys() or first["finish_reasons"].keys() != last["finish_reasons"].keys():
            raise ValueError(f"vLLM counter series changed on engine {engine}")
        delta = {key: value - old[key] for key, value in new.items()}
        reasons = {key: value - first["finish_reasons"][key] for key, value in last["finish_reasons"].items()}
        if any(value < 0 for value in (*delta.values(), *reasons.values())):
            raise ValueError(f"vLLM counter reset on engine {engine}")
        if reasons.get("error", 0) or reasons.get("abort", 0):
            raise ValueError(f"Engine {engine} reported an errored or aborted request")
        if not delta["completed"] and any(delta[key] for key in ("queries", *TIERS) if key in delta):
            raise ValueError(f"Engine {engine} processed unaccounted cache traffic during the phase")
        per_engine[engine] = {**delta, "finish_reasons": reasons}

    completed = sum(delta["completed"] for delta in per_engine.values())
    if completed != expected:
        raise ValueError(f"Expected {expected} request completions in metrics, observed {completed}")
    return {
        "engine": next((engine for engine, delta in per_engine.items() if delta["completed"]), None)
        if completed == 1 else None,
        "completed": completed,
        "device_tokens": sum(delta["device"] for delta in per_engine.values()),
        "queries": sum(delta["queries"] for delta in per_engine.values())
        if all("queries" in delta for delta in per_engine.values()) else None,
        "per_engine": per_engine,
    }
