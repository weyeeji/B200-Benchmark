"""Read AIPerf 0.14 exports and summarize a closed-loop benchmark point."""

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

NA = "N/A"
LATENCIES = ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms")
ITL_STATISTIC = "percentiles_of_per_request_mean_token_intervals"

# Family names and units follow AIPerf 0.14's server-metrics reference.
CACHE_TIERS = ("gpu", "cpu", "ssd")
MONITORING_DEFAULTS = {
    "gpu_cache_hit_rate": [
        ("ratio", "vllm:prefix_cache_hits", "vllm:prefix_cache_queries", 1),
    ],
    "external_cache_hit_rate": [
        ("ratio", "vllm:external_prefix_cache_hits", "vllm:external_prefix_cache_queries", 1),
    ],
    "server_cached_input_tps": [
        ("counter_rate", "vllm:prompt_tokens_cached", None, 1),
        ("counter_rate", "sglang:cached_tokens", None, 1),
    ],
    "server_cached_input_tokens": [
        ("counter_delta", "vllm:prompt_tokens_cached", None, 1),
        ("counter_delta", "sglang:cached_tokens", None, 1),
    ],
    "server_uncached_input_tps": [
        ("counter_rate_difference", "vllm:prompt_tokens", "vllm:prompt_tokens_cached", 1),
        ("counter_rate_difference", "sglang:prompt_tokens", "sglang:cached_tokens", 1),
    ],
    "server_uncached_input_tokens": [
        ("counter_delta_difference", "vllm:prompt_tokens", "vllm:prompt_tokens_cached", 1),
        ("counter_delta_difference", "sglang:prompt_tokens", "sglang:cached_tokens", 1),
    ],
    "server_prefix_cache_hit_rate": [
        ("ratio", "vllm:prompt_tokens_cached", "vllm:prompt_tokens", 1),
        ("ratio", "sglang:cached_tokens", "sglang:prompt_tokens", 1),
        ("gauge_mean", "sglang:cache_hit_rate", None, 1),
    ],
    "gpu_cache_evicted_tokens_per_s": [
        ("counter_rate", "sglang:evicted_tokens", None, 1),
    ],
    "cpu_cache_load_latency_ms": [
        ("histogram_mean", "sglang:load_back_duration_seconds", None, 1000),
    ],
    "kv_transfer_latency_ms": [
        ("histogram_mean", "sglang:kv_transfer_latency_ms", None, 1),
        ("histogram_mean", "vllm:nixl_xfer_time_seconds", None, 1000),
        ("histogram_mean", "vllm:mooncake_xfer_time_seconds", None, 1000),
        ("histogram_mean", "trtllm_kv_transfer_latency_seconds", None, 1000),
    ],
    "kv_transfer_bandwidth_bytes_per_s": [
        ("histogram_mean", "sglang:kv_transfer_speed_gb_s", None, 1e9),
        ("histogram_ratio", "vllm:nixl_bytes_transferred", "vllm:nixl_xfer_time_seconds", 1),
        ("histogram_ratio", "vllm:mooncake_bytes_transferred", "vllm:mooncake_xfer_time_seconds", 1),
        ("histogram_ratio", "trtllm_kv_transfer_bytes", "trtllm_kv_transfer_latency_seconds", 1),
    ],
    "spec_acceptance_length": [
        ("ratio_plus_one", "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts", 1),
        ("gauge_mean", "sglang:spec_accept_length", None, 1),
        ("gauge_mean", "trtllm_spec_decode_acceptance_length", None, 1),
    ],
    "spec_acceptance_rate": [
        ("ratio", "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_draft_tokens", 1),
        ("ratio", "trtllm_spec_decode_num_accepted_tokens", "trtllm_spec_decode_num_draft_tokens", 1),
        ("gauge_mean", "sglang:spec_accept_rate", None, 1),
    ],
    "gpu_utilization_pct": [("gauge_mean", "DCGM_FI_DEV_GPU_UTIL", None, 1)],
    "gpu_memory_used_bytes": [("gauge_sum", "DCGM_FI_DEV_FB_USED", None, 2**20)],
    "cpu_memory_used_bytes": [
        ("gauge_difference", "node_memory_MemTotal_bytes", "node_memory_MemAvailable_bytes", 1),
    ],
    "network_rx_bytes_per_s": [("counter_rate", "node_network_receive_bytes", None, 1)],
    "network_tx_bytes_per_s": [("counter_rate", "node_network_transmit_bytes", None, 1)],
    "disk_read_bytes_per_s": [("counter_rate", "node_disk_read_bytes", None, 1)],
    "disk_write_bytes_per_s": [("counter_rate", "node_disk_written_bytes", None, 1)],
    "server_running_requests_mean": [("gauge_sum", "vllm:num_requests_running", None, 1)],
}
for _tier in CACHE_TIERS:
    for _suffix in ("hit_rate", "eviction_rate", "evicted_tokens_per_s", "load_latency_ms"):
        MONITORING_DEFAULTS.setdefault(f"{_tier}_cache_{_suffix}", [])

RULE_KINDS = {
    "ratio", "ratio_plus_one", "histogram_mean", "histogram_ratio",
    "histogram_sum_rate", "counter_rate", "counter_delta", "gauge_mean",
    "gauge_sum", "gauge_difference", "counter_rate_difference", "counter_delta_difference",
}


def jsonl(path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def profiling_records(directory):
    return [
        record for record in jsonl(Path(directory) / "profile_export.jsonl")
        if record.get("metadata", {}).get("benchmark_phase") == "profiling"
    ]


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def token_count(value):
    return value if type(value) is int and value >= 0 else None


def metric(record, name):
    return number((record.get("metrics") or {}).get(name, {}).get("value"))


def latency(record, name):
    entry = (record.get("metrics") or {}).get(name, {})
    value = number(entry.get("value"))
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1, "s": 1e3}.get(entry.get("unit"))
    return value * scale if value is not None and value >= 0 and scale is not None else None


def failed(record):
    return bool(record.get("error") or record["metadata"].get("was_cancelled"))


def timed_out(record):
    error = record.get("error") or {}
    return "timeout" in str(error.get("type", "")).lower() or error.get("code") in (408, 504)


def usage_count(record, field):
    return token_count(metric(record, f"usage_{field}_tokens"))


def record_counts(artifact_dir, *, isl, osl):
    """Use only server Usage for observed input/output lengths and EOS evidence."""
    return counts_from_records(profiling_records(artifact_dir), isl, osl)


def counts_from_records(records, isl, osl):
    success = [req for req in records if not failed(req)]
    observed_inputs = [usage_count(req, "prompt") for req in success]
    observed_outputs = [usage_count(req, "completion") for req in success]
    isl_count = sum(value is not None for value in observed_inputs)
    osl_count = sum(value is not None for value in observed_outputs)
    osl_matching = sum(value == osl for value in observed_outputs)
    return {
        "attempted_requests": len(records),
        "completed_requests": len(success),
        "failed_requests": len(records) - len(success),
        "cancelled_requests": sum(bool(req["metadata"].get("was_cancelled")) for req in records),
        "server_isl_requests": isl_count,
        "server_osl_requests": osl_count,
        "isl_matching_requests": sum(value == isl for value in observed_inputs),
        "osl_matching_requests": osl_matching,
        "ignore_eos_status": (
            "length_mismatch" if osl_count > osl_matching else
            "target_met" if success and osl_count == len(success) else "unverified"
        ),
    }


@lru_cache(maxsize=4)
def output_tokenizer(path):
    from tokenizers import Tokenizer

    local_file = Path(path) / "tokenizer.json"
    if local_file.is_file():
        return Tokenizer.from_file(str(local_file))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True).backend_tokenizer


def response_messages(record):
    for response in record.get("responses", []):
        packets = response.get("packets")
        data = response.get("text") if packets is None else "\n".join(
            packet.get("value") or "" for packet in packets if packet.get("name") == "data"
        )
        if data and data != "[DONE]":
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def fill_from_raw(artifact_dir, requests, tokenizer_path, input_token_count):
    by_id = {req["metadata"]["x_request_id"]: req for req in requests
             if req["metadata"].get("x_request_id")}
    finish_reasons = Counter()
    for raw in jsonl(artifact_dir / "profile_export_raw.jsonl"):
        req = by_id.get(raw["metadata"].get("x_request_id"))
        if req is None:
            continue
        payload = raw.get("payload")
        if req["input"] is None and input_token_count and payload:
            req["input"] = token_count(input_token_count(payload))
            if req["input"] is not None:
                req["input_source"] = "client_tokenizer"
        need_text = req["output"] is None and bool(tokenizer_path)
        content, reasoning = [], []
        finish_reason = None
        for message in response_messages(raw):
            for choice in message.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                if need_text:
                    delta = choice.get("delta") or choice.get("message") or {}
                    text = delta.get("content")
                    thought = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(text, str) and text:
                        content.append(text)
                    if isinstance(thought, str) and thought:
                        reasoning.append(thought)
        if finish_reason:
            finish_reasons[finish_reason] += 1
        if need_text and (content or reasoning):
            tokenizer = output_tokenizer(str(tokenizer_path))
            reasoning_count, content_count = (
                len(tokenizer.encode("".join(parts), add_special_tokens=False).ids) if parts else 0
                for parts in (reasoning, content)
            )
            if reasoning and req["reasoning"] is None:
                req["reasoning"] = reasoning_count
            req["output"] = reasoning_count + content_count
            req["output_source"] = "client_tokenizer"
    return finish_reasons


def percentile(ordered, q):
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def stats(values, name):
    ordered = sorted(value for value in values if value is not None)
    calculations = [("mean", statistics.fmean(ordered) if ordered else NA)]
    calculations.extend(
        (f"p{p}", percentile(ordered, p / 100) if ordered else NA)
        for p in (50, 90, 95, 99)
    )
    base, unit = name.rsplit("_", 1)
    calculations.append(("max", ordered[-1] if ordered else NA))
    return {f"{base}_{stat}_{unit}": value for stat, value in calculations}


def complete_sum(values):
    values = list(values)
    return sum(values) if values and all(value is not None for value in values) else None


def ratio(a, b, scale=1):
    return scale * a / b if a is not None and b is not None and b > 0 else NA


def request_record(record):
    metadata = record["metadata"]
    start = metadata.get("request_start_ns")
    ttft = latency(record, "time_to_first_token")
    decode = latency(record, "full_decode_duration")
    full_ms = ttft + decode if ttft is not None and decode is not None else None
    end = metadata.get("request_end_ns")
    timing = "last_wire_event"
    if type(start) is int and start > 0 and full_ms is not None:
        http_end = start + round(full_ms * 1e6)
        if type(end) is not int or http_end >= end - 1:
            end, timing = max(http_end, end or 0), "http_completion"
    cancellation = metadata.get("cancellation_time_ns")
    if type(cancellation) is int and (type(end) is not int or cancellation > end):
        end, timing = cancellation, "cancellation"
    if type(start) is not int or start <= 0 or type(end) is not int or end < start:
        start, end, timing = None, None, "unavailable"
    e2el = (end - start) / 1e6 if start is not None else full_ms
    return {
        "start": start, "end": end,
        "failed": failed(record),
        "timing": timing,
        "timeout": timed_out(record),
        "input": usage_count(record, "prompt"),
        "output": usage_count(record, "completion"),
        "reasoning": usage_count(record, "reasoning"),
        "input_source": "server_usage" if usage_count(record, "prompt") is not None else None,
        "output_source": "server_usage" if usage_count(record, "completion") is not None else None,
        "cached": usage_count(record, "prompt_cache_read"),
        "uncached": usage_count(record, "prompt_cache_miss"),
        "ttft_ms": ttft,
        "tpot_ms": latency(record, "full_response_inter_token_latency"),
        "itl_ms": latency(record, "inter_token_latency"),
        "user_tps": metric(record, "output_token_throughput_per_user"),
        "e2el_ms": e2el,
        "itl_decode_ms": latency(record, "decode_duration"),
        "metadata": metadata,
        "spec_accepted": metric(record, "spec_decode_accepted_draft_tokens"),
        "spec_drafted": metric(record, "spec_decode_draft_tokens"),
        "spec_steps": metric(record, "spec_decode_steps"),
    }


def steady_window(requests, concurrency):
    events = Counter()
    for req in requests:
        events[req["start"]] += 1
        events[req["end"]] -= 1
    active = 0
    steady_start = None
    last_send = max((req["start"] for req in requests), default=None)
    last_time = None
    integral = 0
    target_duration = 0
    peak = 0
    for timestamp, change in sorted(events.items()):
        if steady_start is not None and last_time < last_send:
            elapsed = min(timestamp, last_send) - last_time
            integral += active * elapsed
            if active == concurrency:
                target_duration += elapsed
        active += change
        if active >= concurrency and steady_start is None:
            steady_start = timestamp
        peak = max(peak, active)
        last_time = timestamp
    if steady_start is None or last_send <= steady_start:
        return None, None, NA, peak if requests else NA, NA
    duration = last_send - steady_start
    return steady_start, last_send, integral / duration, peak, target_duration / duration


def endpoint_key(url):
    """AIPerf appends /metrics to scrape URLs that have no path."""
    parsed = urlsplit(url)
    return urlunsplit(parsed._replace(path=parsed.path if parsed.path not in ("", "/") else "/metrics"))


def select_samples(artifact_dir, start, end, rule_index, urls=()):
    """Retain the known series roster and complete observations inside the steady window."""
    series = defaultdict(lambda: defaultdict(list))
    if start is None or end is None:
        return series
    urls = {endpoint_key(url) for url in urls}
    path = artifact_dir / "profile_export_server_metrics.jsonl"
    if not path.is_file():
        path = artifact_dir / "server_metrics_export.jsonl"
    for snapshot in jsonl(path):
        endpoint = snapshot.get("endpoint_url")
        if not endpoint:
            continue
        endpoint = endpoint_key(endpoint)
        if urls and endpoint not in urls:
            continue
        timestamp = snapshot.get("timestamp_ns")
        sent = snapshot.get("request_sent_ns")
        if sent is None and type(timestamp) is int:
            latency = snapshot.get("endpoint_latency_ns")
            sent = timestamp - latency if type(latency) is int and latency >= 0 else timestamp
        in_window = type(timestamp) is int and type(sent) is int and start <= sent <= timestamp <= end
        snapshot_metrics = snapshot.get("metrics") or {}
        for family in snapshot_metrics.keys() & rule_index:
            for sample in snapshot_metrics[family]:
                labels = tuple(sorted((sample.get("labels") or {}).items()))
                observations = series[family][(endpoint, labels)]
                if in_window:
                    observations.append((sent, sample))
    return series


def series_value(observations, field, mode):
    points = {}
    for sent, sample in observations:
        value = number(sample.get(field))
        if value is None or (sent in points and points[sent] != value):
            return None
        points[sent] = value
    times = tuple(sorted(points))
    if not times:
        return None
    values = [points[sent] for sent in times]
    if mode == "gauge":
        return statistics.fmean(values), times
    # Without a reset timestamp or a resolvable order, a decrease gives no exact delta.
    if len(times) < 2 or values[0] < 0 or any(b < a for a, b in zip(values, values[1:])):
        return None
    delta = values[-1] - values[0]
    return (delta / ((times[-1] - times[0]) / 1e9) if mode == "rate" else delta), times


def timeseries_measure(series, family, field, mode, labels, label_regex, endpoints, ignore):
    values = {}
    samples_by_endpoint = {}
    for (endpoint, series_labels), observations in series.get(family, {}).items():
        if endpoints and endpoint not in endpoints:
            continue
        label_map = dict(series_labels)
        if any(label_map.get(key) != value for key, value in labels.items()):
            continue
        if any(not re.fullmatch(pattern, label_map.get(key, "")) for key, pattern in label_regex.items()):
            continue
        measure = series_value(observations, field, mode)
        if measure is None:
            return {}
        value, times = measure
        if endpoint in samples_by_endpoint and samples_by_endpoint[endpoint] != times:
            return {}
        samples_by_endpoint[endpoint] = times
        key = endpoint, tuple((key, value) for key, value in series_labels if key not in ignore)
        previous, _ = values.get(key, (0, times))
        values[key] = previous + value, times
    if endpoints and endpoints != samples_by_endpoint.keys():
        return {}
    if not comparable_scrape_windows(samples_by_endpoint.values()):
        return {}
    return values


def comparable_scrape_windows(windows):
    """Allow polling skew when combining distinct endpoints on one steady window."""
    if len(windows) < 2:
        return True
    start = min(times[0] for times in windows)
    end = max(times[-1] for times in windows)
    for times in windows:
        tolerance = statistics.median(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else 0
        if times[0] - start > tolerance or end - times[-1] > tolerance:
            return False
    return True


def evaluate_rule(series, rule):
    kind, family = rule["kind"], rule["metric"]
    denominator = rule.get("denominator")
    labels = rule.get("labels", {})
    label_regex = rule.get("label_regex", {})
    endpoints = {endpoint_key(url) for url in rule.get("endpoints", [])}
    ignore = rule.get("ignore_labels", [])
    scale = rule.get("scale", 1)

    def measure(name, field, mode, denominator_sample=False):
        selected_labels = rule.get("denominator_labels", labels) if denominator_sample else labels
        return timeseries_measure(series, name, field, mode, selected_labels, label_regex, endpoints, ignore)

    def matched(a, b):
        if a and a.keys() == b.keys() and all(a[key][1] == b[key][1] for key in a):
            return sum(item[0] for item in a.values()), sum(item[0] for item in b.values())
        return None, None

    if kind in ("ratio", "ratio_plus_one", "histogram_ratio", "histogram_mean"):
        field_a = "sum" if kind.startswith("histogram") else "value"
        field_b = "count" if kind == "histogram_mean" else field_a
        a, b = matched(measure(family, field_a, "delta"),
                       measure(denominator or family, field_b, "delta", True))
        result = ratio(a, b)
        if result != NA:
            if kind == "ratio" and not 0 <= result <= 1:
                return NA
            return (result + (kind == "ratio_plus_one")) * scale
        return NA
    if kind in ("gauge_difference", "counter_rate_difference", "counter_delta_difference"):
        mode = {"gauge_difference": "gauge", "counter_rate_difference": "rate",
                "counter_delta_difference": "delta"}[kind]
        a, b = matched(measure(family, "value", mode), measure(denominator, "value", mode, True))
        return (a - b) * scale if a is not None and b is not None and a >= b else NA
    if kind in ("gauge_sum", "gauge_mean"):
        values = [item[0] for item in measure(family, "value", "gauge").values()]
        if not values:
            return NA
        return (statistics.fmean(values) if kind == "gauge_mean" else sum(values)) * scale
    field = "sum" if kind == "histogram_sum_rate" else "value"
    values = measure(family, field, "delta" if kind == "counter_delta" else "rate")
    return sum(item[0] for item in values.values()) * scale if values else NA


def server_summary(artifact_dir, start, end, config):
    rules = {
        name: [
            {"kind": kind, "metric": family, "denominator": divisor, "scale": scale}
            for kind, family, divisor, scale in candidates
        ]
        for name, candidates in MONITORING_DEFAULTS.items()
    }
    for tier, source in zip(CACHE_TIERS, ("device", "host", "disk")):
        rules[f"{tier}_cache_hit_rate"].insert(0, {
            "kind": "ratio", "metric": "vllm:prompt_tokens_cached_by_source",
            "denominator": "vllm:prompt_tokens", "labels": {"source": source},
            "denominator_labels": {}, "ignore_labels": ["source"],
        })
    for name, candidates in rules.items():
        if name.startswith("network_"):
            candidates[0]["label_regex"] = {"device": r"(eth|en|ib).*"}
        if name.startswith("disk_"):
            candidates[0]["label_regex"] = {"device": r"(nvme[0-9]+n[0-9]+|[sv]d[a-z]+|xvd[a-z]+|mmcblk[0-9]+)"}
    for name, override in config.get("rules", {}).items():
        rules[name] = override if isinstance(override, list) else [override]
    for candidates in rules.values():
        for rule in candidates:
            if rule["kind"] not in RULE_KINDS:
                raise ValueError(f"Unknown Prometheus rule kind {rule['kind']!r}")
    family_index = {
        rule[key] for candidates in rules.values() for rule in candidates
        for key in ("metric", "denominator") if rule.get(key)
    }
    series = select_samples(artifact_dir, start, end, family_index, config.get("urls", []))
    result = {}
    for name, candidates in rules.items():
        result[name] = NA
        for rule in candidates:
            value = evaluate_rule(series, rule)
            if value != NA:
                result[name] = value
                result[f"{name}_source"] = rule["metric"]
                break
    return result


def summarize(artifact_dir, *, concurrency, gpu_count, isl, osl, sla=None,
              prometheus=None, tokenizer_path=None, input_token_count=None):
    """Summarize successes and errors by completion during offered steady load."""
    artifact_dir = Path(artifact_dir)
    records = profiling_records(artifact_dir)
    parsed = [request_record(record) for record in records]
    all_requests = [req for req in parsed if req["start"] is not None]
    success = [req for req in parsed if not req["failed"]]
    finish_reasons = fill_from_raw(artifact_dir, success, tokenizer_path, input_token_count)
    for req in success:
        total = req["input"] if req["input_source"] == "server_usage" else None
        cached, uncached = req["cached"], req["uncached"]
        if (
            total is None or (cached is not None and cached > total)
            or (uncached is not None and uncached > total)
            or (cached is not None and uncached is not None and cached + uncached != total)
        ):
            req["cached"], req["uncached"] = None, None
        elif cached is not None:
            req["uncached"] = total - cached
        elif uncached is not None:
            req["cached"] = total - uncached
        # With no server Usage, approximate from tokenizer counts across joined text.
        if req["output_source"] == "client_tokenizer" and req["output"] > 1:
            if req["itl_decode_ms"] is not None:
                req["itl_ms"] = req["itl_decode_ms"] / (req["output"] - 1)
            if req["ttft_ms"] is not None and req["e2el_ms"] is not None and req["e2el_ms"] >= req["ttft_ms"]:
                req["tpot_ms"] = (req["e2el_ms"] - req["ttft_ms"]) / (req["output"] - 1)

    untimed = len(records) - len(all_requests)
    steady_start, steady_end, active_mean, active_peak, target_fraction = (
        steady_window(all_requests, concurrency) if not untimed else (None, None, NA, NA, NA)
    )
    steady_duration = (steady_end - steady_start) / 1e9 if steady_start is not None else None
    startup = [req for req in all_requests if steady_duration and req["end"] <= steady_start]
    drain = [req for req in all_requests if steady_duration and req["end"] > steady_end]
    finalized = [req for req in all_requests if steady_start is not None
                 and steady_start < req["end"] <= steady_end]
    done = [req for req in finalized if not req["failed"]]
    counts = counts_from_records(records, isl, osl)
    result = {
        **counts,
        "target_isl_tokens": isl,
        "target_osl_tokens": osl,
        "timed_requests": len(all_requests),
        "untimed_requests": untimed,
        "http_completion_timestamp_requests": sum(req["timing"] == "http_completion" for req in all_requests),
        "steady_start_ns": steady_start if steady_start is not None else NA,
        "steady_end_ns": steady_end if steady_end is not None else NA,
        "steady_seconds": steady_duration if steady_duration is not None else NA,
        "steady_end_source": "last_observed_request_start" if steady_duration else NA,
        "startup_attempts": len(startup) if steady_duration else NA,
        "startup_completed_requests": sum(not req["failed"] for req in startup) if steady_duration else NA,
        "steady_completed_requests": len(done) if steady_duration else NA,
        "steady_attempts": len(finalized) if steady_duration else NA,
        "drain_attempts": len(drain) if steady_duration else NA,
        "drain_completed_requests": sum(not req["failed"] for req in drain) if steady_duration else NA,
        "drain_failed_requests": sum(req["failed"] for req in drain) if steady_duration else NA,
        "actual_concurrency_mean": active_mean,
        "steady_concurrency_at_target_fraction": target_fraction,
        "actual_concurrency_peak": active_peak,
        "rps": ratio(len(done), steady_duration),
        "request_success_rate": ratio(len(done), len(finalized)),
        "timeout_rate": ratio(sum(req["timeout"] for req in finalized), len(finalized)),
        "error_rate": ratio(sum(req["failed"] for req in finalized), len(finalized)),
        "whole_run_request_success_rate": ratio(counts["completed_requests"], len(records)),
        "whole_run_timeout_rate": ratio(sum(timed_out(record) for record in records), len(records)),
        "whole_run_error_rate": ratio(counts["failed_requests"], len(records)),
        "finish_reasons": json.dumps(dict(finish_reasons), sort_keys=True) if finish_reasons else NA,
        "itl_statistic": ITL_STATISTIC,
    }

    for field, expected in (("input", isl), ("output", osl)):
        actual = [req[field] for req in success if req[field] is not None]
        kind = "isl" if field == "input" else "osl"
        result[f"actual_{kind}_mean_tokens"] = statistics.fmean(actual) if actual else NA
        result[f"actual_{kind}_min_tokens"] = min(actual) if actual else NA
        result[f"actual_{kind}_max_tokens"] = max(actual) if actual else NA
        result[f"{kind}_mean_delta_tokens"] = statistics.fmean(x - expected for x in actual) if actual else NA
        result[f"{kind}_mismatched_tokens_requests"] = sum(x != expected for x in actual) if actual else NA
        sources = sorted({req[f"{field}_source"] for req in success if req[f"{field}_source"]})
        result[f"{kind}_source"] = "+".join(sources) if sources else NA

    actual_reasoning = [req["reasoning"] for req in success if req["reasoning"] is not None]
    result["observed_reasoning_lengths"] = len(actual_reasoning)
    result["actual_reasoning_mean_tokens"] = statistics.fmean(actual_reasoning) if actual_reasoning else NA
    for field in LATENCIES:
        label = field.removesuffix("_ms")
        steady_values = [req[field] for req in done if req[field] is not None]
        formal_values = [req[field] for req in success if req[field] is not None]
        result.update(stats(steady_values, field))
        result.update(stats(formal_values, f"formal_{field}"))
        result[f"steady_{label}_samples"] = len(steady_values)
        result[f"formal_{label}_samples"] = len(formal_values)
    input_tokens = complete_sum(req["input"] for req in done)
    output_tokens = complete_sum(req["output"] for req in done)
    cached_tokens = complete_sum(req["cached"] for req in done)
    uncached_tokens = complete_sum(req["uncached"] for req in done)
    total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    for kind, tokens in (("input", input_tokens), ("output", output_tokens), ("total", total_tokens),
                         ("cached_input", cached_tokens), ("uncached_input", uncached_tokens)):
        result[f"{kind}_tokens"] = tokens if tokens is not None else NA
        result[f"{kind}_tps"] = ratio(tokens, steady_duration)
        result[f"{kind}_tpm"] = ratio(tokens, steady_duration, 60)
    result["actual_cache_hit_rate"] = ratio(cached_tokens, input_tokens)
    result["output_tps_per_gpu"] = ratio(number(result["output_tps"]), gpu_count)
    result["total_tps_per_gpu"] = ratio(number(result["total_tps"]), gpu_count)

    user_tps = [req["user_tps"] or 1000 / req["itl_ms"] for req in done
                if req["itl_ms"] is not None and req["itl_ms"] > 0]
    result["tps_per_user"] = statistics.fmean(user_tps) if user_tps else NA

    sla = sla or {}
    if any(name not in LATENCIES or number(limit) is None or limit <= 0 for name, limit in sla.items()):
        raise ValueError(f"SLA threshold keys must be in {LATENCIES}, with positive numeric values")
    good = None
    if sla and finalized and all(req[name] is not None for req in done for name in sla):
        good = sum(all(req[name] <= limit for name, limit in sla.items()) for req in done)
    result["goodput_rps"] = ratio(good, steady_duration)
    result["sla_pass_rate"] = ratio(good, len(finalized))
    result.update(server_summary(artifact_dir, steady_start, steady_end, prometheus or {}))
    for name in ("cached_input", "uncached_input"):
        result[f"{name}_source"] = "server_usage" if result[f"{name}_tps"] != NA else NA
    result["actual_cache_hit_rate_source"] = "server_usage" if result["actual_cache_hit_rate"] != NA else NA

    accepted = complete_sum(req["spec_accepted"] for req in done)
    steps = complete_sum(req["spec_steps"] for req in done)
    draft = complete_sum(req["spec_drafted"] for req in done)
    acceptance = ratio(accepted, steps)
    if acceptance != NA:
        result["spec_acceptance_length"] = 1 + acceptance
        result["spec_acceptance_length_source"] = "aiperf_request_records"
    acceptance_rate = ratio(accepted, draft)
    if acceptance_rate != NA and 0 <= acceptance_rate <= 1:
        result["spec_acceptance_rate"] = acceptance_rate
        result["spec_acceptance_rate_source"] = "aiperf_request_records"
    return result


def write_summaries(output_dir, rows):
    """Write tabular and JSON representations of the same rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = [{column: row.get(column, NA) for column in columns} for row in rows]
    for name, write in (
        ("summary.json", lambda stream: json.dump(table, stream, ensure_ascii=False, indent=2, allow_nan=False)),
        ("summary.csv", lambda stream: write_csv(stream, table, columns)),
    ):
        temporary = output_dir / (name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            write(stream)
        temporary.replace(output_dir / name)


def write_csv(stream, rows, columns):
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
