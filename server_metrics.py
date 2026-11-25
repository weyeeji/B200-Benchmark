"""Optional OpenAI-compatible backend counter and gauge collection."""

import math
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families


def capture(url, path=None):
    try:
        with urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8")
        if path is not None:
            path.write_text(text, encoding="utf-8")
        return {
            (sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in text_string_to_metric_families(text)
            for sample in family.samples
        }
    except (OSError, ValueError) as exc:
        if path is not None:
            path.with_suffix(".error.txt").write_text(f"{exc}\n", encoding="utf-8")
        return None


def series(snapshot, metric, model):
    found = {}
    for (name, labels), value in snapshot.items():
        if name not in (metric, f"{metric}_total"):
            continue
        tags = dict(labels)
        if tags.get("model_name", tags.get("model")) in (None, model):
            found[labels] = value
    return found


def delta(before, after, metric, model):
    start = series(before, metric, model)
    end = series(after, metric, model)
    if not start or start.keys() != end.keys():
        return None
    if any(not math.isfinite(value) or value < 0 for value in (*start.values(), *end.values())):
        return None
    if any(end[key] < value for key, value in start.items()):
        return None
    return sum(end[key] - value for key, value in start.items())


def ratio(numerator, denominator):
    if numerator is None or denominator is None or not 0 <= numerator <= denominator or denominator == 0:
        return None
    return numerator / denominator


def window_metrics(before, after, model):
    if before is None or after is None:
        return {}
    values = {}
    for prefix, cached in (("vllm", "prompt_tokens_cached"), ("sglang", "cached_tokens")):
        prompt_tokens = delta(before, after, f"{prefix}:prompt_tokens", model)
        hit = ratio(delta(before, after, f"{prefix}:{cached}", model), prompt_tokens)
        if hit is not None:
            values["kv_hit_percent"] = 100 * hit
            break
    accepted = delta(before, after, "vllm:spec_decode_num_accepted_tokens", model)
    proposed = delta(before, after, "vllm:spec_decode_num_draft_tokens", model)
    steps = delta(before, after, "vllm:spec_decode_num_drafts", model)
    acceptance = ratio(accepted, proposed)
    if accepted is not None and steps and (proposed is None or accepted <= proposed):
        values["spec_acceptance_length"] = 1 + accepted / steps
    if acceptance is not None:
        values["spec_acceptance_percent"] = 100 * acceptance
    return values


def request_count(before, after, model):
    if before is None or after is None:
        return None
    for name in ("vllm:e2e_request_latency_seconds_count", "vllm:request_success"):
        count = delta(before, after, name, model)
        if count is not None:
            return count
    return None


def compact_sample(snapshot):
    names = {
        "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens", "vllm:prompt_tokens_cached", "vllm:generation_tokens",
        "vllm:request_success", "vllm:request_queue_time_seconds_count",
        "vllm:request_queue_time_seconds_sum", "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens",
        "sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage",
        "sglang:prompt_tokens", "sglang:cached_tokens", "sglang:generation_tokens",
    }
    return [
        {"metric": name, "labels": dict(labels), "value": value}
        for (name, labels), value in sorted(snapshot.items())
        if name.removesuffix("_total") in names and math.isfinite(value)
    ]
