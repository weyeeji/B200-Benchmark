"""Read optional OpenAI-compatible backend counters around a benchmark point."""

import math
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families


def capture(url, path):
    try:
        with urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
        return {
            (sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in text_string_to_metric_families(text)
            for sample in family.samples
        }
    except (OSError, ValueError) as exc:
        path.with_suffix(".error.txt").write_text(f"{exc}\n", encoding="utf-8")
        return None


def series(snapshot, metric, model):
    found = {}
    for (name, labels), value in snapshot.items():
        if name not in (metric, f"{metric}_total"):
            continue
        tags = dict(labels)
        series_model = tags.get("model_name", tags.get("model"))
        if series_model in (None, model):
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


def enrich(row, before, after, model, *, includes_warmup):
    if before is None or after is None:
        return
    scope = "prometheus_with_warmup" if includes_warmup else "prometheus_formal"

    if row.get("kv_hit_percent") is None:
        for prefix, cached in (("vllm", "prompt_tokens_cached"), ("sglang", "cached_tokens")):
            hit = ratio(
                delta(before, after, f"{prefix}:{cached}", model),
                delta(before, after, f"{prefix}:prompt_tokens", model),
            )
            if hit is not None:
                row["kv_hit_percent"] = hit * 100
                row["kv_source"] = scope
                break

    accepted = delta(before, after, "vllm:spec_decode_num_accepted_tokens", model)
    proposed = delta(before, after, "vllm:spec_decode_num_draft_tokens", model)
    steps = delta(before, after, "vllm:spec_decode_num_drafts", model)
    acceptance = ratio(accepted, proposed)
    supplemented = []
    if row.get("spec_acceptance_length") is None and accepted is not None and steps:
        row["spec_acceptance_length"] = 1 + accepted / steps
        supplemented.append("接受长度")
    if row.get("spec_acceptance_percent") is None and acceptance is not None:
        row["spec_acceptance_percent"] = acceptance * 100
        supplemented.append("接受率")
    if supplemented:
        source = f"{scope} ({'、'.join(supplemented)})"
        row["spec_source"] = " + ".join(filter(None, [row.get("spec_source"), source]))
