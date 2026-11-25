"""Bin AIPerf profiling records on a shared, elapsed-time axis."""

from collections import Counter
import math


def _positive_ns(value):
    return value if type(value) is int and value > 0 else None


def _linear(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _tpm(samples, attribute, seconds):
    values = [getattr(sample, attribute) for sample in samples]
    return 60 * sum(values) / seconds if all(value is not None for value in values) else None


def build_timeline(export, profile_start_ns=None, send_end_ns=None, requests_end_ns=None):
    """Return interval-weighted in-flight counts and completed-request metrics by time bin."""
    intervals = [pair for pair in export.intervals if pair[1] > pair[0]]
    if not intervals:
        return None

    start = min(a for a, _ in intervals)
    stop = max(b for _, b in intervals)
    profile_start_ns = _positive_ns(profile_start_ns)
    send_end_ns = _positive_ns(send_end_ns)
    requests_end_ns = _positive_ns(requests_end_ns)
    origin = min(profile_start_ns, start) if profile_start_ns is not None else start
    end = max(stop, send_end_ns or 0, requests_end_ns or 0)
    span = (end - origin) / 1e9

    # Aim for at most about 60 bins, using whole seconds on a common grid.
    width_seconds = max(1, math.ceil(span / 60))
    width_ns = width_seconds * 1_000_000_000
    bin_count = (end - origin + width_ns - 1) // width_ns
    completed = [[] for _ in range(bin_count)]
    for sample in export.samples:
        if sample.wire is not None:
            index = min(bin_count - 1, (sample.wire[1] - origin) // width_ns)
            if index >= 0:
                completed[index].append(sample)

    changes = Counter()
    for left, right in intervals:
        changes[left] += 1
        changes[right] -= 1
    events = sorted((t, delta) for t, delta in changes.items() if delta)
    event_index = 0
    current = 0
    bins = []
    entire_area = 0
    maximum = 0
    for index, samples in enumerate(completed):
        left = origin + index * width_ns
        right = min(end, left + width_ns)
        while event_index < len(events) and events[event_index][0] <= left:
            current += events[event_index][1]
            event_index += 1
        peak = current
        area = 0
        previous = left
        while event_index < len(events) and events[event_index][0] < right:
            time, change = events[event_index]
            area += (time - previous) * current
            current += change
            peak = max(peak, current)
            previous = time
            event_index += 1
        area += (right - previous) * current
        entire_area += area
        maximum = max(maximum, peak)
        seconds = (right - left) / 1e9
        ttft = [sample.ttft for sample in samples if sample.ttft is not None]
        tpot = [sample.tpot for sample in samples if sample.tpot is not None]
        bins.append({
            "seconds": ((left - origin) + (right - origin)) / 2e9,
            "in_flight_mean": area / (right - left),
            "in_flight_peak": peak,
            "input_tpm": _tpm(samples, "input_tokens", seconds),
            "output_tpm": _tpm(samples, "output_tokens", seconds),
            "ttft_p50_ms": _linear(ttft, 0.50),
            "ttft_p99_ms": _linear(ttft, 0.99),
            "tpot_p50_ms": _linear(tpot, 0.50),
            "tpot_p99_ms": _linear(tpot, 0.99),
        })
    cutoff = (send_end_ns - origin) / 1e9 if send_end_ns is not None else None
    return {
        "rows": bins, "span": span, "cutoff": cutoff, "bin_seconds": width_seconds,
        "mean": entire_area / (end - origin), "peak": maximum,
        "intervals": len(intervals),
        "complete": export.malformed == 0 and len(intervals) == len(export.samples) + export.failed,
        "origin": "正式阶段" if origin == profile_start_ns else "首个可用请求",
    }
