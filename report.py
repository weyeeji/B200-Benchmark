"""Summarize profiling requests and render a standalone benchmark report."""

import csv
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from html import escape
import json
import math
from pathlib import Path
from urllib.parse import quote, urlsplit

from report_timeline import build_timeline


PERCENTILES = (50, 90, 99)
SLA_MS = {
    "ttft_p50_ms": 5_000,
    "ttft_p90_ms": 10_000,
    "ttft_p99_ms": 30_000,
    "tpot_p50_ms": 30,
    "tpot_p90_ms": 50,
    "tpot_p99_ms": 80,
}
TIME_TO_MS = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1, "s": 1e3, "sec": 1e3}
EXPERIMENT_NAMES = {"P01": "重 Prefill", "P02": "重 Decode", "P03": "AgentX 真实流量"}
SLA_NAMES = {"pass": "达标", "fail": "超出 SLA", "unknown": "未知"}
VALIDITY_NAMES = {"valid": "有效", "invalid": "无效", "incomplete": "不完整"}
COLUMNS = (
    ("experiment", "实验"), ("concurrency", "配置并发或轨迹 lane"),
    ("repeat", "重复"), ("repeat_count", "计划重复数"), ("status", "综合状态"),
    ("sla_status", "实测 SLA"), ("validity", "有效性"),
    ("ttft_p50_ms", "TTFT P50 · ms"), ("ttft_p90_ms", "TTFT P90 · ms"),
    ("ttft_p99_ms", "TTFT P99 · ms"), ("tpot_p50_ms", "TPOT P50 · ms"),
    ("tpot_p90_ms", "TPOT P90 · ms"), ("tpot_p99_ms", "TPOT P99 · ms"),
    ("input_tpm", "输入 TPM"), ("output_tpm", "输出 TPM"), ("total_tpm", "总 TPM"),
    ("kv_hit_percent", "正式阶段 KVCache 命中 · %"),
    ("spec_acceptance_length", "正式阶段接受长度"),
    ("spec_acceptance_percent", "正式阶段草稿接受率 · %"),
    ("window_kv_hit_percent", "正式窗口 KVCache 命中 · %"),
    ("window_spec_acceptance_length", "正式窗口接受长度"),
    ("window_spec_acceptance_percent", "正式窗口草稿接受率 · %"),
    ("whole_run_kv_hit_percent", "整轮 KVCache 命中 · %"),
    ("whole_run_spec_acceptance_length", "整轮接受长度"),
    ("whole_run_spec_acceptance_percent", "整轮草稿接受率 · %"),
    ("attempted_requests", "正式请求"),
    ("completed_requests", "成功"), ("failed_requests", "失败"),
    ("cancelled_requests", "取消"), ("context_skips", "上下文跳过"),
    ("ttft_samples", "TTFT 样本"), ("tpot_samples", "TPOT 样本"),
    ("input_usage_samples", "输入计量样本"), ("output_usage_samples", "输出计量样本"),
    ("kv_usage_samples", "缓存计量样本"), ("spec_samples", "推测解码样本"),
    ("duration_seconds", "正式测量 · 秒"), ("duration_source", "正式测量来源"),
    ("kv_source", "缓存数据来源"),
    ("spec_source", "推测数据来源"), ("window_source", "正式窗口来源"),
    ("whole_run_source", "整轮来源"), ("window_scope", "正式窗口范围"),
    ("whole_run_scope", "整轮范围"), ("submission_valid", "AIPerf 提交有效"),
    ("submission_notes", "AIPerf 提交备注"),
    ("prepare_seconds", "准备 · 秒"), ("warmup_seconds", "预热 · 秒"),
    ("send_seconds", "正式发包 · 秒"), ("drain_seconds", "实际收尾 · 秒"),
    ("configured_send_seconds", "配置发包 · 秒"), ("configured_drain_seconds", "配置收尾 · 秒"),
    ("in_flight_at_cutoff", "截止时在途"), ("drain_last_response_seconds", "最后响应距截止 · 秒"),
    ("profile_start_ns", "正式开始时间戳 · ns"), ("send_end_ns", "发包截止时间戳 · ns"),
    ("requests_end_ns", "请求结束时间戳 · ns"), ("dataset_sha256", "正式数据 SHA-256"),
    ("warmup_dataset_sha256", "预热数据 SHA-256"),
    ("grace_period_timeout_triggered", "触发收尾超时"),
    ("final_requests_completed", "阶段日志完成数"),
    ("final_requests_cancelled", "阶段日志取消数"), ("final_request_errors", "阶段日志错误数"),
    ("exit_code", "进程退出码"), ("failure_reason", "执行异常说明"),
    ("validity_reason", "有效性说明"), ("detail", "判定说明"),
    ("point_dir", "档位目录"), ("phase_summary", "阶段摘要"),
    ("artifact_dir", "原始导出目录"), ("log_dir", "日志目录"),
)


@dataclass(frozen=True, slots=True)
class RequestSample:
    ttft: float | None
    tpot: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    spec_steps: int | None
    spec_accepted: int | None
    spec_drafts: int | None

    wire: tuple[int, int] | None


@dataclass(slots=True)
class ProfilingExport:
    samples: list[RequestSample] = field(default_factory=list)
    intervals: list[tuple[int, int]] = field(default_factory=list)
    failed: int = 0
    cancelled: int = 0
    skipped: int = 0
    malformed: int = 0
    first_malformed_line: int | None = None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _count(value):
    return int(value) if _number(value) is not None and int(value) == value else None


def _metric(metrics, name, field="value"):
    entry = metrics.get(name)
    return _number(entry.get(field)) if isinstance(entry, dict) else None


def _milliseconds(metrics, name, field="value"):
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        return None
    value = _number(entry.get(field))
    factor = TIME_TO_MS.get(entry.get("unit"))
    return value * factor if value is not None and factor is not None else None


def _wire_interval(metadata, full_ms=None):
    start = metadata.get("request_start_ns")
    if type(start) is not int or start <= 0:
        return None
    candidates = (metadata.get("request_end_ns"), metadata.get("cancellation_time_ns"))
    ends = [end for end in candidates if type(end) is int and end >= start]
    if full_ms is not None:
        ends.append(start + round(full_ms * 1_000_000))
    return (start, max(ends)) if ends else None


def _sample(record):
    metrics = record.get("metrics") or {}
    input_tokens = _count(_metric(metrics, "usage_prompt_tokens"))
    output_tokens = _count(_metric(metrics, "usage_completion_tokens"))
    cached_tokens = _count(_metric(metrics, "usage_prompt_cache_read_tokens"))
    if input_tokens is None or (cached_tokens is not None and cached_tokens > input_tokens):
        cached_tokens = None

    ttft = _milliseconds(metrics, "time_to_first_token")
    decode = _milliseconds(metrics, "full_decode_duration")
    tpot = None
    if output_tokens not in (0, 1):
        tpot = _milliseconds(metrics, "full_response_inter_token_latency")
    if tpot is None and output_tokens is not None and output_tokens > 1 and decode is not None:
        tpot = decode / (output_tokens - 1)
    spec = record.get("spec_decode_acceptance") or {}
    steps = _count(_metric(metrics, "spec_decode_steps"))
    accepted = _count(_metric(metrics, "spec_decode_accepted_draft_tokens"))
    drafts = _count(_metric(metrics, "spec_decode_draft_tokens"))
    full_ms = ttft + decode if ttft is not None and decode is not None else None
    return RequestSample(
        ttft, tpot, input_tokens, output_tokens, cached_tokens,
        steps if steps is not None else _count(spec.get("num_spec_steps")),
        accepted if accepted is not None else _count(spec.get("num_accepted_draft_tokens")),
        drafts if drafts is not None else _count(spec.get("num_draft_tokens")),
        _wire_interval(record["metadata"], full_ms),
    )


def _profiling_records(path, *, keep_intervals=False):
    export = ProfilingExport()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record["metadata"]
                if metadata["benchmark_phase"] != "profiling":
                    continue
                if metadata.get("context_overflow_skip"):
                    export.skipped += 1
                    continue
                if record.get("error") or metadata.get("was_cancelled"):
                    export.failed += 1
                    export.cancelled += bool(metadata.get("was_cancelled"))
                    interval = _wire_interval(metadata) if keep_intervals else None
                else:
                    sample = _sample(record)
                    export.samples.append(sample)
                    interval = sample.wire
                if keep_intervals and interval is not None:
                    export.intervals.append(interval)
            except (ValueError, TypeError, KeyError, AttributeError):
                export.malformed += 1
                if export.first_malformed_line is None:
                    export.first_malformed_line = line_number
    return export


def _percentiles(values, prefix):
    ordered = sorted(value for value in values if value is not None)
    stats = {}
    if ordered:
        for percentile in PERCENTILES:
            index = (len(ordered) - 1) * percentile / 100
            low, high = math.floor(index), math.ceil(index)
            stats[f"{prefix}_p{percentile}_ms"] = (
                ordered[low] + (ordered[high] - ordered[low]) * (index - low)
            )
    return stats, len(ordered)


def _speculation(samples):
    step_totals = draft_totals = accepted_steps = accepted_drafts = coverage = 0
    for sample in samples:
        accepted, steps, drafts = sample.spec_accepted, sample.spec_steps, sample.spec_drafts
        if accepted is None or (drafts is not None and accepted > drafts):
            continue
        valid_steps = steps is not None and steps > 0
        valid_drafts = drafts is not None and drafts > 0
        if valid_steps:
            step_totals += steps
            accepted_steps += accepted
        if valid_drafts:
            draft_totals += drafts
            accepted_drafts += accepted
        coverage += valid_steps or valid_drafts
    return {
        "spec_samples": coverage,
        "spec_acceptance_length": 1 + accepted_steps / step_totals if step_totals else None,
        "spec_acceptance_percent": 100 * accepted_drafts / draft_totals if draft_totals else None,
        "spec_source": "响应推测计数" if coverage else None,
    }


def _token_metrics(samples, seconds):
    totals = {}
    row = {}
    for tag, attribute in (("input", "input_tokens"), ("output", "output_tokens"),
                           ("kv", "cached_tokens")):
        values = [value for sample in samples if (value := getattr(sample, attribute)) is not None]
        row[f"{tag}_usage_samples"] = len(values)
        totals[tag] = sum(values) if values and len(values) == len(samples) else None
    if seconds:
        for tag in ("input", "output"):
            if totals[tag] is not None:
                row[f"{tag}_tpm"] = 60 * totals[tag] / seconds
        if totals["input"] is not None and totals["output"] is not None:
            row["total_tpm"] = 60 * (totals["input"] + totals["output"]) / seconds
    if totals["input"] and totals["kv"] is not None:
        row["kv_hit_percent"] = 100 * totals["kv"] / totals["input"]
        row["kv_source"] = "服务端响应 Usage"
    return row


def _aggregate_path(artifact_dir):
    preferred = artifact_dir / "profile_export_aiperf.json"
    return preferred if preferred.is_file() else artifact_dir / "profile_export.json"


def _sla(row):
    if any(_number(row.get(key)) is None for key in SLA_MS):
        return "unknown", "延迟分位数不足"
    over = [
        f"{key.removesuffix('_ms').replace('_', ' ').upper()} {row[key]:,.2f} > {limit:,} ms"
        for key, limit in SLA_MS.items() if row[key] > limit
    ]
    return ("fail", "；".join(over)) if over else ("pass", "六项延迟均在 SLA 范围内")


def summarize(artifact_dir: Path, experiment: str, concurrency: int, *,
              expected_requests: int | None = None, expected_input: int | None = None,
              expected_output: int | None = None) -> dict:
    """Assess SLA and evidence independently, even when a profile exported only partly."""
    artifact_dir = Path(artifact_dir)
    row = {key: None for key, _ in COLUMNS}
    row.update(experiment=experiment, concurrency=concurrency, repeat=1, repeat_count=1)
    incomplete, invalid = [], []
    data = {}
    try:
        data = json.loads(_aggregate_path(artifact_dir).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
            raise ValueError("汇总应为 JSON 对象")
    except FileNotFoundError:
        incomplete.append("缺少 AIPerf 汇总")
    except (OSError, ValueError) as exc:
        incomplete.append(f"无法读取 AIPerf 汇总（{exc}）")

    duration = _milliseconds(data, "benchmark_duration", "avg")
    seconds = duration / 1_000 if duration else None
    duration_source = "AIPerf 正式测量" if seconds is not None else None
    export = None
    try:
        export = _profiling_records(artifact_dir / "profile_export.jsonl", keep_intervals=seconds is None)
        if seconds is None and export.intervals:
            span = max(end for _, end in export.intervals) - min(start for start, _ in export.intervals)
            if span > 0:
                seconds, duration_source = span / 1e9, "可用 HTTP 请求区间"
    except FileNotFoundError:
        incomplete.append("缺少逐请求导出")
    except (OSError, ValueError) as exc:
        incomplete.append(f"无法读取逐请求导出（{exc}）")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    submission_valid = metadata.get("submission_valid", data.get("submission_valid"))
    submission_notes = metadata.get("submission_invalid_reasons") or data.get(
        "runtime_submission_invalid_reasons"
    ) or ()
    if isinstance(submission_notes, str):
        submission_notes = [submission_notes]
    elif not isinstance(submission_notes, (list, tuple)):
        submission_notes = [str(submission_notes)]

    row.update(duration_seconds=seconds, duration_source=duration_source,
               submission_valid=submission_valid,
               submission_notes=" / ".join(map(str, submission_notes)) or None)
    if data.get("is_complete") is False or data.get("was_cancelled") is True:
        incomplete.append("AIPerf 未完成或取消")
    if submission_valid is False:
        invalid.append("AIPerf 场景提交判定无效")
    if export is not None:
        samples, failed, cancelled, skipped = export.samples, export.failed, export.cancelled, export.skipped
        aggregate_skips = _count(_metric(data, "skipped_context_overflow_count", "avg"))
        row.update(
            attempted_requests=len(samples) + failed, completed_requests=len(samples),
            failed_requests=failed, cancelled_requests=cancelled,
            context_skips=max(skipped, aggregate_skips or 0),
        )
        ttft, ttft_count = _percentiles((sample.ttft for sample in samples), "ttft")
        tpot, tpot_count = _percentiles((sample.tpot for sample in samples), "tpot")
        row.update(ttft, ttft_samples=ttft_count)
        row.update(tpot, tpot_samples=tpot_count)
        row.update(_token_metrics(samples, seconds))
        row.update(_speculation(samples))

        if export.malformed:
            incomplete.append(
                f"{export.malformed} 行逐请求导出无法解析，首处第 {export.first_malformed_line} 行"
            )
        aggregate_count = _count(_metric(data, "request_count", "avg"))
        aggregate_errors = _count(_metric(data, "error_request_count", "avg"))
        if aggregate_count is not None and aggregate_count != len(samples):
            incomplete.append(f"汇总成功数 {aggregate_count} 与逐请求成功数 {len(samples)} 不符")
        if aggregate_errors is not None and aggregate_errors != failed:
            incomplete.append(f"汇总失败数 {aggregate_errors} 与逐请求失败数 {failed} 不符")
        if failed:
            invalid.append(f"{failed} 个正式请求失败（其中 {cancelled} 个取消）")
        if expected_requests is not None:
            attempted = row["attempted_requests"]
            if attempted < expected_requests:
                incomplete.append(f"正式请求仅 {attempted}/{expected_requests}")
            elif attempted > expected_requests:
                invalid.append(f"正式请求数 {attempted} 超过预期 {expected_requests}")
        for label, attr, expected in (("输入", "input_tokens", expected_input),
                                      ("输出", "output_tokens", expected_output)):
            if expected is not None:
                mismatches = sum(getattr(sample, attr) != expected for sample in samples)
                if mismatches:
                    invalid.append(f"{mismatches} 个请求的服务端{label}计量缺失或不等于 {expected}")

        tpot_expected = sum(sample.output_tokens not in (0, 1) for sample in samples)
        if not samples or ttft_count != len(samples) or not tpot_count or tpot_count != tpot_expected:
            incomplete.append(f"延迟样本不足（TTFT {ttft_count}/{len(samples)}，TPOT {tpot_count}/{tpot_expected}）")

    sla_status, sla_detail = _sla(row)
    validity = "invalid" if invalid else ("incomplete" if incomplete or sla_status == "unknown" else "valid")
    validity_reason = "；".join(invalid + incomplete) or None
    status = {"invalid": "invalid", "incomplete": "error", "valid": sla_status}[validity]
    row.update(sla_status=sla_status, validity=validity, validity_reason=validity_reason,
               status=status, detail="；".join(filter(None, (validity_reason, sla_detail))))
    return row


COLORS = ("#175eb9", "#b45b13", "#8849ae")
LATENCY_SERIES = tuple(zip(("P50", "P90", "P99"), COLORS))
OBSERVED = "#16786a"
STYLE = """
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#16283a;background:#f3f7fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:32px}
header{border-bottom:1px solid #cddbe6;padding:8px 0 24px;margin-bottom:24px}
h1{font-size:clamp(1.8rem,3vw,2.6rem);margin:0 0 8px;letter-spacing:-.035em}
h2{font-size:1.55rem;margin:0}.muted,p{color:#536579}p{line-height:1.6}
header p{margin:0}a{color:#175eb9}nav{display:flex;flex-wrap:wrap;gap:18px;margin-top:16px}
section{margin:34px 0}.heading{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:14px}
.heading p{margin:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}
figure,.table-wrap,.limits{background:#fff;border:1px solid #d9e3eb;border-radius:10px}
figure{min-width:0;margin:0;padding:14px 16px}figcaption{font-weight:650;margin-bottom:6px}
figure svg{display:block;width:100%;height:auto;overflow:visible}
figure .missing{display:grid;place-items:center;min-height:180px;margin:0;color:#687b8c}
svg text{font-family:inherit;fill:#546678;font-size:11px}.axis{stroke:#8d9faf;stroke-width:1px}
.gridline{stroke:#dce6ee;stroke-width:1px}.limits{padding:12px 16px;color:#536579;margin:0 0 24px}
.table-wrap{overflow:auto;margin-top:16px}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}
th,td{text-align:right;padding:10px 11px;border-bottom:1px solid #e4ebf1;font-variant-numeric:tabular-nums}
th{background:#eaf2f8;font-weight:650}td:nth-child(-n+2),th:nth-child(-n+2),td:last-child,th:last-child{text-align:left}
td:last-child{min-width:180px;max-width:450px;white-space:normal}tr:last-child td{border-bottom:0}
.badge{font-weight:700}.badge.pass,.badge.valid{color:#07674b}
.badge.fail,.badge.invalid{color:#a32c34}.badge.incomplete,.badge.unknown{color:#82520f}
.metadata{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:10px;margin:16px 0 0}
.meta,.summary-card{background:#fff;border:1px solid #d9e3eb;border-radius:10px;padding:11px 14px;min-width:0}
.meta dt,.summary-card small{color:#536579;font-size:13px}.meta dd{margin:4px 0 0;overflow-wrap:anywhere}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,165px),1fr));gap:10px;margin:0 0 14px}
.summary-card b{display:block;margin:3px 0 0;font-size:1.3rem;font-variant-numeric:tabular-nums}
.compact{margin:0 0 14px}.compact td:last-child{min-width:0;text-align:right}
.group{margin-top:14px}.group>summary,.point>summary{cursor:pointer;font-weight:650;color:#24445f}
.group>summary{margin:12px 0}.point{background:#fff;border:1px solid #d9e3eb;border-radius:10px;padding:10px 14px;margin:8px 0}
.point>p{margin:8px 0}.point nav{margin:8px 0;gap:16px}.point .grid{margin-top:12px}
svg .cutoff{stroke:#a32c34;stroke-width:1.2;stroke-dasharray:5 4}
.report-note{font-size:13px;color:#536579;margin:8px 0}
@media(max-width:560px){main{padding:16px}figure{padding:10px}.heading{gap:6px}}
"""


def _chart(title, unit, rows, series, *, thresholds=(), minimum_ceiling=0,
           axis_seconds=None, cutoff=None):
    observations = [
        value / scale for key, _, _, scale in series for row in rows
        if (value := _number(row.get(key))) is not None
    ]
    heading = f"<figure><figcaption>{escape(title)}</figcaption>"
    if not observations:
        return heading + '<p class="missing">无可用数据</p></figure>'

    ceiling = max([minimum_ceiling, 0.01, *observations, *(limit for limit, _ in thresholds)])
    ceiling *= 1.05 if ceiling > minimum_ceiling else 1
    left, right, top, bottom = 76, 522, 16, 166
    width, height = right - left, bottom - top

    def trace_x(seconds):
        return left + width * seconds / axis_seconds

    def x(index):
        if axis_seconds is not None:
            return trace_x(rows[index]["seconds"])
        return left + width / 2 if len(rows) == 1 else left + index * width / (len(rows) - 1)

    def y(value):
        return bottom - height * value / ceiling

    chart = [heading, f'<svg viewBox="0 0 546 224" role="img" aria-label="{escape(title)}">']
    if axis_seconds is not None and cutoff is not None and 0 <= cutoff <= axis_seconds:
        cutoff_x = trace_x(cutoff)
        chart.append(
            f'<rect x="{cutoff_x:.1f}" y="{top}" width="{right - cutoff_x:.1f}" height="{height}" fill="#fbefed"/>'
        )
    for step in range(5):
        value = step * ceiling / 4
        offset = y(value)
        chart.append(
            f'<path class="gridline" d="M{left} {offset:.1f}H{right}"/>'
            f'<text x="{left - 8}" y="{offset + 4:.1f}" text-anchor="end">{value:,.3g}</text>'
        )
    chart.append(f'<path class="axis" fill="none" d="M{left} {top}V{bottom}H{right}"/>')
    if axis_seconds is not None:
        for index in range(5):
            seconds = axis_seconds * index / 4
            chart.append(
                f'<text x="{trace_x(seconds):.1f}" y="{bottom + 15}" text-anchor="middle">'
                f'{seconds:,.0f} s</text>'
            )
        if cutoff is not None and 0 <= cutoff <= axis_seconds:
            cutoff_x = trace_x(cutoff)
            chart.append(
                f'<path class="cutoff" d="M{cutoff_x:.1f} {top}V{bottom}">'
                f'<title>发包截止 {cutoff:,.2f} 秒</title></path>'
            )
    else:
        for index, row in enumerate(rows):
            if len(rows) <= 10 or index in (0, len(rows) - 1) or index % math.ceil(len(rows) / 8) == 0:
                chart.append(
                    f'<text x="{x(index):.1f}" y="{bottom + 15}" text-anchor="middle">'
                    f'{escape(_point_label(row))}</text>'
                )
    for limit, color in thresholds:
        line = y(limit)
        chart.append(
            f'<path d="M{left} {line:.1f}H{right}" stroke="{color}" stroke-dasharray="5 4" opacity=".65">'
            f'<title>参考线 {limit:g} {escape(unit)}</title></path>'
        )
    for key, label, color, scale in series:
        previous = None
        for index, row in enumerate(rows):
            measurement = _number(row.get(key))
            if measurement is None:
                previous = None
                continue
            value = measurement / scale
            point = (x(index), y(value))
            if previous is not None:
                chart.append(
                    f'<path d="M{previous[0]:.1f} {previous[1]:.1f}L{point[0]:.1f} {point[1]:.1f}" '
                    f'stroke="{color}" stroke-width="2" fill="none"/>'
                )
            label_x = f'{row["seconds"]:,.1f} s' if axis_seconds is not None else _point_label(row)
            radius = 2.6 if axis_seconds is not None else 4
            chart.append(
                f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="{radius}" fill="{color}">'
                f'<title>{escape(label_x)} {escape(label)} {value:,.4g} {escape(unit)}'
                '</title></circle>'
            )
            previous = point
    for index, (_, label, color, _) in enumerate(series):
        offset = left + index * 116
        chart.append(
            f'<circle cx="{offset}" cy="204" r="3.5" fill="{color}"/>'
            f'<text x="{offset + 9}" y="208">{escape(label)}</text>'
        )
    chart.append(f'<text x="{right}" y="208" text-anchor="end">{escape(unit)}</text></svg></figure>')
    return "".join(chart)


def _point_label(row):
    return f"C{row['concurrency']} / R{row.get('repeat', 1)}"


def _badge(value, labels):
    if value not in labels:
        return "—"
    return f'<span class="badge {value}">{labels[value]}</span>'


def _format_value(value):
    if value is None:
        return "—"
    if type(value) is bool:
        return "是" if value else "否"
    if type(value) is float:
        return f"{value:,.5g}"
    if isinstance(value, (tuple, list)):
        return escape(", ".join(map(str, value))) or "—"
    if isinstance(value, dict):
        return escape(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return escape(str(value))


def _table(rows):
    fields = [(key, label) for key, label in COLUMNS[1:]
              if key not in ("status", "point_dir", "artifact_dir", "log_dir", "phase_summary")]
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend(f'<th scope="col">{escape(label)}</th>' for _, label in fields)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in fields:
            value = row.get(key)
            if key == "sla_status":
                display = _badge(value, SLA_NAMES)
            elif key == "validity":
                display = _badge(value, VALIDITY_NAMES)
            else:
                display = _format_value(value)
            parts.append(f"<td>{display}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _compact_table(rows):
    headers = ("档位 / 重复", "实测 SLA", "有效性", "TTFT P50 / P90 / P99 · s",
               "TPOT P50 / P90 / P99 · ms", "输入 TPM", "输出 TPM", "总 TPM")
    parts = ['<div class="table-wrap compact"><table><thead><tr>']
    parts.extend(f'<th scope="col">{label}</th>' for label in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        cells = [
            escape(_point_label(row)),
            _badge(row.get("sla_status"), SLA_NAMES),
            _badge(row.get("validity"), VALIDITY_NAMES),
        ]
        for key, scale in (("ttft", 1_000), ("tpot", 1)):
            values = (_number(row.get(f"{key}_p{p}_ms")) for p in PERCENTILES)
            cells.append(" / ".join(f"{value / scale:,.2f}" if value is not None else "—"
                                    for value in values))
        cells.extend(f"{value:,.0f}" if (value := _number(row.get(key))) is not None else "—"
                     for key in ("input_tpm", "output_tpm", "total_tpm"))
        parts.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _prepare_row(row, manifest_repeats):
    row = dict(row)
    row["repeat"] = row.get("repeat") or 1
    row["repeat_count"] = max(row.get("repeat_count") or 1, manifest_repeats)
    if row.get("sla_status") not in SLA_NAMES:
        row["sla_status"] = row["status"] if row.get("status") in ("pass", "fail") else _sla(row)[0]
    if row.get("validity") not in VALIDITY_NAMES:
        row["validity"] = {"pass": "valid", "fail": "valid", "invalid": "invalid"}.get(
            row.get("status"), "incomplete"
        )
    if row.get("submission_valid") is False:
        reason = row.get("validity_reason") or "AIPerf 场景提交判定无效"
        row.update(status="invalid", validity="invalid", validity_reason=reason)
    for source_key, tags in (
        ("kv_source", ("kv_hit_percent",)),
        ("spec_source", ("spec_acceptance_length", "spec_acceptance_percent")),
    ):
        source = str(row.get(source_key) or "")
        # Old reports placed Prometheus values in the unscoped columns.
        if source.startswith("prometheus_with_warmup"):
            for tag in tags:
                if row.get(f"whole_run_{tag}") is None:
                    row[f"whole_run_{tag}"] = row.get(tag)
                row[tag] = None
            row["whole_run_scope"] = "prometheus_process_including_warmup"
            row["whole_run_source"] = row.get("whole_run_source") or source
            row[source_key] = None
    return row


def _summary_cards(experiment, rows):
    repeats_by_c = {}
    for row in rows:
        repeats_by_c.setdefault(row["concurrency"], []).append(row)
    passed_levels = []
    for concurrency, repeated in repeats_by_c.items():
        expected = max(row["repeat_count"] for row in repeated)
        seen = {row["repeat"] for row in repeated}
        if (len(repeated) == expected and seen == set(range(1, expected + 1))
                and all(row["validity"] == "valid" and row["sla_status"] == "pass" for row in repeated)):
            passed_levels.append(concurrency)
    complete = sum(row["validity"] == "valid" for row in rows)
    passed = sum(row["validity"] == "valid" and row["sla_status"] == "pass" for row in rows)
    unit = "交通轨迹 lane" if experiment == "P03" else "并发请求"
    labels = (
        (f"全部重复达标的最高档 · {unit}", max(passed_levels, default=None)),
        ("有效重复", f"{complete} / {len(rows)}"),
        ("有效且达标", f"{passed} / {len(rows)}"),
        ("已记录请求", sum(row.get("completed_requests") or 0 for row in rows)),
    )
    return '<div class="stats">' + "".join(
        f'<div class="summary-card"><small>{escape(label)}</small><b>{_format_value(value)}</b></div>'
        for label, value in labels
    ) + "</div>"


def _relative(value):
    if not value:
        return None
    value = str(value)
    if Path(value).is_absolute() or "\\" in value or ":" in value or value.startswith("//"):
        return None
    return value


def _link(root, path, text):
    path = _relative(path)
    if path is None or not (root / path).is_file():
        return ""
    return f'<a href="{quote(path)}">{escape(text)}</a>'


@lru_cache(maxsize=48)
def _cached_timeline(path, size, mtime, profile_start, send_end, requests_end):
    try:
        export = _profiling_records(path, keep_intervals=True)
    except (OSError, ValueError):
        return None
    return build_timeline(export, profile_start, send_end, requests_end)


def _timeline_content(point, root):
    artifact = _relative(point.get("artifact_dir"))
    if artifact is None:
        artifact = str(Path("P03") / f"C{point['concurrency']}" / "formal")
    source = root / artifact / "profile_export.jsonl"
    try:
        stat = source.stat()
    except OSError:
        return ""
    timeline = _cached_timeline(
        source, stat.st_size, stat.st_mtime_ns, point.get("profile_start_ns"),
        point.get("send_end_ns"), point.get("requests_end_ns")
    )
    if not timeline:
        return ""
    bins = timeline["rows"]
    notes = (
        f'{point["concurrency"]} 条配置轨迹 lane · 实测 HTTP 在途均值 {timeline["mean"]:.2f}，'
        f'峰值 {timeline["peak"]} · {timeline["intervals"]} 条请求时间区间'
    )
    if not timeline["complete"]:
        notes += " · 仅覆盖可解析的时间区间"
    marker = "红线为实际发包截止。" if timeline["cutoff"] is not None else ""
    html = [
        f'<p><strong>{notes}</strong></p>',
        '<p class="report-note">'
        f'从{timeline["origin"]}起按 {timeline["bin_seconds"]} 秒分箱。{marker}'
        '吞吐为每箱完成请求的计量折算值。延迟样本缺失时曲线留空。'
        '</p><div class="grid">',
    ]
    for title, unit, series in (
        ("HTTP 在途请求", "请求", (
            ("in_flight_mean", "区间平均", COLORS[0], 1), ("in_flight_peak", "区间峰值", COLORS[1], 1))),
        ("完成请求的 Token 吞吐", "Token / 分钟", (
            ("input_tpm", "输入", COLORS[0], 1), ("output_tpm", "输出", COLORS[1], 1))),
        ("TTFT 时间线", "秒", (
            ("ttft_p50_ms", "P50", COLORS[0], 1_000), ("ttft_p99_ms", "P99", COLORS[2], 1_000))),
        ("TPOT 时间线", "毫秒", (
            ("tpot_p50_ms", "P50", COLORS[0], 1), ("tpot_p99_ms", "P99", COLORS[2], 1))),
    ):
        html.append(_chart(title, unit, bins, series,
                           axis_seconds=timeline["span"], cutoff=timeline["cutoff"]))
    html.append("</div>")
    return "".join(html)



METRIC_SCOPE_LABELS = {
    "prometheus_formal": "Prometheus 正式阶段，已核对请求数和输入 token 数",
    "prometheus_formal_count_verified": "Prometheus 正式窗口，已核对边界和请求数",
    "prometheus_approximate_formal_and_drain": "Prometheus 近似正式与收尾窗口",
    "prometheus_process_including_warmup": "Prometheus 整轮计数，含预热",
    "prometheus_process": "Prometheus 进程边界计数，范围未核对",
}


def _point_evidence(point, root, *, expanded=False):
    summary = escape(_point_label(point))
    detail = point.get("detail")
    links = [_link(root, point.get("phase_summary"), "阶段摘要")]
    log_dir = _relative(point.get("log_dir"))
    if log_dir is not None:
        for name, label in (("plan.json", "档位配置"), ("formal.log", "正式日志"),
                            ("warmup.log", "预热日志"), ("metrics-before.prom", "计数器起点"),
                            ("metrics-after.prom", "计数器终点")):
            links.append(_link(root, str(Path(log_dir) / name), label))
    artifact = _relative(point.get("artifact_dir"))
    if artifact is not None:
        for name, label in (("profile_export.jsonl", "逐请求原始导出"),
                            ("profile_export_aiperf.json", "AIPerf 汇总"),
                            ("profile_export.json", "AIPerf 汇总")):
            links.append(_link(root, str(Path(artifact) / name), label))
    label = "时间线与原始证据" if point["experiment"] == "P03" else "原始证据"
    opened = " open" if expanded else ""
    html = [f'<details class="point"{opened}><summary>{summary} {label}</summary>']
    if detail:
        html.append(f'<p>{escape(str(detail))}</p>')
    if any(links):
        html.append('<nav aria-label="原始证据">' + "".join(links) + '</nav>')
    timings = [
        f"{label} {value:,.3f} 秒" for key, label in (
            ("prepare_seconds", "准备"), ("warmup_seconds", "预热"),
            ("send_seconds", "实际发包"), ("drain_seconds", "实际收尾"),
            ("configured_send_seconds", "配置发包"), ("configured_drain_seconds", "配置收尾")
        ) if (value := _number(point.get(key))) is not None
    ]
    in_flight = _count(point.get("in_flight_at_cutoff"))
    if in_flight is not None:
        timings.append(f"发包截止时在途 {in_flight}")
    last_response = point.get("drain_last_response_seconds")
    if type(last_response) in (int, float) and math.isfinite(last_response):
        timings.append(f"最后响应距截止 {last_response:+,.3f} 秒")
    if timings:
        html.append(f'<p class="report-note">{" · ".join(timings)}</p>')
    sources = []
    for keys, label in ((("kv_source",), "正式阶段缓存"), (("spec_source",), "正式阶段推测"),
                        (("window_scope", "window_source"), "正式采集窗口"),
                        (("whole_run_scope", "whole_run_source"), "整轮计数")):
        source = next((point[key] for key in keys if point.get(key)), None)
        if source is not None:
            human = " + ".join(METRIC_SCOPE_LABELS.get(item, item) for item in str(source).split(" + "))
            sources.append(f"{label} {escape(human)}")
    if sources:
        html.append(f'<p class="report-note">{" · ".join(sources)}</p>')
    if point["experiment"] == "P03":
        html.append(_timeline_content(point, root))
    html.append("</details>")
    return "".join(html)


def _section(experiment, rows, root):
    name = EXPERIMENT_NAMES.get(experiment, "")
    html = [
        f'<section id="{escape(experiment)}"><div class="heading"><h2>{escape(experiment)} '
        f'{escape(name)}</h2></div>', _summary_cards(experiment, rows), _compact_table(rows),
        '<div class="grid">',
    ]
    for prefix, unit, scale in (("ttft", "秒", 1_000), ("tpot", "毫秒", 1)):
        series = tuple((f"{prefix}_{label.lower()}_ms", label, color, scale)
                       for label, color in LATENCY_SERIES)
        limits = tuple((SLA_MS[key] / scale, color) for key, _, color, _ in series)
        html.append(_chart(prefix.upper(), unit, rows, series, thresholds=limits))
    html.append(_chart("Token 吞吐", "Token / 分钟", rows, (
        ("input_tpm", "输入", COLORS[0], 1), ("output_tpm", "输出", COLORS[1], 1),
        ("total_tpm", "总量", COLORS[2], 1),
    )))
    for title, metric, unit, ceiling in (
        ("KVCache 命中率", "kv_hit_percent", "%", 100),
        ("推测解码接受长度", "spec_acceptance_length", "Token", 0),
        ("推测解码草稿接受率", "spec_acceptance_percent", "%", 100),
    ):
        series = ((metric, "正式阶段", COLORS[0], 1),
                  (f"window_{metric}", "采集窗口", COLORS[1], 1),
                  (f"whole_run_{metric}", "整轮计数", COLORS[2], 1))
        formal_kv = (experiment == "P03" and metric == "kv_hit_percent"
                     and any(_number(row.get(metric)) is not None for row in rows))
        cache_reference = ((90, COLORS[0]),) if formal_kv else ()
        html.append(_chart(title, unit, rows, series, thresholds=cache_reference,
                           minimum_ceiling=ceiling))
    html.append(_chart("正式请求", "请求", rows, (
        ("completed_requests", "成功", OBSERVED, 1),
        ("failed_requests", "失败", "#a32c34", 1),
        ("context_skips", "上下文跳过", COLORS[1], 1),
    )))
    html.extend(("</div>", '<details class="group"><summary>查看完整指标表</summary>',
                 _table(rows), "</details>"))
    html.extend(_point_evidence(row, root, expanded=experiment == "P03" and len(rows) == 1)
                for row in rows)
    html.append("</section>")
    return "".join(html)


def _display_endpoint(value):
    try:
        parsed = urlsplit(str(value))
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    host = parsed.netloc.rsplit("@", 1)[-1]
    return f"{parsed.scheme}://{host}{parsed.path}"


def _run_metadata(manifest):
    if not manifest:
        return ""
    fields = [
        ("模型", manifest.get("model")),
        ("API", _display_endpoint(manifest.get("url"))),
        ("代码提交", manifest.get("code_commit")),
        ("工作区改动", manifest.get("code_dirty")),
        ("AIPerf", manifest.get("aiperf_version")),
        ("Python", manifest.get("python_version")),
        ("Tokenizer SHA-256", manifest.get("tokenizer_sha256")),
        ("随机种子", manifest.get("seed")),
        ("数据 nonce", manifest.get("dataset_nonce")),
    ]
    dataset = manifest.get("agentx_dataset")
    if isinstance(dataset, dict):
        fields.extend((("AgentX 数据集", dataset.get("public_alias")),
                       ("AgentX 版本", dataset.get("revision", dataset.get("commit")))))
    deployment = manifest.get("deployment")
    if isinstance(deployment, dict):
        fields.extend((label, deployment.get(key)) for key, label in (
            ("label", "部署标签"), ("backend_container", "后端容器"), ("image", "镜像"),
            ("image_id", "镜像 ID"), ("engine_commit", "引擎提交"), ("engine_version", "引擎版本"),
            ("gpu_devices", "GPU 设备"), ("tensor_parallel", "张量并行 TP"),
            ("data_parallel", "数据并行 DP"), ("pipeline_parallel", "流水线并行 PP"),
            ("expert_parallel", "专家并行 EP"), ("prefill_schedule_interval", "Prefill 调度间隔"),
            ("max_num_batched_tokens", "最大批处理 token"), ("max_num_seqs", "最大序列数"),
            ("max_model_len", "模型上下文上限"), ("kv_cache_dtype", "KVCache 类型"),
            ("routing_policy", "路由策略"), ("cache_start_state", "初始缓存状态"),
            ("speculative_config", "推测解码配置"),
        ))
    routing = manifest.get("routing")
    if isinstance(routing, dict):
        fields.extend((f"路由 {key}", value) for key, value in sorted(routing.items()))
    return '<dl class="metadata">' + "".join(
        f'<div class="meta"><dt>{escape(label)}</dt><dd>{_format_value(value)}</dd></div>'
        for label, value in fields if value is not None
    ) + "</dl>"


def write_reports(run_results: Path, rows: list[dict], manifest=None) -> None:
    """Write a CSV and offline HTML report, retaining any available partial measurements."""
    run_results = Path(run_results)
    run_results.mkdir(parents=True, exist_ok=True)
    manifest = manifest or {}
    plans = Counter(
        (point.get("experiment"), point.get("concurrency"))
        for point in manifest.get("points", ()) if isinstance(point, dict)
    )
    ordered = sorted(
        (_prepare_row(row, plans[row["experiment"], row["concurrency"]]) for row in rows),
        key=lambda row: (row["experiment"], row["concurrency"], row["repeat"]),
    )
    csv_tmp = run_results / "summary.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key, _ in COLUMNS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    csv_tmp.replace(run_results / "summary.csv")
    experiments = sorted({row["experiment"] for row in ordered})
    navigation = "".join(f'<a href="#{escape(scene)}">{escape(scene)}</a>' for scene in experiments)
    sections = "".join(
        _section(scene, [row for row in ordered if row["experiment"] == scene], run_results)
        for scene in experiments
    )
    run_id = escape(str(manifest.get("run_id") or run_results.name))
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Benchmark · {run_id}</title><style>{STYLE}</style><main>'
        f'<header><h1>Benchmark 压测结果</h1><p>{run_id}</p>'
        f'<nav>{navigation}<a href="summary.csv">下载 CSV</a></nav>{_run_metadata(manifest)}</header>'
        '<p class="limits">SLA 阈值按 P50、P90、P99 排列。TTFT 为 5 / 10 / 30 秒，'
        'TPOT 为 30 / 50 / 80 毫秒。延迟图的虚线对应同色分位数的阈值。'
        '实测 SLA 和数据有效性分别显示，只有有效且全部重复达标的档位才计入最高达标档。'
        'KVCache 和推测解码分别显示正式阶段、采集窗口和整轮计数的观测值。</p>'
        f'{sections or "<p>尚未记录任何档位。</p>"}</main></html>'
    )
    html_tmp = run_results / "report.html.tmp"
    html_tmp.write_text(html, encoding="utf-8")
    html_tmp.replace(run_results / "report.html")
