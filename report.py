"""Summarize profiling requests and render a standalone benchmark report."""

import csv
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path


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
STATUS_NAMES = {"pass": "达标", "fail": "超出 SLA", "invalid": "无效", "error": "执行异常"}
COLUMNS = (
    ("experiment", "实验"), ("concurrency", "并发"), ("status", "SLA"),
    ("ttft_p50_ms", "TTFT P50 · ms"), ("ttft_p90_ms", "TTFT P90 · ms"),
    ("ttft_p99_ms", "TTFT P99 · ms"), ("tpot_p50_ms", "TPOT P50 · ms"),
    ("tpot_p90_ms", "TPOT P90 · ms"), ("tpot_p99_ms", "TPOT P99 · ms"),
    ("input_tpm", "输入 TPM"), ("output_tpm", "输出 TPM"), ("total_tpm", "总 TPM"),
    ("kv_hit_percent", "KVCache 命中 · %"), ("spec_acceptance_length", "接受长度"),
    ("spec_acceptance_percent", "草稿接受率 · %"), ("attempted_requests", "正式请求"),
    ("completed_requests", "成功"), ("failed_requests", "失败"),
    ("cancelled_requests", "取消"), ("context_skips", "上下文跳过"),
    ("ttft_samples", "TTFT 样本"), ("tpot_samples", "TPOT 样本"),
    ("input_usage_samples", "输入计量样本"), ("output_usage_samples", "输出计量样本"),
    ("kv_usage_samples", "缓存计量样本"), ("spec_samples", "推测解码样本"),
    ("duration_seconds", "正式测量 · 秒"), ("kv_source", "缓存数据来源"),
    ("spec_source", "推测数据来源"), ("submission_valid", "AIPerf 提交有效"),
    ("submission_notes", "AIPerf 提交备注"), ("detail", "判定说明"),
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


def _sample(record):
    metrics = record.get("metrics") or {}
    input_tokens = _count(_metric(metrics, "usage_prompt_tokens"))
    output_tokens = _count(_metric(metrics, "usage_completion_tokens"))
    cached_tokens = _count(_metric(metrics, "usage_prompt_cache_read_tokens"))
    if input_tokens is None or (cached_tokens is not None and cached_tokens > input_tokens):
        cached_tokens = None

    tpot = None
    if output_tokens not in (0, 1):
        tpot = _milliseconds(metrics, "full_response_inter_token_latency")
    if tpot is None and output_tokens is not None and output_tokens > 1:
        decode = _milliseconds(metrics, "full_decode_duration")
        if decode is not None:
            tpot = decode / (output_tokens - 1)
    spec = record.get("spec_decode_acceptance") or {}
    steps = _count(_metric(metrics, "spec_decode_steps"))
    accepted = _count(_metric(metrics, "spec_decode_accepted_draft_tokens"))
    drafts = _count(_metric(metrics, "spec_decode_draft_tokens"))
    return RequestSample(
        _milliseconds(metrics, "time_to_first_token"), tpot, input_tokens, output_tokens,
        cached_tokens,
        steps if steps is not None else _count(spec.get("num_spec_steps")),
        accepted if accepted is not None else _count(spec.get("num_accepted_draft_tokens")),
        drafts if drafts is not None else _count(spec.get("num_draft_tokens")),
    )


def _profiling_records(path):
    samples = []
    failed = cancelled = skipped = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record["metadata"]
                if metadata["benchmark_phase"] != "profiling":
                    continue
                if metadata.get("context_overflow_skip"):
                    skipped += 1
                elif record.get("error") or metadata.get("was_cancelled"):
                    failed += 1
                    cancelled += bool(metadata.get("was_cancelled"))
                else:
                    samples.append(_sample(record))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"逐请求导出的第 {line_number} 行无效") from exc
    return samples, failed, cancelled, skipped


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


def summarize(artifact_dir: Path, experiment: str, concurrency: int, *,
              expected_requests: int | None = None, expected_input: int | None = None,
              expected_output: int | None = None) -> dict:
    """Evaluate one concurrency point from its AIPerf aggregate and profiling JSONL."""
    artifact_dir = Path(artifact_dir)
    row = {key: None for key, _ in COLUMNS}
    row.update(experiment=experiment, concurrency=concurrency, status="error")
    preferred = artifact_dir / "profile_export_aiperf.json"
    aggregate = preferred if preferred.is_file() else artifact_dir / "profile_export.json"
    try:
        data = json.loads(aggregate.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("AIPerf 汇总应为 JSON 对象")
        samples, failed, cancelled, skipped = _profiling_records(artifact_dir / "profile_export.jsonl")
    except (OSError, ValueError) as exc:
        row["detail"] = str(exc)
        return row

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

    duration = _milliseconds(data, "benchmark_duration", "avg")
    seconds = duration / 1_000 if duration is not None else None
    aggregate_skips = _count(_metric(data, "skipped_context_overflow_count", "avg"))
    row.update(
        attempted_requests=len(samples) + failed, completed_requests=len(samples),
        failed_requests=failed, cancelled_requests=cancelled,
        context_skips=max(skipped, aggregate_skips or 0),
        duration_seconds=seconds, submission_valid=submission_valid,
        submission_notes=" / ".join(map(str, submission_notes)) or None,
    )
    ttft, ttft_count = _percentiles((sample.ttft for sample in samples), "ttft")
    tpot, tpot_count = _percentiles((sample.tpot for sample in samples), "tpot")
    row.update(ttft, ttft_samples=ttft_count)
    row.update(tpot, tpot_samples=tpot_count)
    row.update(_token_metrics(samples, seconds))
    row.update(_speculation(samples))

    issues = []
    if data.get("is_complete") is False or data.get("was_cancelled") is True:
        issues.append("AIPerf 未完成或取消")
    aggregate_count = _count(_metric(data, "request_count", "avg"))
    aggregate_errors = _count(_metric(data, "error_request_count", "avg"))
    if aggregate_count is not None and aggregate_count != len(samples):
        issues.append(f"汇总成功数 {aggregate_count} 与逐请求成功数 {len(samples)} 不符")
    if aggregate_errors is not None and aggregate_errors != failed:
        issues.append(f"汇总失败数 {aggregate_errors} 与逐请求失败数 {failed} 不符")
    if failed:
        issues.append(f"{failed} 个正式请求失败（其中 {cancelled} 个取消）")
    if expected_requests is not None and row["attempted_requests"] != expected_requests:
        issues.append(f"正式请求数 {row['attempted_requests']}，预期 {expected_requests}")
    for label, attr, expected in (("输入", "input_tokens", expected_input),
                                  ("输出", "output_tokens", expected_output)):
        if expected is not None:
            mismatches = sum(getattr(sample, attr) != expected for sample in samples)
            if mismatches:
                issues.append(f"{mismatches} 个请求的服务端{label}计量缺失或不等于 {expected}")

    tpot_expected = sum(sample.output_tokens not in (0, 1) for sample in samples)
    if not samples or ttft_count != len(samples) or not tpot_count or tpot_count != tpot_expected:
        issues.append(f"延迟样本不足（TTFT {ttft_count}/{len(samples)}，TPOT {tpot_count}/{tpot_expected}）")
    if issues:
        row.update(status="invalid", detail="；".join(issues))
        return row

    limits_exceeded = [
        f"{key.removesuffix('_ms').replace('_', ' ').upper()} {row[key]:,.2f} > {limit:,} ms"
        for key, limit in SLA_MS.items() if row[key] > limit
    ]
    row.update(status="fail" if limits_exceeded else "pass",
               detail="；".join(limits_exceeded) or "六项延迟均在 SLA 范围内")
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
.badge{font-weight:700}.pass{color:#07674b}.fail,.error{color:#a32c34}.invalid{color:#82520f}
@media(max-width:560px){main{padding:16px}figure{padding:10px}.heading{gap:6px}}
"""


def _chart(title, unit, rows, series, *, thresholds=(), minimum_ceiling=0):
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

    def x(index):
        return left + width / 2 if len(rows) == 1 else left + index * width / (len(rows) - 1)

    def y(value):
        return bottom - height * value / ceiling

    chart = [heading, f'<svg viewBox="0 0 546 224" role="img" aria-label="{escape(title)}">']
    for step in range(5):
        value = step * ceiling / 4
        offset = y(value)
        chart.append(
            f'<path class="gridline" d="M{left} {offset:.1f}H{right}"/>'
            f'<text x="{left - 8}" y="{offset + 4:.1f}" text-anchor="end">{value:,.3g}</text>'
        )
    chart.append(f'<path class="axis" fill="none" d="M{left} {top}V{bottom}H{right}"/>')
    for index, row in enumerate(rows):
        if len(rows) <= 10 or index in (0, len(rows) - 1) or index % math.ceil(len(rows) / 8) == 0:
            chart.append(
                f'<text x="{x(index):.1f}" y="{bottom + 15}" text-anchor="middle">'
                f'C{escape(str(row["concurrency"]))}</text>'
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
            chart.append(
                f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="4" fill="{color}">'
                f'<title>C{escape(str(row["concurrency"]))} {escape(label)} {value:,.4g} {escape(unit)}'
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


def _table(rows):
    fields = COLUMNS[1:]
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend(f'<th scope="col">{escape(label)}</th>' for _, label in fields)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in fields:
            value = row.get(key)
            if key == "status":
                label = STATUS_NAMES.get(value, "—")
                css = value if value in STATUS_NAMES else ""
                parts.append(f'<td><span class="badge {css}">{label}</span></td>')
            elif value is None:
                parts.append("<td>—</td>")
            elif type(value) is bool:
                parts.append("<td>是</td>" if value else "<td>否</td>")
            elif type(value) is float:
                parts.append(f"<td>{value:,.5g}</td>")
            else:
                parts.append(f"<td>{escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _section(experiment, rows):
    name = EXPERIMENT_NAMES.get(experiment, "")
    best = max((row["concurrency"] for row in rows if row.get("status") == "pass"), default=None)
    capacity = f"已测最大达标并发 {best}" if best is not None else "已测档位暂无达标"
    html = [
        f'<section id="{escape(experiment)}"><div class="heading"><h2>{escape(experiment)} '
        f'{escape(name)}</h2><p>{capacity}</p></div><div class="grid">'
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
    cache_reference = ((90, OBSERVED),) if experiment == "P03" else ()
    html.append(_chart("KVCache 命中率", "%", rows, (
        ("kv_hit_percent", "命中率", OBSERVED, 1),
    ), thresholds=cache_reference, minimum_ceiling=100))
    html.append(_chart("推测解码接受长度", "Token", rows, (
        ("spec_acceptance_length", "平均接受长度", COLORS[0], 1),
    )))
    html.append(_chart("推测解码草稿接受率", "%", rows, (
        ("spec_acceptance_percent", "草稿接受率", COLORS[1], 1),
    ), minimum_ceiling=100))
    html.append(_chart("正式请求", "请求", rows, (
        ("completed_requests", "成功", OBSERVED, 1),
        ("failed_requests", "失败", "#a32c34", 1),
        ("context_skips", "上下文跳过", COLORS[1], 1),
    )))
    html.extend(("</div>", _table(rows), "</section>"))
    return "".join(html)


def write_reports(run_results: Path, rows: list[dict]) -> None:
    """Write sortable-source CSV and an offline HTML dashboard for a run."""
    run_results = Path(run_results)
    run_results.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["experiment"], row["concurrency"]))
    csv_tmp = run_results / "summary.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key, _ in COLUMNS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    csv_tmp.replace(run_results / "summary.csv")
    experiments = sorted({row["experiment"] for row in ordered})
    navigation = "".join(f'<a href="#{escape(scene)}">{escape(scene)}</a>' for scene in experiments)
    sections = "".join(
        _section(scene, [row for row in ordered if row["experiment"] == scene])
        for scene in experiments
    )
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Benchmark · {escape(run_results.name)}</title><style>{STYLE}</style><main>'
        f'<header><h1>Benchmark 压测结果</h1><p>{escape(run_results.name)}</p>'
        f'<nav>{navigation}<a href="summary.csv">下载 CSV</a></nav></header>'
        '<p class="limits">SLA 阈值按 P50、P90、P99 排列。TTFT 为 5 / 10 / 30 秒，'
        'TPOT 为 30 / 50 / 80 毫秒。延迟图的虚线对应同色分位数的阈值。'
        'P03 缓存图的虚线标记 90% 观察目标。</p>'
        f'{sections or "<p>尚未记录任何并发档位。</p>"}</main></html>'
    )
    html_tmp = run_results / "report.html.tmp"
    html_tmp.write_text(html, encoding="utf-8")
    html_tmp.replace(run_results / "report.html")
