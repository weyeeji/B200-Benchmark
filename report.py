"""Render a self-contained, printable benchmark report from flat summary rows."""

import math
from collections import defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote

from report_charts import CHART_CSS, profile_charts
from report_trends import TREND_CSS, output_trend

SCHEMA = 1
NA = "N/A"
STATS = ("mean", "p50", "p90", "p95", "p99")
LATENCIES = {"ttft": "TTFT", "tpot": "TPOT", "itl": "平均 ITL", "e2el": "E2EL"}
STATUS = {"complete": "完成", "incomplete": "不完整", "pilot_only": "预实验"}
PROTOCOL = (
    "dataset_formal_sha256", "dataset_warmup_sha256", "formal_target_requests",
    "warmup_target_requests", "duration_cap_s", "request_timeout_s", "per_chunk_usage", "model", "gpu_count",
)
CSS = """
:root { color-scheme: light; font-family: system-ui, "Microsoft YaHei", sans-serif;
  color: #162b2a; background: #edf3f1; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1340px; margin: auto; padding: 28px clamp(14px, 4vw, 52px) 48px; }
header { background: #103936; color: #f4fbf8; border-radius: 14px; padding: 26px;
  border-bottom: 5px solid #64bc9d; }
header p { color: #b9dad1; margin: 8px 0 0; overflow-wrap: anywhere; }
header a { color: #e2f9e8; }
h1 { font-size: clamp(23px, 3vw, 38px); margin: 0; }
h2 { font-size: 20px; margin: 0 0 13px; }
nav, .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 15px; }
.cards div { background: #fff; border: 1px solid #d4e2de; border-radius: 9px; padding: 10px 15px;
  flex: 1 1 130px; font-size: 13px; color: #355651; }
.cards strong { display: block; margin-top: 5px; font-size: 24px; color: #122d29; }
section, details { background: #fff; border: 1px solid #d4e2de; border-radius: 11px;
  margin-top: 16px; padding: 16px; }
details { background: #fafcfb; }
details[open] summary { margin-bottom: 12px; }
summary { cursor: pointer; font-weight: 700; overflow-wrap: anywhere; }
.chart-profile > summary { color: #205d50; font-size: 16px; }
.chart-profile { padding: 12px; margin-top: 12px; scroll-margin-top: 16px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; border-bottom: 1px solid #dde8e4; text-align: right; white-space: nowrap; }
th { color: #37534f; background: #edf5f2; font-weight: 650; }
th:first-child, td:first-child { text-align: left; }
tbody tr:nth-child(even) { background: #f8fbfa; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 5px; font-size: 12px; }
.ok { color: #1f5c3d; background: #dfefe3; }
.review { color: #714507; background: #fff0d3; }
.bad { color: #852e28; background: #f9e1dd; }
.muted { color: #64736e; }
a { color: #155e57; text-underline-offset: 2px; }
p, li { line-height: 1.55; }
small { color: #52645f; }
@media print { :root { background: #fff; } main { max-width: none; padding: 0; }
  section, header { break-inside: avoid; } th, td { padding: 4px; } }
"""


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def formatted(value, digits=2):
    if numeric(value):
        return f"{value:,.{digits}f}" if digits else f"{value:,.0f}"
    return NA if value in (None, "", NA) else escape(str(value))


def table(headers, rows):
    headings = "".join(f'<th scope="col">{escape(name)}</th>' for name in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def point_label(row):
    return "/".join(formatted(row.get(key), 0) for key in ("profile", "concurrency", "repeat"))


def status_badge(row):
    current = row.get("status")
    label = STATUS.get(current, NA)
    if current == "complete":
        kind = "ok" if row.get("valid_for_comparison") else "review"
        if kind == "review":
            label = "完成 · 待核验"
    else:
        kind = "bad" if current == "incomplete" else "review"
    return f'<span class="badge {kind}">{escape(label)}</span>'


def coverage_notes(rows):
    points = defaultdict(list)
    for row in rows:
        points[(row.get("profile"), row.get("concurrency"))].append(row)
    notes = []
    for (profile, concurrency), group in points.items():
        if len({row.get("run_source") for row in group}) < 2:
            continue
        if any(
            len({row.get(key) for row in group}) != 1 or any(row.get(key) in (None, NA) for row in group)
            for key in PROTOCOL
        ):
            notes.append(f"{profile} 并发 {concurrency} 的数据集或运行协议存在差异。")
        if len({row.get("sla_signature") for row in group}) != 1:
            notes.append(f"{profile} 并发 {concurrency} 的 SLA 阈值存在差异。")
    return notes


def overview(rows, combined):
    headers = ["场景 / 并发 / 轮次", "部署"]
    if combined:
        headers.append("运行")
    headers += ["GPU", "状态", "正式成功 / 计划", "稳态样本", "RPS", "输入 tok/s", "输出 tok/s",
                "输出 tok/s/GPU", "稳态 TTFT P95 (ms)", "全量 TTFT P95 (ms)"]
    cells = []
    for row in rows:
        identity = [point_label(row), formatted(row.get("deployment"))]
        if combined:
            identity.append(formatted(row.get("run_source")))
        counts = " / ".join(formatted(row.get(key), 0)
                            for key in ("completed_requests", "formal_target_requests"))
        cells.append(identity + [formatted(row.get("gpu_count"), 0), status_badge(row), counts, formatted(row.get("steady_completed_requests"), 0),
                                 formatted(row.get("rps")), formatted(row.get("input_tps"), 0),
                                 formatted(row.get("output_tps"), 0), formatted(row.get("output_tps_per_gpu"), 0),
                                 formatted(row.get("ttft_p95_ms")), formatted(row.get("formal_ttft_p95_ms"))])
    return table(headers, cells)


def point_details(row, combined):
    latency_rows = []
    for prefix, scope in (("", "稳态完成"), ("formal_", "正式全量")):
        for metric, label in LATENCIES.items():
            sample_key = f"{'formal' if prefix else 'steady'}_{metric}_samples"
            latency_rows.append([scope, label, formatted(row.get(sample_key), 0)] +
                                [formatted(row.get(f"{prefix}{metric}_{stat}_ms")) for stat in STATS])
    latencies = table(["范围", "延迟 (ms)", "样本数", "Mean", "P50", "P90", "P95", "P99"], latency_rows)
    keys = (
        ("实际并发均值", "actual_concurrency_mean"), ("稳态时长 (s)", "steady_seconds"),
        ("Drain 完成请求", "drain_completed_requests"), ("Goodput (req/s)", "goodput_rps"),
        ("同口径缓存输入 (tok/s)", "cached_input_tps"),
        ("同口径未缓存输入 (tok/s)", "uncached_input_tps"),
        ("服务端缓存输入 (tok/s)", "server_cached_input_tps"),
        ("服务端未缓存输入 (tok/s)", "server_uncached_input_tps"),
        ("主机网卡接收 (B/s)", "network_rx_bytes_per_s"), ("主机网卡发送 (B/s)", "network_tx_bytes_per_s"),
        ("GPU 利用率 (%)", "gpu_utilization_pct"),
    )
    figures = table(["指标", "数值"], [(label, formatted(row.get(key))) for label, key in keys])
    checks = [f'缓存清理 {formatted(row.get("cache_clear_status"))}',
              f'Ignore EOS 探针 {formatted(row.get("forced_eos_probe_status"))}',
              f'推测解码 {formatted(row.get("speculative_mode"))}']
    if row.get("issues") not in (None, NA):
        checks.append(formatted(row["issues"]))
    if row.get("comparison_notes") not in (None, NA):
        checks.append(formatted(row["comparison_notes"]))
    title = point_label(row)
    if combined:
        title += f' · {formatted(row.get("deployment"))} · {formatted(row.get("run_source"))}'
    else:
        artifact = Path(row["artifact_dir"]).name if row.get("artifact_dir") else ""
        if artifact:
            checks.append(f'<a href="./{quote(artifact, safe="")}/point.json">测试点 JSON</a>')
    return (f"<details><summary>{title} {status_badge(row)}</summary>"
            f"<p><small>{' · '.join(checks)}</small></p>{latencies}{figures}</details>")


def chart_section(rows, combined):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("profile") or NA)].append(row)
    panels = []
    for index, (profile, matching) in enumerate(sorted(groups.items())):
        matching.sort(key=lambda row: (row.get("concurrency") or 0, str(row.get("deployment") or ""),
                                       str(row.get("run_source") or ""), row.get("repeat") or 0))
        visible = " open" if len(groups) <= 2 or index == 0 else ""
        panels.append(f'<details class="chart-profile"{visible}><summary>{escape(profile)}</summary>'
                      f'{profile_charts(matching, combined=combined, scale_rows=rows)}'
                      f'{output_trend(matching, combined, scale_rows=rows)}</details>')
    note = "<small>同一指标在各场景使用共同刻度。</small>" if len(groups) > 1 else ""
    return f'<section id="charts"><h2>图表</h2>{note}{"".join(panels)}</section>'


def write_report(path, rows, *, combined=False):
    """Write one report atomically. Caller retains ownership of raw artifacts."""
    path = Path(path)
    title = "Serving 多运行对比" if combined else "Serving 压测报告"
    model = "、".join(sorted({str(row["model"]) for row in rows if row.get("model")}))
    counts = (
        ("测试点", len(rows)),
        ("完成", sum(row.get("status") == "complete" for row in rows)),
        ("可对照", sum(row.get("valid_for_comparison") is True for row in rows)),
        ("未完成", sum(row.get("status") == "incomplete" for row in rows)),
    )
    cards = "".join(f"<div>{label}<strong>{value}</strong></div>" for label, value in counts)
    links = '<a href="#charts">图表</a><a href="#table">明细</a>'
    if not combined:
        links += '<a href="summary.csv">CSV</a><a href="summary.json">JSON</a>'
    nav = f"<nav>{links}</nav>"
    notes = coverage_notes(rows) if combined else []
    warnings = ("<section><h2>比较条件</h2><ul>" +
                "".join(f"<li>{escape(note)}</li>" for note in notes) +
                "</ul></section>") if notes else ""
    details = "".join(point_details(row, combined) for row in rows)
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}{CHART_CSS}{TREND_CSS}</style>
<body><main>
<header><h1>{title}</h1><p>{escape(model)}</p>{nav}</header>
<div class="cards">{cards}</div>{warnings}
{chart_section(rows, combined)}
<section id="table"><h2>测试点</h2>{overview(rows, combined)}
<p><small>吞吐使用稳态完成窗。全量延迟包含 drain 完成的请求。ITL 分位以逐请求平均 ITL 计算。</small></p>
</section>
<section><h2>延迟分布与采集口径</h2>{details}</section>
</main></body></html>
"""
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
