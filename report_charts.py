"""Static report charts rendered from one scenario's summary rows."""

from html import escape
from math import ceil, floor, isfinite, log10

NA = "N/A"
PERCENTILES = ("p50", "p95", "p99")
PHASES = (
    ("startup", "启动"),
    ("steady", "稳态"),
    ("drain", "Drain"),
)

CHART_CSS = """
.report-chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 310px), 1fr));
  gap: 14px; margin: 12px 0 20px; font-variant-numeric: tabular-nums; }
.report-chart-grid figure { min-width: 0; margin: 0; padding: 15px;
  border: 1px solid #ccdfd9; border-radius: 12px; background: #f5faf8; color: #173732; }
.report-chart-grid h4 { margin: 0 0 3px; font-size: 16px; }
.report-chart-scope, .report-chart-axis, .report-chart-meta { color: #405e56; font-size: 12px; }
.report-chart-scope { display: block; line-height: 1.45; }
.report-chart-axis, .report-chart-values { display: flex; flex-wrap: wrap;
  justify-content: space-between; gap: 5px 9px; }
.report-chart-axis { margin-top: 13px; }
.report-chart-items { list-style: none; margin: 2px 0 0; padding: 0;
  border-top: 1px solid #d8e6e2; }
.report-chart-items > li { padding: 9px 0; overflow-wrap: anywhere; border-bottom: 1px solid #d8e6e2; }
.report-chart-items > li:last-child { border-bottom: 0; padding-bottom: 0; }
.report-chart-point { display: flex; justify-content: space-between; flex-wrap: wrap;
  gap: 4px 8px; align-items: baseline; font-size: 13px; line-height: 1.4; }
.report-chart-point strong { color: #153b34; }
.report-chart-meta { display: block; margin: 3px 0; }
.report-chart-state { padding: 1px 6px; border-radius: 4px; font-size: 11px; white-space: nowrap; }
.report-chart-state--ok { color: #1f5c3d; background: #dcefe2; }
.report-chart-state--review { color: #674208; background: #f8e9c7; }
.report-chart-state--bad { color: #852e28; background: #f9e1dd; }
.report-chart-track { position: relative; height: 9px; margin: 6px 0 4px;
  border-radius: 5px; background: #e0ece8; }
.report-chart-track::before { content: ""; position: absolute; inset: 0 auto 0 50%;
  border-left: 1px solid #b7cec7; }
.report-chart-bar, .report-chart-stack span { display: block; min-width: 0; height: 100%; width: var(--chart-width); }
.report-chart-bar { position: relative; border-radius: inherit; background: #16816d; }
.report-chart-empty { min-height: 9px; margin: 6px 0 4px; }
.report-chart-track--tail { height: 4px; margin: 12px 4px 10px; }
.report-chart-dot { position: absolute; left: var(--chart-position); top: 50%;
  width: 9px; height: 9px; border: 1px solid #f5faf8; transform: translate(-50%, -50%); }
.report-chart-p50 { background: #16816d; color: #19604e; border-radius: 50%; }
.report-chart-p95 { background: #5654a1; color: #4e4b95; border-radius: 2px; }
.report-chart-p99 { background: #a84916; color: #944315; transform: translate(-50%, -50%) rotate(45deg); }
.report-chart-values { justify-content: flex-start; font-size: 12px; }
.report-chart-values span { white-space: nowrap; }
.report-chart-values b { color: #233e38; font-weight: 600; }
.report-chart-key { display: inline-block; width: 7px; height: 7px; margin: 0 3px 1px 0; vertical-align: middle; }
.report-chart-key.report-chart-p99 { transform: rotate(45deg); }
.report-chart-stack { display: flex; overflow: hidden; }
.report-chart-stack span { position: relative; }
.report-chart-startup { background: #547b75; }
.report-chart-steady { background: #16816d; }
.report-chart-drain { background: #a84916; }
@media print {
  .report-chart-grid { display: block; }
  .report-chart-grid figure { break-inside: avoid; margin-bottom: 9px; }
  .report-chart-track, .report-chart-track span, .report-chart-key {
    print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
"""


def measurement(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if value >= 0 and isfinite(value) else None
    except OverflowError:
        return None


def display(value, digits=2):
    if value is None:
        return NA
    if 0 < value < 10 ** -digits:
        return f"{value:.2g}"
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{value:,.0f}"


def count(value):
    value = measurement(value)
    return int(value) if value is not None and int(value) == value else None


def axis_limit(values):
    peak = max(values)
    if peak == 0:
        return 1
    if peak < 1e-9 or peak > 1e12:
        return peak
    step = 10 ** floor(log10(peak))
    for factor in (1, 2, 2.5, 5, 10):
        candidate = step * factor
        if candidate >= peak:
            return candidate
    return ceil(peak)


def fractional_width(value, limit):
    return f"{value / limit * 100:.8f}%"


def text(value):
    return str(value) if value is not None and value != "" else NA


def row_labels(row, combined):
    label = f'并发 {text(row.get("concurrency"))} · 第 {text(row.get("repeat"))} 轮'
    extra = f'{text(row.get("deployment"))} · {text(row.get("run_source"))}' if combined else ""
    return label, extra


def status(row):
    code = row.get("status")
    if code == "complete":
        return ("ok", "完成") if row.get("valid_for_comparison") is True else ("review", "完成 · 待核验")
    if code == "pilot_only":
        return "review", "预实验"
    if code == "incomplete":
        return "bad", "不完整"
    return "review", NA


def chart_row(row, combined, value_label, drawing, *, samples=None, tooltip=None):
    label, extra = row_labels(row, combined)
    kind, state = status(row)
    meta = " · ".join(part for part in (extra, samples) if part)
    details = f'<span class="report-chart-meta">{escape(meta)}</span>' if meta else ""
    title = " · ".join(part for part in (label, extra, state, samples, tooltip) if part)
    return (
        f'<li title="{escape(title, quote=True)}"><div class="report-chart-point">'
        f'<strong>{escape(label)}</strong><span class="report-chart-state report-chart-state--{kind}">'
        f'{state}</span></div>{details}{drawing}'
        f'<div class="report-chart-values">{value_label}</div></li>'
    )


def tile(title, scope, limit, rows, *, axis_digits=2):
    high = display(limit, axis_digits)
    return (
        f'<figure><figcaption><h4>{escape(title)}</h4><span class="report-chart-scope">'
        f'{escape(scope)}</span></figcaption><div class="report-chart-axis" aria-hidden="true">'
        f'<span>0</span><span>{escape(high)}</span></div>'
        f'<ul class="report-chart-items">{"".join(rows)}</ul></figure>'
    )


def bar_tile(rows, combined, title, metric, scope, unit, *, scale_rows, sample_key=None, fixed_limit=None):
    values = [measurement(row.get(metric)) for row in rows]
    if fixed_limit is not None:
        values = [value if value is None or value <= fixed_limit else None for value in values]
    available = [value for value in values if value is not None]
    if not available:
        return ""
    limits = [value for row in scale_rows if (value := measurement(row.get(metric))) is not None]
    limit = fixed_limit if fixed_limit is not None else axis_limit(available + limits)
    plotted = []
    for row, value in zip(rows, values):
        shown = display(value)
        label = f"{shown} {unit}" if value is not None else NA
        drawing = '<div class="report-chart-empty" aria-hidden="true"></div>'
        if value is not None:
            width = fractional_width(value, limit)
            drawing = (
                '<div class="report-chart-track" aria-hidden="true">'
                f'<span class="report-chart-bar" style="--chart-width:{width}"></span></div>'
            )
        samples = f'样本 {display(count(row.get(sample_key)), 0)}' if sample_key else None
        plotted.append(chart_row(row, combined, f"<b>{escape(label)}</b>", drawing,
                                 samples=samples, tooltip=label))
    return tile(title, f"{scope} · {unit}", limit, plotted)


def latency_tile(rows, combined, title, metric, scale_rows):
    values = []
    for row in rows:
        row_values = [measurement(row.get(f"formal_{metric}_{percentile}_ms")) for percentile in PERCENTILES]
        values.append([value / 1000 if value is not None else None for value in row_values])
    available = [value for row_values in values for value in row_values if value is not None]
    if not available:
        return ""
    limits = [value / 1000 for row in scale_rows for percentile in PERCENTILES
              if (value := measurement(row.get(f"formal_{metric}_{percentile}_ms"))) is not None]
    limit = axis_limit(available + limits)
    plotted = []
    for row, row_values in zip(rows, values):
        dots, labels, tooltips = [], [], []
        for percentile, value in zip(PERCENTILES, row_values):
            shown = display(value, 3)
            if value is not None:
                position = fractional_width(value, limit)
                dots.append(f'<span class="report-chart-dot report-chart-{percentile}" '
                            f'style="--chart-position:{position}"></span>')
            key = f'<i class="report-chart-key report-chart-{percentile}" aria-hidden="true"></i>'
            name = percentile.upper()
            labels.append(f'<span>{key}{name} <b>{escape(shown)}</b></span>')
            tooltips.append(f"{name} {shown} s" if value is not None else f"{name} {NA}")
        drawing = (
            f'<div class="report-chart-track report-chart-track--tail" aria-hidden="true">{"".join(dots)}</div>'
            if dots else '<div class="report-chart-empty" aria-hidden="true"></div>'
        )
        samples = f'样本 {display(count(row.get(f"formal_{metric}_samples")), 0)}'
        plotted.append(chart_row(row, combined, "".join(labels), drawing, samples=samples,
                                 tooltip=" · ".join(tooltips)))
    return tile(title, "正式全量成功请求，包含 drain · s", limit, plotted, axis_digits=3)


def completion_parts(row):
    total = count(row.get("attempted_requests"))
    parts = [count(row.get(f"{phase}_attempts")) for phase, _ in PHASES]
    return parts if total is not None and None not in parts and sum(parts) == total else None


def completion_tile(rows, combined, scale_rows):
    series = [completion_parts(row) for row in rows]
    totals = [sum(parts) for parts in series if parts is not None]
    if not totals:
        return ""
    limit = axis_limit(totals + [sum(parts) for row in scale_rows
                                 if (parts := completion_parts(row)) is not None])
    plotted = []
    for row, parts in zip(rows, series):
        if parts is None:
            plotted.append(chart_row(row, combined, NA, '<div class="report-chart-empty" aria-hidden="true"></div>'))
            continue
        bars, labels, tooltips = [], [], []
        for (phase, name), value in zip(PHASES, parts):
            width = fractional_width(value, limit)
            bars.append(f'<span class="report-chart-{phase}" style="--chart-width:{width}"></span>')
            shown = display(value, 0)
            labels.append(f'<span><i class="report-chart-key report-chart-{phase}" aria-hidden="true"></i>'
                          f'{name} <b>{shown}</b></span>')
            tooltips.append(f"{name} {shown}")
        drawing = f'<div class="report-chart-track report-chart-stack" aria-hidden="true">{"".join(bars)}</div>'
        plotted.append(chart_row(row, combined, "".join(labels), drawing, tooltip=" · ".join(tooltips)))
    return tile("请求终止分布", "按终止时刻归类，包含成功、失败及取消 · 请求数", limit, plotted, axis_digits=0)


def profile_charts(profile_rows, *, combined=False, scale_rows=()):
    """Render one scenario's nonempty metric tiles using explicit zero-based scales."""
    rows = tuple(profile_rows)
    tiles = (
        bar_tile(rows, combined, "请求吞吐", "rps", "稳态完成请求", "req/s",
                 scale_rows=scale_rows, sample_key="steady_completed_requests"),
        bar_tile(rows, combined, "输出 Token 吞吐", "output_tps", "稳态完成请求", "tok/s",
                 scale_rows=scale_rows, sample_key="steady_completed_requests"),
        latency_tile(rows, combined, "首 Token 延迟（TTFT）", "ttft", scale_rows),
        latency_tile(rows, combined, "端到端延迟（E2EL）", "e2el", scale_rows),
        completion_tile(rows, combined, scale_rows),
        bar_tile(rows, combined, "GPU 利用率", "gpu_utilization_pct", "Prometheus 稳态采集均值", "%",
                 scale_rows=scale_rows, fixed_limit=100),
    )
    charts = "".join(tiles)
    return f'<div class="report-chart-grid">{charts}</div>' if charts else ""
