"""Show measured output throughput across concurrency levels in the HTML report."""

from collections import defaultdict
from html import escape
import math

TREND_CSS = """
.trend { margin: 12px 0 0; padding: 14px; border: 1px solid #d4e2de; border-radius: 10px; background: #fff; }
.trend h4 { margin: 0 0 4px; font-size: 15px; }
.trend svg { display: block; width: 100%; min-width: 420px; max-height: 330px; background: #fbfdfc; }
.trend text { font: 12px system-ui, sans-serif; fill: #395650; }
.trend figcaption { margin-top: 6px; color: #52645f; font-size: 12px; }
.trend-legend { display: flex; gap: 6px 18px; flex-wrap: wrap; margin: 8px 0 0; padding: 0;
  list-style: none; font-size: 12px; color: #355651; overflow-wrap: anywhere; }
.trend-legend li::before { content: ""; display: inline-block; width: 13px; height: 3px;
  margin: 0 5px 3px 0; background: var(--series); }
"""

COLORS = ("#13705f", "#315a9c", "#9b4752", "#875600", "#6a4a9a")


def axis_limit(maximum):
    if maximum <= 0:
        return 1.0
    # Round each of four intervals upward to a conventional tick size.
    step = max(maximum / 4, 1e-12)
    power = 10.0 ** math.floor(math.log10(step))
    rounded = next((n for n in (1, 2, 2.5, 5, 10) if step <= n * power), 10)
    upper = 4 * rounded * power
    return upper if math.isfinite(upper) else maximum


def series_label(key, combined):
    source, deployment, repeat, kind = key
    label = str(deployment or "当前部署")
    if combined and source:
        label += f" · {source}"
    label += f" · 第 {repeat} 轮"
    return f"{label} · 预实验" if kind == "pilot" else label


def output_trend(rows, combined, *, scale_rows=()):
    series = defaultdict(dict)
    for row in rows:
        concurrency, value = row.get("concurrency"), row.get("output_tps")
        if (row.get("status") in ("complete", "pilot_only") and type(concurrency) is int
                and concurrency > 0 and type(value) in (int, float) and math.isfinite(value) and value >= 0):
            key = (row.get("run_source"), row.get("deployment"), row.get("repeat"), row.get("kind"))
            series[key][concurrency] = (value, row.get("valid_for_comparison") is True)
    measured_levels = {level for values in series.values() for level in values}
    if len(measured_levels) < 2:
        return ""
    levels = sorted({row["concurrency"] for row in rows
                     if type(row.get("concurrency")) is int and row["concurrency"] > 0})

    left, right, top, bottom = 82, 695, 20, 206
    values_for_scale = [value for values in series.values() for value, _ in values.values()]
    values_for_scale.extend(value for row in scale_rows
                            if row.get("status") in ("complete", "pilot_only")
                            and type(value := row.get("output_tps")) in (int, float)
                            and math.isfinite(value) and value >= 0)
    y_max = axis_limit(max(values_for_scale))
    x_by_level = {level: left + (right - left) * index / (len(levels) - 1)
                  for index, level in enumerate(levels)}
    svg = []
    for mark in range(5):
        y = bottom - (bottom - top) * mark / 4
        value = y_max * mark / 4
        tick = f"{value:,.0f}" if value >= 1000 else f"{value:.4g}"
        svg.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#d9e7e2"/>'
                   f'<text x="{left - 9}" y="{y + 4}" text-anchor="end">{tick}</text>')
    for level, x in x_by_level.items():
        svg.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#e5eeeb"/>'
                   f'<text x="{x}" y="{bottom + 20}" text-anchor="middle">{level}</text>')
    svg.append(f'<text x="{(left + right) / 2}" y="{bottom + 39}" text-anchor="middle">并发档位</text>')

    legend = []
    for number, (key, values) in enumerate(series.items()):
        color = COLORS[number % len(COLORS)]
        description = escape(series_label(key, combined))
        legend.append(f'<li style="--series:{color}">{description}</li>')
        previous = None
        for level in levels:
            measurement = values.get(level)
            if measurement is None:
                previous = None
                continue
            value, confirmed = measurement
            x = x_by_level[level]
            y = bottom - (bottom - top) * (value / y_max)
            if previous is not None:
                px, py = previous
                svg.append(f'<line class="trend-segment" x1="{px}" y1="{py}" x2="{x}" y2="{y}"'
                           f' stroke="{color}" stroke-width="2.5"/>')
            fill = color if confirmed else "#fff"
            svg.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="1.5">'
                       f'<title>{description} · 并发 {level} · {value:,.2f} tok/s</title></circle>')
            previous = (x, y)
    return ('<figure class="trend"><h4>并发扩展 · 稳态输出 tok/s</h4>'
            '<div class="scroll"><svg viewBox="0 0 720 254" role="img" aria-label="按并发档位绘制稳态输出吞吐">'
            + "".join(svg) + '</svg></div><ul class="trend-legend">' + "".join(legend)
            + '</ul><figcaption>连线连接同一次运行、同一轮的相邻已测档位。空心点表示预实验或待核验。</figcaption></figure>')
