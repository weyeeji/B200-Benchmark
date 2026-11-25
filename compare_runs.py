#!/usr/bin/env python3
"""Compare two completed Benchmark runs and write an offline report."""

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA256 = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
EXPERIMENT = re.compile(r"P[0-9]{2}")
METRICS = (
    *((f"ttft_p{p}_ms", f"TTFT P{p}", "ms", None) for p in (50, 90, 99)),
    *((f"tpot_p{p}_ms", f"TPOT P{p}", "ms", None) for p in (50, 90, 99)),
    ("input_tpm", "输入 TPM", "token/min", None),
    ("output_tpm", "输出 TPM", "token/min", None),
    ("total_tpm", "总 TPM", "token/min", None),
    ("kv_hit_percent", "KVCache 命中率", "%", "kv_source"),
    ("spec_acceptance_length", "推测解码接受长度", "token/轮", "spec_source"),
    ("spec_acceptance_percent", "推测解码草稿接受率", "%", "spec_source"),
)
TELEMETRY_METRICS = tuple(
    (f"{family}_{field}", f"{label} · {family_label}", unit, f"{family}_scope")
    for family, family_label in (("window", "近似正式阶段"), ("whole_run", "整个进程"))
    for field, label, unit, source in METRICS if source
)
WORKLOAD_FLAGS = (
    "--endpoint-type", "--streaming", "--use-server-token-count",
    "--custom-dataset-type", "--dataset-sampling-strategy", "--request-count",
    "--concurrency", "--extra-inputs", "--benchmark-duration",
    "--benchmark-grace-period", "--agentic-warmup-grace-period",
    "--warmup-requests-per-lane", "--cache-bust", "--max-context-length",
)
PROVENANCE = (
    ("code_commit", "代码提交"), ("code_dirty", "未提交改动"),
    ("model", "模型"), ("url", "服务地址"), ("deployment", "部署配置"),
    ("routing", "会话路由与 AgentX 设置"), ("tokenizer_sha256", "分词器指纹"),
    ("aiperf_version", "AIPerf 版本"), ("python_version", "Python 版本"),
    ("seed", "随机种子"), ("dataset_nonce", "合成数据标识"),
    ("agentx_dataset", "AgentX 数据集"),
)
LABELS = {
    "comparable": "可对比", "unverified": "证据不完整",
    "noncomparable": "条件不匹配或运行无效", "baseline_only": "仅基线",
    "candidate_only": "仅候选", "missing": "无测量结果",
}
SIDES = ("baseline", "candidate")


@dataclass
class Run:
    directory: Path
    log_directory: Path
    manifest_path: Path | None
    manifest: dict
    points: dict
    rows: dict

    @property
    def name(self):
        return self.manifest.get("run_id") or self.directory.name


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"预期 JSON 对象：{path}")
    return data


def point_key(record):
    experiment = record.get("experiment")
    if not isinstance(experiment, str) or not EXPERIMENT.fullmatch(experiment):
        raise ValueError(f"无效的实验编号：{experiment!r}")
    concurrency = int(record["concurrency"])
    repeat = int(record.get("repeat") or 1)
    if concurrency < 1 or repeat < 1:
        raise ValueError(f"并发和重复编号必须为正数：{experiment} C{concurrency} R{repeat}")
    return experiment, concurrency, repeat


def index_points(records, path):
    indexed = {}
    for record in records:
        try:
            key = point_key(record)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"{path} 的测点不合法：{exc}") from exc
        if key in indexed:
            raise ValueError(f"{path} 含重复测点 {key}")
        indexed[key] = record
    return indexed


def load_run(value):
    directory = Path(value).expanduser()
    if not directory.exists() and directory.name == value:
        directory = ROOT / "results" / directory
    directory = directory.resolve(strict=True)
    summary_path = directory / "summary.csv"
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"experiment", "concurrency", "status"} <= set(reader.fieldnames or ()):
            raise ValueError(f"缺少测点或状态字段：{summary_path}")
        rows = index_points(reader, summary_path)

    project_logs = directory.parent.parent / "logs"
    log_directory = project_logs / directory.name
    locations = (
        directory / "manifest.json", directory / "run.json", log_directory / "run.json",
    )
    manifest_path = next((path for path in locations if path.is_file()), None)
    manifest = read_json(manifest_path) if manifest_path is not None else {}
    if manifest_path is not None and manifest_path == locations[-1]:
        stored_id = manifest.get("run_id")
        if stored_id and stored_id != directory.name:
            raise ValueError(f"运行目录与旧版日志的运行 ID 不匹配：{manifest_path}")
    points = index_points(manifest.get("points", ()), manifest_path)
    return Run(directory, log_directory, manifest_path, manifest, points, rows)


def local_directory(root, value):
    if not isinstance(value, str) or not value:
        return None
    path = (root / value).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def hashes(run, key, phase):
    experiment, concurrency, repeat = key
    row = run.rows.get(key, {})
    plan = run.points.get(key, {})
    csv_field = "dataset_sha256" if phase == "formal" else "warmup_dataset_sha256"
    found = {
        value.lower() for data in (row, plan)
        if isinstance(value := data.get(csv_field), str) and SHA256.fullmatch(value)
    }
    explicit_point = row.get("point_dir") or plan.get("point_dir")
    fallback = Path(experiment) / f"C{concurrency}"
    paths = [fallback / f"R{repeat:02d}"]
    if repeat == 1:
        paths.append(fallback)
    for root in (run.directory, run.log_directory):
        directories = [local_directory(root, explicit_point)]
        directories.append(local_directory(root, row.get("log_dir") or plan.get("log_dir")))
        directories.extend(root / path for path in paths)
        for directory in dict.fromkeys(directories):
            if directory is None:
                continue
            meta = directory / f"{phase}.dataset.json"
            if meta.is_file():
                value = read_json(meta).get("sha256")
                if isinstance(value, str) and SHA256.fullmatch(value):
                    found.add(value.lower())
    return found


def command_controls(point):
    controls = {}
    for phase in point.get("phases", ()):
        args = phase.get("argv", ())
        flags = {}
        for index, token in enumerate(args):
            arg, separator, value = token.partition("=")
            if arg not in WORKLOAD_FLAGS:
                continue
            if not separator:
                following = args[index + 1] if index + 1 < len(args) else ""
                value = following if following and not following.startswith("--") else True
            if arg == "--extra-inputs" and isinstance(value, str) and value.startswith("{"):
                value = json.loads(value)
            flags[arg] = value
        if flags:
            controls[phase["name"]] = flags
    return controls


def valid_row(row):
    if row.get("status") not in ("pass", "fail") or row.get("validity") in ("invalid", "incomplete"):
        return False
    if str(row.get("submission_valid", "")).strip().lower() in ("false", "0", "no"):
        return False
    if str(row.get("grace_period_timeout_triggered", "")).strip().lower() in ("true", "1", "yes"):
        return False
    return not any((number(row.get(field)) or 0) > 0 for field in ("failed_requests", "cancelled_requests"))


def check_controls(baseline, candidate, key):
    b, c = baseline.manifest, candidate.manifest
    experiment = key[0]
    differences, gaps, legacy_gaps, dataset_matches = [], [], [], []
    facts = {}

    def check(label, left, right, missing=gaps, *, available=None):
        if available is None:
            available = left is not None and right is not None
        if not available:
            missing.append(label)
        elif left != right:
            differences.append(label)

    for field, label in (
        ("model", "模型"), ("aiperf_version", "AIPerf 版本"),
        ("python_version", "Python 版本"), ("seed", "随机种子"),
        ("request_timeout_seconds", "单请求超时"),
    ):
        check(label, b.get(field), c.get(field))
    for field, label in (("tokenizer_sha256", "分词器指纹"), ("routing", "会话路由与 AgentX 设置")):
        check(label, b.get(field), c.get(field), legacy_gaps)
    for field, label in (("code_commit", "代码提交"), ("deployment", "部署配置")):
        if not b.get(field) or not c.get(field):
            legacy_gaps.append(label)
    if b.get("code_dirty") is True or c.get("code_dirty") is True:
        legacy_gaps.append("工作区有未提交改动")

    point_b, point_c = baseline.points.get(key, {}), candidate.points.get(key, {})
    settings_b, settings_c = point_b.get("settings"), point_c.get("settings")
    if isinstance(settings_b, dict) and isinstance(settings_c, dict):
        differences.extend(
            f"测点参数 {name}" for name in sorted(settings_b.keys() | settings_c.keys())
            if settings_b.get(name) != settings_c.get(name) or (name in settings_b) != (name in settings_c)
        )
    else:
        gaps.append("测点参数")
    check("AIPerf 请求参数", command_controls(point_b) or None, command_controls(point_c) or None)

    if experiment in ("P01", "P02"):
        for phase, label in (("warmup", "预热数据 SHA-256"), ("formal", "正式数据 SHA-256")):
            left, right = hashes(baseline, key, phase), hashes(candidate, key, phase)
            facts[f"baseline_{phase}_dataset_sha256"] = " | ".join(sorted(left))
            facts[f"candidate_{phase}_dataset_sha256"] = " | ".join(sorted(right))
            if len(left) > 1 or len(right) > 1:
                differences.append(f"{label}记录冲突")
            else:
                check(label, next(iter(left), None), next(iter(right), None))
                if left and left == right:
                    dataset_matches.append(label)
    elif experiment == "P03":
        check("上下文过滤", b.get("max_context_length"), c.get("max_context_length"),
              available="max_context_length" in b and "max_context_length" in c)
        data_b, data_c = (run.get("agentx_dataset") or {} for run in (b, c))
        revision_b, revision_c = data_b.get("revision"), data_c.get("revision")
        facts.update(baseline_agentx_revision=revision_b, candidate_agentx_revision=revision_c,
                     baseline_agentx_trace_sha256=data_b.get("trace_sha256"),
                     candidate_agentx_trace_sha256=data_c.get("trace_sha256"))
        check("AgentX 数据集版本", revision_b, revision_c)
        check("AgentX 数据集内容 SHA-256", data_b.get("trace_sha256"),
              data_c.get("trace_sha256"), legacy_gaps)
        if revision_b and revision_b == revision_c:
            dataset_matches.append("AgentX 数据集版本")
        deploy_b, deploy_c = (run.get("deployment") or {} for run in (b, c))
        for field, label in (("cache_start_state", "缓存起点"),):
            state_b = deploy_b.get(field) if isinstance(deploy_b, dict) else None
            state_c = deploy_c.get(field) if isinstance(deploy_c, dict) else None
            check(label, state_b, state_c, legacy_gaps)
    else:
        gaps.append("未知工作负载的专用对比条件")

    return differences, gaps, legacy_gaps, dataset_matches, facts


def number(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def metric_values(field, source, row_b, row_c, show_delta):
    left, right = number(row_b.get(field)), number(row_c.get(field))
    source_b = row_b.get(source) if source else None
    source_c = row_c.get(source) if source else None
    if left is None or right is None:
        return left, right, None, None, "缺少测量"
    if source and (not source_b or not source_c or source_b != source_c):
        return left, right, None, None, "指标来源或统计范围不同或缺失"
    if not show_delta:
        return left, right, None, None, "测点对比条件未通过"
    delta = right - left
    relative = 100 * delta / left if left > 0 and not field.endswith("_percent") else None
    return left, right, delta, relative, ""


def compare_point(baseline, candidate, key):
    experiment, concurrency, repeat = key
    row_b, row_c = baseline.rows.get(key, {}), candidate.rows.get(key, {})
    show_delta = False
    facts = {}
    if not row_b or not row_c:
        if row_b:
            status, detail = "baseline_only", "候选缺少对应测量"
        elif row_c:
            status, detail = "candidate_only", "基线缺少对应测量"
        else:
            status, detail = "missing", "计划有此测点，两次运行均未留下对应测量"
    else:
        differences, gaps, legacy_gaps, matches, facts = check_controls(baseline, candidate, key)
        notes = []
        if not valid_row(row_b):
            differences.append("基线运行无效")
        if not valid_row(row_c):
            differences.append("候选运行无效")
        if differences:
            notes.append("条件不同或未通过有效性检查：" + "、".join(differences))
        if gaps or legacy_gaps:
            notes.append("缺少或无法核验：" + "、".join(gaps + legacy_gaps))
        if matches:
            notes.append("一致：" + "、".join(matches))
        status = "noncomparable" if differences else "unverified" if gaps or legacy_gaps else "comparable"
        detail = "。".join(notes) or "运行有效，已核验条件一致"
        show_delta = not differences and not gaps

    row = dict(experiment=experiment, concurrency=concurrency, repeat=repeat, comparison=status, detail=detail)
    row.update(facts)
    for side, run, source_row in zip(SIDES, (baseline, candidate), (row_b, row_c)):
        row[f"{side}_run_id"] = run.name
        row[f"{side}_code_commit"] = run.manifest.get("code_commit")
        for field in ("status", "validity", "submission_valid", "detail", "kv_source", "spec_source",
                      "window_scope", "whole_run_scope", "completed_requests", "failed_requests",
                      "cancelled_requests", "ttft_samples", "tpot_samples"):
            row[f"{side}_{field}"] = source_row.get(field)
    for field, _, _, source in (*METRICS, *TELEMETRY_METRICS):
        left, right, delta, relative, note = metric_values(field, source, row_b, row_c, show_delta)
        row[f"baseline_{field}"] = left
        row[f"candidate_{field}"] = right
        row[f"delta_{field}"] = delta
        row[f"delta_percent_{field}"] = relative
        row[f"note_{field}"] = note
    return row


def controls_html(baseline, candidate):
    parts = ["<details><summary>运行与部署记录</summary><table>",
             "<thead><tr><th>字段</th><th>基线</th><th>候选</th></tr></thead><tbody>"]
    displayed = (
        ("运行 ID", baseline.name, candidate.name),
        ("元数据文件", baseline.manifest_path, candidate.manifest_path),
        *((label, baseline.manifest.get(field), candidate.manifest.get(field)) for field, label in PROVENANCE),
    )
    for label, left, right in displayed:
        values = (left, right)
        cells = []
        for value in values:
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            cells.append(f"<td><pre>{escape(str(value)) if value is not None else '—'}</pre></td>")
        parts.extend((f"<tr><th>{label}</th>", *cells, "</tr>"))
    parts.append("</tbody></table></details>")
    return "".join(parts)


def format_value(value, *, signed=False):
    if value is None:
        return "—"
    return format(value, "+,.6g" if signed else ",.6g")


def point_html(row):
    change_status = row["comparison"]
    title = f'{row["experiment"]} · C{row["concurrency"]} · R{row["repeat"]:02d}'
    parts = [
        f"<section><h2>{title} <span class='{change_status}'>{LABELS[change_status]}</span></h2>",
        f"<p>{escape(row['detail'])}</p>",
    ]
    for side, label in zip(SIDES, ("基线", "候选")):
        status = row.get(f"{side}_status")
        if status is not None:
            detail = row.get(f"{side}_detail") or ""
            parts.append(f"<p>{label} {escape(status)} · {escape(detail)}</p>")
    parts.append("<table><thead><tr><th>指标</th><th>基线</th><th>候选</th><th>差值</th><th>变化率</th>"
                 "<th>来源或说明</th></tr></thead><tbody>")
    for field, label, unit, source in (*METRICS, *TELEMETRY_METRICS):
        if source in ("window_scope", "whole_run_scope") and all(row[f"{side}_{field}"] is None for side in SIDES):
            continue
        left = format_value(row[f"baseline_{field}"])
        right = format_value(row[f"candidate_{field}"])
        delta = format_value(row[f"delta_{field}"], signed=True)
        relative = format_value(row[f"delta_percent_{field}"], signed=True)
        if delta != "—" and unit == "%":
            delta += " pp"
        if relative != "—":
            relative += "%"
        note = row[f"note_{field}"]
        if source:
            left_source, right_source = (row.get(f"{side}_{source}") for side in SIDES)
            source_label = left_source if left_source == right_source else (
                f"基线 {left_source or '—'} / 候选 {right_source or '—'}"
            )
            note = " · ".join(filter(None, (source_label, note)))
        parts.append(f"<tr><th>{label} <small>{unit}</small></th><td>{left}</td><td>{right}</td>"
                     f"<td>{delta}</td><td>{relative}</td><td>{escape(note)}</td></tr>")
    parts.append("</tbody></table></section>")
    return "".join(parts)


def write_comparison(directory, baseline, candidate, rows):
    directory.mkdir(parents=True, exist_ok=False)
    columns = list(dict.fromkeys(field for row in rows for field in row))
    with (directory / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Benchmark 跨运行对比</title><style>'
        'body{margin:24px auto;padding:0 18px;max-width:1000px;color:#16283a;background:#f3f7fa;'
        'font:15px/1.45 system-ui,sans-serif}h1,h2{line-height:1.2}section,details{background:#fff;'
        'border:1px solid #ccd9e4;border-radius:10px;padding:14px;margin:18px 0;overflow:auto}'
        'table{width:100%;border-collapse:collapse;margin-top:8px}th,td{padding:6px 8px;'
        'border-bottom:1px solid #dfe8ef;text-align:left}td{font-variant-numeric:tabular-nums}'
        'span{font-size:.7em;color:#82520f}.comparable{color:#106845}.noncomparable{color:#a32c34}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}small{color:#546678}'
        '</style><main><h1>Benchmark 跨运行对比</h1>'
        f'<p>基线 {escape(str(baseline.name))} · 候选 {escape(str(candidate.name))} · '
        '<a href="comparison.csv">下载 CSV</a></p>'
        '<p>差值为候选减基线。命中率和接受率的差值使用百分点（pp）。'
        '证据不完整时，已匹配工作负载的差值仅用于描述观测结果。'
        '指标来源或统计范围不一致时保留原值。</p>'
        + controls_html(baseline, candidate)
        + "".join(map(point_html, rows))
        + "</main></html>"
    )
    (directory / "comparison.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="baseline run directory or ID under results")
    parser.add_argument("candidate", help="candidate run directory or ID under results")
    parser.add_argument("--output", type=Path, help="new directory for comparison.csv and comparison.html")
    args = parser.parse_args()
    try:
        baseline, candidate = (load_run(value) for value in (args.baseline, args.candidate))
        keys = sorted(baseline.rows.keys() | candidate.rows.keys() | baseline.points.keys() | candidate.points.keys())
        if not keys:
            raise ValueError("输入运行没有测点")
        rows = [compare_point(baseline, candidate, key) for key in keys]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = args.output or ROOT / "results" / "comparisons" / timestamp
        write_comparison(output, baseline, candidate, rows)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        print(f"compare_runs: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison: {output.resolve()}")
    for status, label in LABELS.items():
        count = sum(row["comparison"] == status for row in rows)
        if count:
            print(f"{label} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
