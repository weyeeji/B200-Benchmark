"""NVIDIA AIPerf adapter for the pinned benchmark protocol."""

import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ARTIFACT = "profile_export"
AFFINITY_ENV = {
    "x-session-id": "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID",
    "x-session-affinity": "AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID",
    "x-smg-routing-key": "AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID",
    "x-dynamo-session-id": "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
    "x-dynamo-parent-session-id": "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
}
SLA_TAGS = {
    "ttft_ms": "time_to_first_token",
    "tpot_ms": "full_response_inter_token_latency",
    "itl_ms": "inter_token_latency",
    "e2el_ms": "request_latency",
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as rows:
        for row in rows:
            if row.strip():
                yield json.loads(row)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_settings(cfg):
    settings = {"ignore_eos": True, "temperature": 0}
    if cfg["chat_template_kwargs"]:
        settings["chat_template_kwargs"] = cfg["chat_template_kwargs"]
    return settings


def aiperf_command(cfg, dataset_file, artifact, count, concurrency, duration=None, monitor=False, *, headers=None):
    command = [
        str(Path(sys.executable).with_name("aiperf")), "profile",
        "--model", cfg["model"], "--url", cfg["api_url"], "--endpoint-type", "chat",
        "--tokenizer", cfg["tokenizer"], "--streaming", "--use-server-token-count",
        "--wait-for-model-timeout", "0",
        "--input-file", str(dataset_file), "--custom-dataset-type", "single_turn",
        "--dataset-sampling-strategy", "sequential", "--no-allow-dataset-wrap",
        "--random-seed", str(cfg["seed"]), "--request-count", str(count),
        "--concurrency", str(concurrency),
        "--request-timeout-seconds", str(cfg["request_timeout_s"]),
        "--artifact-dir", str(artifact), "--profile-export-prefix", ARTIFACT,
        "--export-level", "raw", "--no-export-outputs-json", "--no-auto-plot", "--ui", "none",
        "--extra-inputs", json.dumps(request_settings(cfg), separators=(",", ":")),
    ]
    if headers:
        command += ["--header", json.dumps(headers, separators=(",", ":"))]
    if cfg["use_legacy_max_tokens"]:
        command.append("--use-legacy-max-tokens")
    if cfg["per_chunk_usage"]:
        command.append("--per-chunk-usage")
    if duration is not None:
        command += ["--benchmark-duration", str(duration),
                    "--benchmark-grace-period", str(cfg["request_timeout_s"])]
    urls = cfg["prometheus"].get("urls", []) if monitor else []
    if urls:
        command += ["--server-metrics", *urls, "--server-metrics-formats", "json", "csv", "jsonl"]
    else:
        command.append("--no-server-metrics")
    command.append("--no-gpu-telemetry")
    if monitor and cfg["sla"]:
        thresholds = {}
        for key, value in cfg["sla"].items():
            tag = SLA_TAGS[key]
            thresholds[tag] = min(value, thresholds.get(tag, value))
        command += ["--goodput", " ".join(f"{tag}:{value}" for tag, value in thresholds.items())]
    return command


def aiperf_env(key_var, headers=None):
    env = os.environ.copy()
    key = env.pop(key_var, None) if key_var else None
    for name in ("OPENAI_API_KEY", "AIPERF_INJECTED_API_KEY", "AIPERF_INJECTED_HEADERS",
                 "AIPERF_INJECTED_ENDPOINT_URLS", "AIPERF_INJECTED_DATASET_URLS"):
        env.pop(name, None)
    # Keep the standard AIPerf routing defaults. Explicit calibration headers
    # must take precedence over generated headers with the same name.
    env.update({variable: str(header == "x-session-affinity").lower()
                for header, variable in AFFINITY_ENV.items()})
    for name in headers or {}:
        if variable := AFFINITY_ENV.get(name.lower()):
            env[variable] = "false"
    if key:
        env["OPENAI_API_KEY"] = key
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def run_aiperf(cfg, dataset_file, artifact, count, concurrency, duration=None, monitor=False, *, headers=None):
    artifact.mkdir()
    command = aiperf_command(cfg, dataset_file, artifact, count, concurrency, duration, monitor, headers=headers)
    env = aiperf_env(cfg["api_key_env"], headers)
    if env.get("OPENAI_API_KEY"):
        # The profile parent resolves authored config, then injects the key into
        # its workers. Supplying the worker injection variable here is ignored.
        auth_config = artifact / "aiperf-auth-config.json"
        write_json(auth_config, {"schemaVersion": "2.0", "benchmark": {
            "model": cfg["model"],
            "endpoint": {"url": cfg["api_url"], "type": "chat", "api_key": "${OPENAI_API_KEY}"},
            "dataset": {"type": "file", "path": str(dataset_file), "format": "single_turn"},
            "phases": {"type": "concurrency", "requests": count, "concurrency": concurrency},
        }})
        command += ["--config", str(auth_config)]

    status = {"started_utc": utc_now(), "command": command,
              "target_requests": count, "concurrency": concurrency, "duration_cap_s": duration}
    write_json(artifact / "invocation.json", status)
    try:
        with (artifact / "stdout.log").open("wb") as out, (artifact / "stderr.log").open("wb") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                     start_new_session=True)
            try:
                status["returncode"] = child.wait()
            except KeyboardInterrupt:
                terminate_group(child, signal.SIGINT)
                try:
                    status["returncode"] = child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    terminate_group(child, signal.SIGKILL)
                    status["returncode"] = child.wait()
                status["interrupted"] = True
                raise
    finally:
        status["finished_utc"] = utc_now()
        write_json(artifact / "invocation.json", status)
    return status["returncode"]


def terminate_group(child, sig):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def verify_dispatch(artifact, dataset_file):
    records = artifact / f"{ARTIFACT}.jsonl"
    if not records.exists():
        return {"status": "no_records", "recorded": 0, "last_dispatch_s": None}
    issued = []
    for record in jsonl(records):
        meta = record.get("metadata") or {}
        issued.append((meta.get("session_num"), meta.get("conversation_id"),
                       meta.get("credit_issued_ns")))
    result = {"status": "unverified_dispatch", "recorded": len(issued), "last_dispatch_s": None}
    if not issued or any(type(num) is not int or not isinstance(cid, str)
                         or type(ts) is not int or ts <= 0 for num, cid, ts in issued):
        return result
    issued.sort(key=lambda item: item[0])
    planned = (row["session_id"] for row in jsonl(dataset_file))
    matched = all(num == index and conversation_id == next(planned, None)
                  for index, (num, conversation_id, _) in enumerate(issued))
    timestamps = [item[2] for item in issued]
    if not matched or any(previous > current for previous, current in zip(timestamps, timestamps[1:])):
        return result | {"status": "dispatch_mismatch"}
    elapsed = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else None
    return result | {"status": "ordered", "last_dispatch_s": elapsed}


def clear_cache(cfg, artifact, profile, concurrency, repeat):
    hook = cfg["cache_hook"]
    result = {"status": "unverified", "gpu": "unverified", "cpu": "unverified",
              "ssd": "unverified", "returncode": None}
    if not hook["command"]:
        result["detail"] = "No cache-clear hook configured."
        return result
    started = time.monotonic()
    env = os.environ | {"BENCHMARK_API_URL": cfg["api_url"], "BENCHMARK_MODEL": cfg["model"],
                        "BENCHMARK_PROFILE": profile, "BENCHMARK_CONCURRENCY": str(concurrency),
                        "BENCHMARK_REPEAT": str(repeat), "BENCHMARK_ARTIFACT_DIR": str(artifact)}
    path = artifact / "cache-hook.stdout.log"
    try:
        with path.open("wb") as out, (artifact / "cache-hook.stderr.log").open("wb") as err:
            child = subprocess.Popen(hook["command"], stdout=out, stderr=err, env=env,
                                     start_new_session=True)
            try:
                result["returncode"] = child.wait(timeout=hook["timeout_s"])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                terminate_group(child, signal.SIGKILL)
                child.wait()
                raise
        if result["returncode"] != 0:
            result["detail"] = "Cache-clear hook returned a nonzero exit status."
        else:
            attestation = json.loads(path.read_text(encoding="utf-8"))
            for tier in ("gpu", "cpu", "ssd"):
                status = attestation.get(tier)
                if status in ("cleared", "not_applicable"):
                    result[tier] = status
            if all(result[tier] != "unverified" for tier in ("gpu", "cpu", "ssd")):
                result["status"] = "confirmed"
            result["detail"] = attestation.get("detail")
    except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError) as error:
        result["detail"] = f"{type(error).__name__}: {error}"
    result["elapsed_s"] = time.monotonic() - started
    return result


def probe_request(cfg, payload):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    key = os.environ.get(cfg["api_key_env"]) if cfg["api_key_env"] else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(cfg["api_url"] + "/v1/chat/completions", data=data, headers=headers)
    info = {"http_status": None, "usage": None, "finish_reason": None, "sse_done": False}
    try:
        with urlopen(request, timeout=cfg["request_timeout_s"]) as response:
            info["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                event = line[5:].strip()
                if event == b"[DONE]":
                    info["sse_done"] = True
                    break
                if not event:
                    continue
                body = json.loads(event)
                if body.get("error"):
                    info["error"] = body["error"]
                if body.get("usage") is not None:
                    info["usage"] = body["usage"]
                for choice in body.get("choices", []):
                    if choice.get("finish_reason"):
                        info["finish_reason"] = choice["finish_reason"]
    except HTTPError as error:
        info.update(http_status=error.code, error=error.read(4096).decode(errors="replace"))
    except (OSError, ValueError) as error:
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def probe_eos(cfg, eos_token_id, limit=8):
    name = "max_tokens" if cfg["use_legacy_max_tokens"] else "max_completion_tokens"
    payload = request_settings(cfg) | {
        "model": cfg["model"], "messages": [{"role": "user", "content": "Say hello."}],
        "stream": True, "stream_options": {"include_usage": True}, name: limit,
        "logit_bias": {str(eos_token_id): 100},
    }
    results = {}
    for enabled in (False, True):
        payload["ignore_eos"] = enabled
        results[str(enabled).lower()] = probe_request(cfg, payload)
    control, enabled = results["false"], results["true"]
    counts = [(x["usage"] or {}).get("completion_tokens") for x in (control, enabled)]
    status = "inconclusive_forced_eos"
    if all(x["http_status"] == 200 and x["sse_done"] for x in (control, enabled)):
        if isinstance(counts[0], int) and counts[0] < limit and control["finish_reason"] == "stop":
            if counts[1] == limit and enabled["finish_reason"] == "length":
                status = "verified_by_forced_eos"
            elif isinstance(counts[1], int) and counts[1] < limit and enabled["finish_reason"] == "stop":
                status = "failed_forced_eos"
    return {"status": status, "tested_utc": utc_now(), "api_url": cfg["api_url"],
            "model": cfg["model"], "eos_token_id": eos_token_id, "completion_budget": limit,
            "chat_template_kwargs": cfg["chat_template_kwargs"],
            "use_legacy_max_tokens": cfg["use_legacy_max_tokens"], "results": results}


def recommended_duration(current, dispatch_seconds):
    return max(current, math.ceil(dispatch_seconds * 1.25 / 10) * 10)
