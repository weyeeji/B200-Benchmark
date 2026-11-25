"""Capture the code, tokenizer and explicitly supplied serving configuration."""

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


ROUTING_ENV = (
    "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES",
    "AIPERF_AGENTX_CONTEXT_OVERFLOW_RATE_LIMIT",
)
DEPLOYMENT_FIELDS = {
    "label", "backend_container", "image", "image_id", "engine_commit", "engine_version",
    "gpu_devices", "tensor_parallel", "data_parallel", "pipeline_parallel", "expert_parallel",
    "prefill_schedule_interval", "max_num_batched_tokens", "max_num_seqs",
    "max_model_len", "kv_cache_dtype", "routing_policy", "cache_start_state", "speculative_config",
}
SPEC_FIELDS = {
    "method", "model", "num_speculative_tokens", "draft_sample_method", "rejection_sample_method",
    "synthetic_acceptance_length", "synthetic_acceptance_rates", "enable_adaptive_verification",
}
BACKEND_FLAGS = {
    "tensor-parallel-size": "tensor_parallel",
    "data-parallel-size": "data_parallel",
    "pipeline-parallel-size": "pipeline_parallel",
    "prefill-schedule-interval": "prefill_schedule_interval",
    "max-num-batched-tokens": "max_num_batched_tokens",
    "max-num-seqs": "max_num_seqs",
    "max-model-len": "max_model_len",
    "kv-cache-dtype": "kv_cache_dtype",
    "speculative-config": "speculative_config",
}
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "encoding/vllm_encoding.py",
)


def deployment_from_file(path):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.keys() - DEPLOYMENT_FIELDS:
        raise ValueError(f"{path}: expected an object with documented deployment fields")
    spec = data.get("speculative_config", {})
    if not isinstance(spec, dict) or spec.keys() - SPEC_FIELDS:
        raise ValueError(f"{path}: unsupported speculative_config fields")
    for key, value in data.items():
        if key in ("gpu_devices", "speculative_config"):
            continue
        if type(value) not in (str, int, float, bool) and value is not None:
            raise ValueError(f"{path}: {key} must be a scalar")
    devices = data.get("gpu_devices", [])
    if not isinstance(devices, list) or any(type(device) not in (str, int) for device in devices):
        raise ValueError(f"{path}: gpu_devices must be a list of device IDs")
    return data


def deployment_from_container(name):
    if name is None:
        return {}
    if name.startswith("-"):
        raise ValueError("--backend-container must be a container name or ID")
    try:
        container, = json.loads(subprocess.check_output(
            ["docker", "inspect", "--type", "container", name], text=True,
            stderr=subprocess.PIPE, timeout=8,
        ))
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f"could not inspect backend container {name!r}") from exc

    config = container["Config"]
    labels = config.get("Labels") or {}
    args = container.get("Args", [])
    devices = (container.get("HostConfig", {}).get("DeviceRequests") or [])
    result = {
        "backend_container": container["Name"].lstrip("/"),
        "image": config["Image"],
        "image_id": container["Image"],
        "engine_commit": labels.get("ai.vllm.build.commit") or labels.get("org.opencontainers.image.revision"),
        "gpu_devices": [device for request in devices for device in request.get("DeviceIDs", [])],
        "expert_parallel": "--enable-expert-parallel" in args,
    }
    for flag, field in BACKEND_FLAGS.items():
        for index, value in enumerate(args):
            key, separator, inline = value.partition("=")
            if key == f"--{flag}":
                if not separator and index + 1 == len(args):
                    break
                argument = inline if separator else args[index + 1]
                if field == "speculative_config":
                    parsed = json.loads(argument)
                    result[field] = {key: value for key, value in parsed.items() if key in SPEC_FIELDS}
                else:
                    result[field] = int(argument) if argument.isdecimal() else argument
                break
    return {key: value for key, value in result.items() if value is not None}


def tokenizer_fingerprints(tokenizer):
    files = {}
    for name in TOKENIZER_FILES:
        path = tokenizer / name
        if path.is_file():
            with path.open("rb") as stream:
                files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files.get("tokenizer.json"), fingerprint, files


def git_provenance(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
                                       timeout=5).strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def capture_run_metadata(root: Path, tokenizer: Path, *, deployment_file=None, backend_container=None):
    code_commit, code_dirty = git_provenance(root)
    tokenizer_hash, bundle_hash, files = tokenizer_fingerprints(tokenizer)
    return {
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "aiperf_version": version("aiperf"),
        "python_version": sys.version.split()[0],
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_bundle_sha256": bundle_hash,
        "tokenizer_files": files,
        "deployment": {**deployment_from_file(deployment_file), **deployment_from_container(backend_container)},
        "routing": {name: os.environ[name] for name in ROUTING_ENV if name in os.environ},
    }
