"""Build and verify frozen AIPerf single-turn datasets."""

import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
from array import array
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

PROFILES = {
    "P01": (4096, 1024, 0),
    "P02": (65536, 600, 0),
    "P03": (65536, 600, 58880),
    "P04": (65536, 600, 58880),
}
DATASET_VERSION = 1
TEXT_TOKEN = re.compile(r" [A-Za-z]+\Z")
DATA_FILES = ("warmup.jsonl", "formal.jsonl", "order.jsonl")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid benchmark JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _seed(seed, *parts):
    label = repr((seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest(), "big")


def _token_hash(ids):
    return hashlib.sha256(array("I", ids).tobytes()).digest()


def _common_length(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return min(len(first), len(second))


def _groups_for_capacity(profile, capacity):
    prefix = PROFILES[profile][2]
    if profile == "P03":
        return capacity // (2 * prefix)
    return (2 * capacity + prefix - 1) // prefix


def _derived_capacity(profile, archived):
    if archived is None:
        return None
    capacity = archived.get("baseline_cache_capacity_tokens")
    origin = archived.get("prefix_groups_source")
    if archived.get("profile") != profile or origin not in (None, "capacity", "explicit"):
        raise ValueError(f"Invalid frozen prefix group source for {profile}")
    if origin == "capacity":
        if (
            type(capacity) is not int or capacity <= 0
            or archived.get("prefix_groups") != _groups_for_capacity(profile, capacity)
        ):
            raise ValueError(f"Invalid frozen baseline cache capacity for {profile}")
        return capacity
    if origin == "explicit" and capacity is not None:
        raise ValueError(f"Explicit groups must not declare a derivation capacity for {profile}")
    # Older archives did not store the origin. A matching formula is the
    # available evidence that their group count was derived from this capacity.
    if (
        origin is None and type(capacity) is int and capacity > 0
        and archived.get("prefix_groups") == _groups_for_capacity(profile, capacity)
    ):
        return capacity
    return None


def _groups(config, profile, archived, sibling):
    if not PROFILES[profile][2]:
        return 0, None, None
    count = (config.get("prefix_groups") or {}).get(profile)
    requested_capacity = config.get("cache_capacity_tokens")
    if requested_capacity is not None and (type(requested_capacity) is not int or requested_capacity <= 0):
        raise ValueError("cache_capacity_tokens must be a positive integer")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError(f"prefix_groups.{profile} must be a positive integer")
    origin = None
    if archived is not None:
        baseline_capacity = archived.get("baseline_cache_capacity_tokens")
        origin = archived.get("prefix_groups_source")
        derived_capacity = _derived_capacity(profile, archived)
        if derived_capacity is not None:
            if sibling is not None and sibling != derived_capacity:
                raise ValueError("P03 and P04 have different frozen derivation capacities; use a new dataset root")
            if count is None and requested_capacity not in (None, derived_capacity):
                raise ValueError(f"{profile} uses frozen baseline cache capacity {derived_capacity}")
        if count is None:
            count = archived.get("prefix_groups")
    elif count is not None:
        baseline_capacity = None
        origin = "explicit"
    else:
        if sibling is not None and requested_capacity not in (None, sibling):
            raise ValueError(
                f"P03/P04 use frozen baseline cache capacity {sibling}; "
                "use the same capacity or an explicit group count"
            )
        baseline_capacity = sibling if sibling is not None else requested_capacity
        if baseline_capacity is not None:
            count = _groups_for_capacity(profile, baseline_capacity)
        origin = "capacity"
    if type(count) is not int or count < 1:
        raise ValueError(
            f"{profile} needs a positive prefix_groups.{profile} or a baseline "
            "cache_capacity_tokens yielding at least one group"
        )
    return count, baseline_capacity, origin


def _archived_settings(profile_dir, preferred=None):
    baseline_path = profile_dir / "baseline.json"
    if baseline_path.is_file():
        return _load_json(baseline_path)
    # A point can commit before its baseline, and copied archives may omit it.
    if preferred is None or not preferred.is_file():
        preferred = next(iter(sorted(profile_dir.glob("c*/manifest.json"))), None)
    if preferred is None:
        return None
    settings = _load_json(preferred).get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Invalid frozen dataset settings: {preferred}")
    return settings


def _tokenizer_hashes(path, renderer):
    if not path.is_dir():
        raise ValueError(f"tokenizer must point to a local snapshot directory: {path}")
    names = {
        "config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "spiece.model", "sentencepiece.bpe.model", "added_tokens.json",
        "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt",
    }
    files = [item for item in path.iterdir() if item.is_file() and item.name in names]
    files.extend(path.glob("*.jinja"))
    files.extend((path / "chat_templates").glob("*.jinja"))
    if renderer == "deepseek":
        encoder = path / "encoding" / "vllm_encoding.py"
        if not encoder.is_file():
            encoder = path / "encoding" / "encoding.py"
        if not encoder.is_file():
            raise ValueError(f"DeepSeek chat encoder is missing under {path}/encoding")
        files.append(encoder)
    if not files:
        raise ValueError(f"No tokenizer assets found in {path}")
    return {item.relative_to(path).as_posix(): sha256(item) for item in sorted(files)}


@lru_cache(maxsize=4)
def _tokenizer(path, renderer, kwargs_json, hashes_json):
    from tokenizers import Tokenizer

    kwargs = json.loads(kwargs_json)
    if renderer == "deepseek":
        backend = Tokenizer.from_file(str(path / "tokenizer.json"))
        encoder_name = next(name for name in json.loads(hashes_json) if name.startswith("encoding/"))
        encoder_path = path / encoder_name
        spec = importlib.util.spec_from_file_location("benchmark_chat_encoder", encoder_path)
        module = importlib.util.module_from_spec(spec)
        # Load current bytes even when an encoder edit preserves mtime and size.
        exec(compile(encoder_path.read_bytes(), str(encoder_path), "exec"), vars(module))
        unknown = kwargs.keys() - {"thinking", "enable_thinking", "reasoning_effort", "drop_thinking"}
        if unknown:
            raise ValueError(f"Unsupported DeepSeek chat template kwargs: {sorted(unknown)}")
        thinking = True
        if "thinking" in kwargs or "enable_thinking" in kwargs:
            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            thinking, effort = False, None

        def render(text):
            return module.encode_messages(
                [{"role": "user", "content": text}],
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort="high" if effort is None else effort,
            )
    elif renderer == "hf":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if not tokenizer.is_fast:
            raise ValueError("The benchmark dataset generator requires a fast tokenizer")
        backend = tokenizer.backend_tokenizer

        def render(text):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
    else:
        raise ValueError("chat_encoder must be deepseek or hf")

    backend.no_padding()
    backend.no_truncation()
    vocab_ids = sorted(set(backend.get_vocab(with_added_tokens=False).values()))
    pieces = backend.decode_batch([[token_id] for token_id in vocab_ids])
    candidates = [
        (token_id, piece) for token_id, piece in zip(vocab_ids, pieces)
        if TEXT_TOKEN.fullmatch(piece) and 2 <= len(piece) <= 18
    ]
    round_trips = backend.encode_batch([piece for _, piece in candidates], add_special_tokens=False)
    pool = tuple(
        piece for (token_id, piece), encoding in zip(candidates, round_trips)
        if encoding.ids == [token_id]
    )
    if len(pool) < 256:
        raise ValueError("Tokenizer has fewer than 256 round-trippable leading-space ASCII tokens")

    def encode(text):
        return backend.encode(render(text), add_special_tokens=False).ids

    return encode, pool


def input_token_counter(config):
    """Return a lazy fallback that tokenizes a supported request as served."""
    tokenizer_path = Path(config["tokenizer"]).expanduser().resolve()
    renderer = config.get("chat_encoder", "deepseek")
    expected_kwargs = config.get("chat_template_kwargs") or {}
    encode = None

    def count(payload):
        nonlocal encode
        if not isinstance(payload, dict):
            return None
        if (payload.get("chat_template_kwargs") or {}) != expected_kwargs:
            return None
        if any(payload.get(key) is not None for key in (
            "chat_template", "reasoning_effort", "thinking", "truncate_prompt_tokens",
            "documents", "prompt_token_ids", "prompt_embeds",
        )):
            return None
        if (payload.get("add_generation_prompt") is False or payload.get("add_special_tokens")
                or payload.get("continue_final_message") or payload.get("tools")):
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
            return None
        message = messages[0]
        if message.get("role") != "user" or set(message) != {"role", "content"}:
            return None
        content = message.get("content")
        if renderer == "deepseek" and isinstance(content, list) and len(content) == 1:
            part = content[0]
            if isinstance(part, dict) and part.get("type") == "text" and set(part) == {"type", "text"}:
                content = part.get("text")
        if not isinstance(content, str):
            return None
        if encode is None:
            hashes = _tokenizer_hashes(tokenizer_path, renderer)
            encode, _ = _tokenizer(
                tokenizer_path, renderer, json.dumps(expected_kwargs, sort_keys=True),
                json.dumps(hashes, sort_keys=True),
            )
        return len(encode(content))

    return count


def _fit(encode, pool, rng, count, target, prefix="", compare=None):
    pieces = rng.choices(pool, k=max(1, count))
    for _ in range(8):
        text = "".join(pieces)
        if compare is not None:
            first = encode(text + compare[0])
            second = encode(text + compare[1])
            actual = _common_length(first, second)
            ids = first[:actual]
        else:
            ids = encode(prefix + text)
            actual = len(ids)
        delta = target - actual
        if delta == 0:
            return text, ids
        if delta > 0:
            pieces.extend(rng.choices(pool, k=delta))
        elif len(pieces) + delta >= 1:
            del pieces[delta:]
        else:
            break
    raise ValueError(f"Cannot synthesize an exact {target}-token chat input with this tokenizer/template")


def _order(group_count, total, seed):
    rng = random.Random(seed)
    if not group_count:
        yield from (None for _ in range(total))
        return
    remaining = total
    while remaining:
        groups = list(range(group_count))
        rng.shuffle(groups)
        yield from groups[:remaining]
        remaining -= min(remaining, group_count)


def _generate(directory, settings, requests, tokenizer_path, tokenizer_hashes, *, phase_orders=None):
    encode, pool = _tokenizer(
        tokenizer_path, settings["chat_encoder"], json.dumps(settings["chat_template_kwargs"], sort_keys=True),
        json.dumps(tokenizer_hashes, sort_keys=True),
    )
    profile = settings["profile"]
    seed = settings["seed"]
    isl, prefix_size = settings["isl"], settings["prefix_tokens"]
    osl = settings["osl"]
    group_count = settings["prefix_groups"]
    scenario_seed = "P03/P04" if group_count else profile
    if phase_orders is not None:
        scenario_seed = "cache-capacity"
    anchor_a, anchor_b = pool[0], pool[-1]
    first, second = encode(anchor_a), encode(anchor_b)
    template_head = _common_length(first, second)
    template_overhead = len(first) - 1
    # Groups can outnumber individual token IDs, so distinguish them within four
    # variable tokens. Distinguish each prompt or suffix at its first variable token.
    signature_size = 4
    early_signatures = set()
    suffix_signatures = {}
    prefixes = {}

    def group_prefix(group):
        if group not in prefixes:
            rng = random.Random(_seed(seed, scenario_seed, "prefix", group))
            for _ in range(32):
                content, ids = _fit(
                    encode, pool, rng, prefix_size - template_head, prefix_size,
                    compare=(anchor_a, anchor_b),
                )
                signature = tuple(ids[template_head:template_head + signature_size])
                if len(signature) == signature_size and signature not in early_signatures:
                    early_signatures.add(signature)
                    prefixes[group] = content, _token_hash(ids)
                    break
            else:
                raise ValueError(f"Could not make group {group} distinct near the chat template header")
        return prefixes[group]

    # Generate shared prefixes in stable order independently of the request schedule.
    for group in range(group_count):
        group_prefix(group)

    with (directory / "order.jsonl").open("w", encoding="utf-8") as orders:
        for phase, total in requests.items():
            with (directory / f"{phase}.jsonl").open("w", encoding="utf-8") as records:
                order = (
                    phase_orders[phase] if phase_orders is not None
                    else _order(group_count, total, _seed(seed, scenario_seed, phase, "order"))
                )
                for index, group in enumerate(order):
                    rng = random.Random(_seed(seed, scenario_seed, phase, "prompt", index))
                    shared, prefix_digest = ("", None) if group is None else group_prefix(group)
                    limit = template_head if group is None else prefix_size
                    seen = early_signatures if group is None else suffix_signatures.setdefault(group, set())
                    estimate = isl - template_overhead - (prefix_size - template_head if group is not None else 0)
                    for _ in range(32):
                        own_text, ids = _fit(
                            encode, pool, rng, estimate, isl, prefix=shared,
                        )
                        if prefix_digest is not None and _token_hash(ids[:prefix_size]) != prefix_digest:
                            continue
                        signature = ids[limit]
                        if signature not in seen:
                            seen.add(signature)
                            break
                    else:
                        raise ValueError(f"Failed to make {phase} request {index} distinct with its exact prefix")
                    session_id = f"{phase}-{index:09d}"
                    _jsonl(records, {"session_id": session_id, "text": shared + own_text, "output_length": osl})
                    _jsonl(orders, {
                        "session_id": session_id, "phase": phase,
                        "prefix_group": group, "cycle": index // group_count if group_count else None,
                    })
    return {
        "cross_group_prefix_max_extra_tokens": signature_size - 1 if group_count else None,
        "distinct_first_variable_token_per_group_or_uncached_prompt": True,
        "template_common_header_tokens": template_head,
        "every_input_verified": True,
        "every_shared_prefix_verified": bool(group_count),
    }


def _point_fields(settings, requests, concurrency):
    groups = settings["prefix_groups"]
    return {
        "settings": settings,
        "profile": settings["profile"],
        "concurrency": concurrency,
        "isl": settings["isl"],
        "osl": settings["osl"],
        "prefix_tokens": settings["prefix_tokens"],
        "prefix_groups": groups,
        "warmup_requests": requests["warmup"],
        "formal_requests": requests["formal"],
        "formal_full_cycles": requests["formal"] // groups if groups else None,
        "theoretical_shared_prefix_ratio": settings["prefix_tokens"] / settings["isl"],
    }


def _check_point(directory, expected):
    manifest_path = directory / "manifest.json"
    result = _load_json(manifest_path)
    for key, value in expected.items():
        actual = result.get(key)
        if key not in result or type(actual) is not type(value) or actual != value:
            raise ValueError(f"Frozen dataset configuration mismatch for {key}: {manifest_path}")
    files = result.get("files")
    if not isinstance(files, dict) or set(files) != set(DATA_FILES):
        raise ValueError(f"Frozen dataset manifest lists unexpected files: {manifest_path}")
    for filename, checksum in files.items():
        path = directory / filename
        if sha256(path) != checksum:
            raise ValueError(f"Frozen dataset SHA-256 mismatch: {path}")
    return {
        **result,
        "manifest": manifest_path,
        "formal": directory / "formal.jsonl",
        "warmup": directory / "warmup.jsonl",
        "order": directory / "order.jsonl",
    }


def _check_snapshot(tokenizer_path, renderer, hashes):
    if _tokenizer_hashes(tokenizer_path, renderer) != hashes:
        raise ValueError(f"Tokenizer snapshot changed while generating data: {tokenizer_path}")


def prepare_capacity_trial(config, directory, groups, seed, osl):
    """Materialize an anchor, one fill per competing group and a fresh anchor probe."""
    if type(groups) is not int or groups < 1:
        raise ValueError("Calibration requires a positive number of prefix groups")
    if type(seed) is not int or seed < 0:
        raise ValueError("Calibration seed must be a nonnegative integer")
    if type(osl) is not int or osl < 1:
        raise ValueError("Calibration OSL must be a positive integer")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = ("prime.jsonl", "fill.jsonl", "probe.jsonl", "order.jsonl")
        manifest = directory / "manifest.json"
        if any((directory / name).exists() for name in (*files, manifest.name)):
            raise FileExistsError(f"Calibration trial dataset already exists in {directory}")

        tokenizer_path = Path(config["tokenizer"]).expanduser().resolve()
        renderer = config.get("chat_encoder", "deepseek")
        tokenizer_hashes = _tokenizer_hashes(tokenizer_path, renderer)
        isl, _, prefix_size = PROFILES["P03"]
        settings = {
            "dataset_version": DATASET_VERSION,
            "profile": "P03",
            "seed": seed,
            "isl": isl,
            "osl": osl,
            "prefix_tokens": prefix_size,
            "prefix_groups": groups,
            "chat_encoder": renderer,
            "chat_template_kwargs": config.get("chat_template_kwargs") or {},
            "tokenizer_sha256": tokenizer_hashes,
        }
        fill_order = list(range(1, groups))
        random.Random(_seed(seed, "cache-capacity", "fill")).shuffle(fill_order)
        phase_orders = {"prime": [0], "fill": fill_order, "probe": [0]}
        requests = {phase: len(order) for phase, order in phase_orders.items()}
        with TemporaryDirectory(prefix=".building-", dir=directory) as temporary:
            staging = Path(temporary)
            checks = _generate(
                staging, settings, requests, tokenizer_path, tokenizer_hashes, phase_orders=phase_orders,
            )
            _check_snapshot(tokenizer_path, renderer, tokenizer_hashes)
            result = {
                "purpose": "gpu_prefix_cache_capacity",
                "settings": settings,
                "groups": groups,
                "seed": seed,
                "isl": isl,
                "osl": osl,
                "prefix_tokens": prefix_size,
                "requests": requests,
                "tokenizer_version": importlib.metadata.version("tokenizers"),
                "tokenizer_path_at_generation": str(tokenizer_path),
                "generator_sha256": sha256(Path(__file__).resolve()),
                "validation": checks,
                "files": {name: sha256(staging / name) for name in files},
            }
            _write_json(staging / manifest.name, result)
            published = []
            try:
                for name in (*files, manifest.name):
                    (directory / name).hardlink_to(staging / name)
                    published.append(name)
            except BaseException:
                for name in published:
                    (directory / name).unlink()
                raise
    return {**result, "manifest": manifest, **{name[:-6]: directory / name for name in files}}


def prepare(config, root, profile, concurrency, *, create=True):
    """Build a point atomically if needed, then verify its frozen config and hashes."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    seed = config.get("seed", 20261125)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tokenizer_path = Path(config["tokenizer"]).expanduser().resolve()
    renderer = config.get("chat_encoder", "deepseek")
    root = Path(root).expanduser().resolve()
    profile_dir = root / profile
    directory = profile_dir / f"c{concurrency}"
    manifest_path = directory / "manifest.json"
    if not create and not manifest_path.is_file():
        raise ValueError(f"Frozen dataset is missing; run prepare first: {directory}")
    cached = bool(PROFILES[profile][2])
    lock_path = root / ".prepare-cache.lock" if cached else profile_dir / ".prepare.lock"
    if create:
        profile_dir.mkdir(parents=True, exist_ok=True)
    with (lock_path.open("a") if create else nullcontext()) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        tokenizer_hashes = _tokenizer_hashes(tokenizer_path, renderer)
        baseline_path = profile_dir / "baseline.json"
        baseline = _archived_settings(profile_dir, manifest_path)
        sibling_capacity = None
        if cached:
            sibling = "P04" if profile == "P03" else "P03"
            sibling_capacity = _derived_capacity(sibling, _archived_settings(root / sibling))
        groups, baseline_capacity, origin = _groups(config, profile, baseline, sibling_capacity)
        isl, osl, prefix_size = PROFILES[profile]
        settings = {
            "dataset_version": DATASET_VERSION,
            "profile": profile,
            "seed": seed,
            "isl": isl,
            "osl": osl,
            "prefix_tokens": prefix_size,
            "prefix_groups": groups,
            "chat_encoder": renderer,
            "chat_template_kwargs": config.get("chat_template_kwargs") or {},
            "tokenizer_sha256": tokenizer_hashes,
            "baseline_cache_capacity_tokens": baseline_capacity,
        }
        if origin is not None:
            settings["prefix_groups_source"] = origin
        if baseline is not None and settings != baseline:
            raise ValueError(
                f"{profile} dataset root already has a different frozen baseline: {baseline_path}"
            )
        requests = {"warmup": max(16, groups), "formal": max(3 * concurrency, 2 * groups)}
        fields = _point_fields(settings, requests, concurrency)
        if create and not directory.exists():
            with TemporaryDirectory(prefix=".building-", dir=profile_dir) as temporary:
                staging = Path(temporary) / "point"
                staging.mkdir()
                checks = _generate(staging, settings, requests, tokenizer_path, tokenizer_hashes)
                _check_snapshot(tokenizer_path, renderer, tokenizer_hashes)
                result = {
                    **fields,
                    "tokenizer_version": importlib.metadata.version("tokenizers"),
                    "tokenizer_path_at_generation": str(tokenizer_path),
                    "generator_sha256": sha256(Path(__file__).resolve()),
                    "validation": checks,
                    "files": {name: sha256(staging / name) for name in DATA_FILES},
                }
                _write_json(staging / "manifest.json", result)
                staging.rename(directory)
        result = _check_point(directory, fields)
        if create and not baseline_path.exists():
            temporary_path = profile_dir / ".baseline.tmp"
            _write_json(temporary_path, settings)
            os.replace(temporary_path, baseline_path)
        return result
