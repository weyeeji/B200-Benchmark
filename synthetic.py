"""Build deterministic single-turn request data for the cold-cache experiments."""

import hashlib
import importlib.util
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

WORD_TOKEN = re.compile(r" [A-Za-z]{1,17}")
PREFIX_TOKENS = 32


@lru_cache(maxsize=4)
def _load_tokenizer(directory: Path):
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
    backend.no_padding()
    backend.no_truncation()
    renderer = directory / "encoding" / "vllm_encoding.py"
    if not renderer.is_file():
        raise ValueError(f"DeepSeek V4.1 vLLM chat renderer is missing: {renderer}")
    spec = importlib.util.spec_from_file_location("benchmark_vllm_encoding", renderer)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)

    def encode(text: str) -> list[int]:
        prompt = encoder.encode_messages(
            [{"role": "user", "content": text}],
            thinking_mode="thinking",
            reasoning_effort="high",
        )
        return backend.encode(prompt, add_special_tokens=False).ids

    vocabulary_ids = sorted(set(backend.get_vocab(with_added_tokens=False).values()))
    pieces = backend.decode_batch([[token_id] for token_id in vocabulary_ids])
    words = [
        (token_id, piece)
        for token_id, piece in zip(vocabulary_ids, pieces)
        if WORD_TOKEN.fullmatch(piece)
    ]
    round_trips = backend.encode_batch(
        [word for _, word in words], add_special_tokens=False
    )
    pool = [
        (token_id, word)
        for (token_id, word), decoded in zip(words, round_trips)
        if decoded.ids == [token_id]
    ]
    if len(pool) < 256:
        raise ValueError(f"Not enough independent word tokens in {directory}")
    left = encode(pool[0][1])
    right = encode(pool[-1][1])
    first_word = next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        len(left),
    )
    if first_word >= PREFIX_TOKENS or len(left) != len(right):
        raise ValueError("DeepSeek chat renderer has an unexpected fixed prompt prefix")
    return encode, pool, first_word, len(left) - 1


def _fit_prompt(encode, pieces: list[str], pool: list[str], rng, target: int):
    for _ in range(8):
        text = "".join(pieces)
        token_ids = encode(text)
        difference = target - len(token_ids)
        if difference == 0:
            return text, token_ids
        if difference > 0:
            pieces.extend(rng.choices(pool, k=difference))
        elif len(pieces) + difference > 1:
            del pieces[difference:]
        else:
            break
    raise ValueError(f"Could not fit an exact {target}-token rendered prompt")


def write_dataset(
    path: Path,
    tokenizer_dir: Path,
    experiment: str,
    concurrency: int,
    *,
    input_tokens: int,
    output_tokens: int,
    count: int,
    seed: int,
    phase: str,
    nonce: str = "",
) -> str:
    """Write one AIPerf single_turn file in dispatch order and return its SHA-256."""
    if experiment not in ("P01", "P02") or phase not in ("warmup", "formal"):
        raise ValueError("Cold-cache datasets require P01 or P02 and a warmup or formal phase")
    if min(concurrency, output_tokens, count) < 1 or input_tokens < PREFIX_TOKENS:
        raise ValueError("Synthetic dataset sizes and concurrency must be positive")

    encode, pool, first_word, overhead = _load_tokenizer(tokenizer_dir.resolve())
    seed_bytes = json.dumps([seed, experiment, concurrency, phase, nonce]).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest(), "big"))
    words = [word for _, word in pool]
    # Partition the first user token so the two phases cannot share a 32-token prefix.
    offset = 0 if phase == "warmup" else 1
    leading = pool[offset::2]
    body_length = input_tokens - overhead - 1
    signatures = set()
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            for index in range(count):
                for _ in range(32):
                    token_id, head = rng.choice(leading)
                    text, token_ids = _fit_prompt(
                        encode, [head, *rng.choices(words, k=body_length)],
                        words, rng, input_tokens,
                    )
                    signature = tuple(token_ids[:PREFIX_TOKENS])
                    if token_ids[first_word] == token_id and signature not in signatures:
                        signatures.add(signature)
                        break
                else:
                    raise ValueError(f"Could not make {experiment} {phase} row {index} distinct")
                cache_salt = hashlib.sha256(seed_bytes + index.to_bytes(8, "big")).hexdigest()
                row = {
                    "session_id": f"{experiment}-C{concurrency}-{phase}-{index:06d}",
                    "text": text,
                    "output_length": output_tokens,
                    "extra": {"cache_salt": cache_salt},
                }
                line = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(line)
                digest.update(line)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return digest.hexdigest()
