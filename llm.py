"""One small wrapper around the Anthropic Messages API, shared by score.py and extract.py.

Both steps want the same thing: send a prompt, get JSON back, check its shape, retry
once if it is broken, and record tokens, latency, and dollars.
"""
import hashlib
import json
import os
import re
import threading
import time

import config

_client = None
KEY_STATUS = "unchecked"   # set by check_key(): ok, rejected, missing, unreachable


def api_key():
    """ANTHROPIC_API_KEY from .env, forgiving stray quotes and spaces; None if unusable."""
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'").strip()
    if not key or key.endswith("...") or key in ("sk-ant-", "your-key-here"):
        return None   # missing, or still the placeholder from .env.example
    return key


def masked_key():
    key = api_key()
    return f"{key[:12]}…{key[-4:]}" if key and len(key) > 20 else ("(set)" if key else "(none)")


def has_key():
    """True when there's a key worth trying (not known to be rejected)."""
    return api_key() is not None and KEY_STATUS != "rejected"


def check_key():
    """One free call (list models) to find out whether Anthropic accepts the key."""
    global KEY_STATUS
    if api_key() is None:
        KEY_STATUS = "missing"
        return KEY_STATUS
    try:
        client().models.list(limit=1)
        KEY_STATUS = "ok"
    except Exception as e:
        status = getattr(e, "status_code", None)
        KEY_STATUS = "rejected" if status in (401, 403) else "unreachable"
    return KEY_STATUS


def client():
    """Create the API client lazily so scripts that never call the LLM don't need a key."""
    global _client
    if _client is None:
        import anthropic  # imported here so tests run without the package configured
        _client = anthropic.Anthropic(api_key=api_key())
    return _client


def cost_usd(input_tokens, output_tokens):
    return (input_tokens * config.PRICE_INPUT_PER_M
            + output_tokens * config.PRICE_OUTPUT_PER_M) / 1_000_000


def parse_json(text):
    """Pull the first {...} object out of a reply. Models sometimes wrap JSON in ```fences```."""
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _cache_path(model, system, user, max_tokens):
    key = hashlib.sha256(json.dumps([model, system, user, max_tokens]).encode()).hexdigest()[:32]
    return config.CACHE_DIR / "llm" / f"{key}.json"


def ask_json(system, user, max_tokens, validate, model=None, cache=False):
    """Ask for JSON. `validate(dict) -> bool` decides if the reply is usable.

    Returns (data_or_None, stats) where stats sums tokens, latency, and cost over attempts.
    Two attempts total: the original and one retry, as the spec asks.

    cache=True reuses an earlier valid answer to the exact same request, so retrying a
    failed search only pays for the steps that hadn't finished. (score.py leaves it off,
    because evaluate.py measures real latency.)
    """
    stats = {"latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
             "cached": False}
    model = model or config.LLM_MODEL
    path = _cache_path(model, system, user, max_tokens) if cache else None
    if path and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if validate(data):
                return data, {**stats, "cached": True}
        except (ValueError, OSError):
            pass  # a damaged cache file is just a miss
    messages = [{"role": "user", "content": user}]
    for attempt in range(2):
        start = time.perf_counter()
        resp = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        stats["latency_ms"] += (time.perf_counter() - start) * 1000
        stats["input_tokens"] += resp.usage.input_tokens
        stats["output_tokens"] += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        data = parse_json(text)
        if data is not None and validate(data):
            stats["cost_usd"] = cost_usd(stats["input_tokens"], stats["output_tokens"])
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(path)  # atomic, so parallel workers never read half a file
            return data, stats
        # Retry with a firmer reminder rather than the same prompt.
        messages = [{"role": "user", "content": user + "\n\nReturn ONLY valid JSON, "
                     "exactly in the requested shape, with no other text."}]
    stats["cost_usd"] = cost_usd(stats["input_tokens"], stats["output_tokens"])
    return None, stats
