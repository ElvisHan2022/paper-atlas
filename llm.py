"""One small wrapper around the Anthropic Messages API, shared by score.py and extract.py.

Both steps want the same thing: send a prompt, get JSON back, check its shape, retry
once if it is broken, and record tokens, latency, and dollars.
"""
import json
import re
import time

import config

_client = None


def client():
    """Create the API client lazily so scripts that never call the LLM don't need a key."""
    global _client
    if _client is None:
        import anthropic  # imported here so tests run without the package configured
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
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


def ask_json(system, user, max_tokens, validate):
    """Ask for JSON. `validate(dict) -> bool` decides if the reply is usable.

    Returns (data_or_None, stats) where stats sums tokens, latency, and cost over attempts.
    Two attempts total: the original and one retry, as the spec asks.
    """
    stats = {"latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    messages = [{"role": "user", "content": user}]
    for attempt in range(2):
        start = time.perf_counter()
        resp = client().messages.create(
            model=config.LLM_MODEL,
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
            return data, stats
        # Retry with a firmer reminder rather than the same prompt.
        messages = [{"role": "user", "content": user + "\n\nReturn ONLY valid JSON, "
                     "exactly in the requested shape, with no other text."}]
    stats["cost_usd"] = cost_usd(stats["input_tokens"], stats["output_tokens"])
    return None, stats
