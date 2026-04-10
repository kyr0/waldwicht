#!/usr/bin/env python3
"""Calibration test: sends prompts to the server and records timing stats."""

import json
import time

from test_common import apply_chat_defaults, create_client, require_content, resolve_model

client = create_client()
MODEL, AVAILABLE_MODELS = resolve_model(client)

PROMPTS = [
    "What is 2+2? Answer with just the number.",
    "Solve: 17 x 23. Show your work step by step.",
    "If a train travels 120 km in 2 hours, what is its average speed in m/s?",
    "What is the derivative of x³ + 2x² - 5x + 3?",
    "A box has 3 red, 5 blue, and 2 green balls. What's the probability of picking blue?",
    "What is the sum of the first 100 positive integers?",
    "Convert 0.375 to a fraction in simplest form.",
    "If f(x) = 2x + 1 and g(x) = x², what is f(g(3))?",
]

results = []

for i, prompt in enumerate(PROMPTS):
    print(f"[{i}] Sending: {prompt[:60]}...")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        **apply_chat_defaults(),
    )
    t1 = time.perf_counter()

    content = require_content(resp)
    completion_tokens = resp.usage.completion_tokens if resp.usage else 0

    elapsed = round(t1 - t0, 2)
    tok_s = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

    entry = {
        "id": i,
        "prompt": prompt,
        "response": content,
        "tokens": completion_tokens,
        "time_s": elapsed,
        "tok_per_sec": tok_s,
    }
    results.append(entry)
    print(f"    => {completion_tokens} tokens in {elapsed}s ({tok_s} tok/s)")

with open("calibration.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nWrote {len(results)} entries to calibration.json")
