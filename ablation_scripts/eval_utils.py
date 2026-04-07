"""Shared evaluation utilities for E2B ablation studies.

Extracted from E4B ablation scripts for reuse across all E2B phases.
Same prompts and PPL text as E4B for cross-model comparison.
"""
import os
import time

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ── Evaluation Constants (same as E4B for comparability) ──────────────────────

PROMPTS = [
    "What is 2+2? Answer with just the number.",
    "Explain quantum entanglement in 2 sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the three laws of thermodynamics? Be concise.",
    "Translate 'The cat sat on the mat' into French.",
]

PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was nothing but darkness and silence. "
    "Then the universe expanded rapidly, creating matter and energy. "
    "Stars formed, planets coalesced, and life emerged on at least one world. "
    "Humans developed language, tools, agriculture, and eventually technology. "
    "Today we stand at the threshold of artificial general intelligence."
)


# ── Measurement Functions ─────────────────────────────────────────────────────

def model_size_gb(path: str) -> float:
    """Compute total safetensors file size in GB."""
    total = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".safetensors")
    )
    return total / 1e9


def compute_ppl(model, tokenizer, text: str = PPL_TEXT) -> float:
    """Compute perplexity on the given text."""
    tokens = tokenizer.encode(text, return_tensors=None)
    token_ids = tokens if isinstance(tokens, list) else list(tokens)
    if len(token_ids) < 2:
        return float("inf")
    x = mx.array(token_ids)[None, :]
    logits = model(x)
    mx.eval(logits)
    logits = logits[:, :-1, :]
    targets = mx.array(token_ids[1:])[None, :]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = 0.0
    for i in range(targets.shape[1]):
        nll -= log_probs[0, i, targets[0, i].item()].item()
    return float(mx.exp(mx.array(nll / targets.shape[1])).item())


def test_generation(model, tok, prompts=None, max_tokens: int = 80) -> list[dict]:
    """Run chat-templated generation on prompts, returning quality metrics."""
    if prompts is None:
        prompts = PROMPTS
    sampler = make_sampler(temp=0.0)
    results = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        out = generate(model, tok, prompt=chat, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        dt = time.time() - t0
        n_tok = len(tok.encode(out))
        results.append({
            "prompt": p,
            "response": out.strip().split("\n")[0][:150],
            "tokens": n_tok,
            "tok_per_sec": n_tok / dt if dt > 0 else 0,
        })
    return results


def print_generation_results(gen_results: list[dict]):
    """Pretty-print generation results."""
    for g in gen_results:
        quality = "✅" if g["tokens"] > 0 else "❌"
        print(f"  {quality} [{g['tok_per_sec']:.0f} t/s] {g['prompt'][:40]}")
        print(f"     → {g['response'][:100]}")


def avg_tok_per_sec(gen_results: list[dict]) -> float:
    """Average tokens/sec across generation results (excluding 1-token responses)."""
    valid = [g["tok_per_sec"] for g in gen_results if g["tokens"] > 1]
    return sum(valid) / len(valid) if valid else 0.0
