"""Memory diagnostic for TurboQuant KV cache leak hunting.

Tests the exact speculative decoding path: draft N tokens, verify, trim, repeat.
"""
import os, sys, gc

venv = os.path.join(os.path.dirname(__file__), "..", ".venv")
sys.path.insert(0, os.path.join(venv, "lib/python3.12/site-packages"))

import mlx.core as mx

def active():
    return mx.get_active_memory() / 1e6

def peak():
    return mx.get_peak_memory() / 1e6

def cache_mem():
    return mx.get_cache_memory() / 1e6

def report(label):
    gc.collect()
    mx.eval(mx.array(0))
    print(f"[{label}]  active={active():.0f} MB  peak={peak():.0f} MB  cache={cache_mem():.0f} MB")

print(f"PID: {os.getpid()}")
report("startup")

from mlx_lm import load
model, tokenizer = load("prism-ml/Waldwicht-8B-mlx-1bit")
mx.eval(model.parameters())
report("model loaded")

draft_model, _ = load("prism-ml/Waldwicht-1.7B-mlx-1bit")
mx.eval(draft_model.parameters())
report("draft loaded")

# ---- Test 1: Speculative decode simulation with TurboQuant ----
print("\n=== TEST 1: Speculative decode + TurboQuant ===")
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
cache = make_prompt_cache(model, turbo_kv_bits=4, turbo_fp16_layers=2)
draft_cache = make_prompt_cache(draft_model)

prompt = "Explain the theory of relativity in detail, including both special and general relativity. "
tokens = tokenizer.encode(prompt)
y = mx.array(tokens)

# Prefill both models
model_cache = cache
while y.size > 512:
    model(y[:512][None], cache=model_cache)
    mx.eval([c.state for c in model_cache])
    y = y[512:]
    mx.clear_cache()

logits = model(y[None], cache=model_cache)
mx.eval([c.state for c in model_cache])

draft_y = mx.array(tokens)
while draft_y.size > 512:
    draft_model(draft_y[:512][None], cache=draft_cache)
    mx.eval([c.state for c in draft_cache])
    draft_y = draft_y[512:]
draft_logits = draft_model(draft_y[None], cache=draft_cache)
mx.eval([c.state for c in draft_cache])

report("after prefill")

# Simulate speculative decoding loop: draft 3 => verify => trim => repeat
NUM_DRAFT = 3
NUM_STEPS = 200
print(f"\nSpec decode: {NUM_DRAFT} draft tokens, {NUM_STEPS} steps")

y = mx.argmax(logits[:, -1, :], axis=-1)
draft_y = mx.array(y)

for step in range(NUM_STEPS):
    # Draft phase: generate NUM_DRAFT tokens with draft model
    draft_tokens = []
    for _ in range(NUM_DRAFT):
        dl = draft_model(draft_y[None, None] if draft_y.ndim == 0 else draft_y[None], cache=draft_cache)
        draft_y = mx.argmax(dl[:, -1, :], axis=-1).squeeze()
        mx.async_eval(draft_y)
        draft_tokens.append(draft_y)
    draft_tokens = mx.stack(draft_tokens)

    # Verify phase: main model processes all tokens at once
    verify_input = mx.concatenate([y.reshape(-1), draft_tokens.reshape(-1)])
    vlogits = model(verify_input[None], cache=model_cache)
    verified = mx.argmax(vlogits[:, :, :], axis=-1).squeeze()
    mx.eval(verified, draft_tokens)

    # Simulate partial acceptance (accept 1 of 3 drafts on average)
    n_accept = step % NUM_DRAFT  # 0, 1, 2, 0, 1, 2, ...
    n_reject = NUM_DRAFT - n_accept

    # Trim rejected tokens from both caches
    trim_prompt_cache(model_cache, n_reject)
    trim_prompt_cache(draft_cache, max(n_reject - 1, 0))

    # Next token
    y = verified[n_accept]
    draft_y = mx.array(y)

    if (step + 1) % 25 == 0:
        # Check TurboQuant cache state
        deq_buf_bytes = 0
        for c in model_cache:
            if hasattr(c, '_k_deq_buf') and c._k_deq_buf is not None:
                deq_buf_bytes += c._k_deq_buf.nbytes + c._v_deq_buf.nbytes
        cache_nbytes = sum(c.nbytes for c in model_cache)
        cache_offset = model_cache[2].offset if hasattr(model_cache[2], 'offset') else '?'
        report(f"step {step+1} (offset={cache_offset}, cache={cache_nbytes/1e6:.1f}MB, deq_buf={deq_buf_bytes/1e6:.1f}MB)")

report("after spec decode")

# ---- Test 2: Same thing but with regular KVCache (no turbo) ----
print("\n=== TEST 2: Speculative decode + regular KVCache ===")
mx.clear_cache()
del cache, model_cache, draft_cache, logits, draft_logits, vlogits, y, draft_y
gc.collect()
mx.eval(mx.array(0))
report("cleaned up")

cache2 = make_prompt_cache(model)
draft_cache2 = make_prompt_cache(draft_model)

y2 = mx.array(tokens)
while y2.size > 512:
    model(y2[:512][None], cache=cache2)
    mx.eval([c.state for c in cache2])
    y2 = y2[512:]
logits2 = model(y2[None], cache=cache2)
mx.eval([c.state for c in cache2])

dy2 = mx.array(tokens)
while dy2.size > 512:
    draft_model(dy2[:512][None], cache=draft_cache2)
    mx.eval([c.state for c in draft_cache2])
    dy2 = dy2[512:]
dl2 = draft_model(dy2[None], cache=draft_cache2)
mx.eval([c.state for c in draft_cache2])

report("after prefill (no turbo)")

y2 = mx.argmax(logits2[:, -1, :], axis=-1)
dy2 = mx.array(y2)

for step in range(NUM_STEPS):
    draft_tokens = []
    for _ in range(NUM_DRAFT):
        dl2 = draft_model(dy2[None, None] if dy2.ndim == 0 else dy2[None], cache=draft_cache2)
        dy2 = mx.argmax(dl2[:, -1, :], axis=-1).squeeze()
        mx.async_eval(dy2)
        draft_tokens.append(dy2)
    draft_tokens = mx.stack(draft_tokens)

    verify_input = mx.concatenate([y2.reshape(-1), draft_tokens.reshape(-1)])
    vlogits = model(verify_input[None], cache=cache2)
    verified = mx.argmax(vlogits[:, :, :], axis=-1).squeeze()
    mx.eval(verified, draft_tokens)

    n_accept = step % NUM_DRAFT
    n_reject = NUM_DRAFT - n_accept
    trim_prompt_cache(cache2, n_reject)
    trim_prompt_cache(draft_cache2, max(n_reject - 1, 0))

    y2 = verified[n_accept]
    dy2 = mx.array(y2)

    if (step + 1) % 25 == 0:
        cache_nbytes = sum(c.nbytes for c in cache2)
        cache_offset = cache2[0].offset
        report(f"step {step+1} (offset={cache_offset}, cache={cache_nbytes/1e6:.1f}MB)")

report("after spec decode (no turbo)")

print("\n=== COMPARISON DONE ===")
