"""Memory diagnostic for TurboQuant KV cache leak hunting."""
import os, sys, subprocess, gc

# Activate venv
venv = os.path.join(os.path.dirname(__file__), "..", ".venv")
sys.path.insert(0, os.path.join(venv, "lib/python3.12/site-packages"))

import mlx.core as mx

def mem_mb():
    """Return current process memory footprint in MB via footprint tool."""
    try:
        out = subprocess.check_output(["footprint", str(os.getpid())], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "footprint" in line.lower() and "MB" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "MB" and i > 0:
                        return float(parts[i-1])
    except Exception:
        pass
    # Fallback: MLX metal memory
    return mx.metal.get_active_memory() / 1e6

def metal_mb():
    return mx.metal.get_active_memory() / 1e6

def peak_mb():
    return mx.metal.get_peak_memory() / 1e6

def report(label):
    gc.collect()
    mx.eval(mx.array(0))  # force graph eval
    print(f"[{label}]  metal_active={metal_mb():.1f} MB  metal_peak={peak_mb():.1f} MB")

print(f"PID: {os.getpid()}")
report("startup")

# Load model
from mlx_lm import load
model, tokenizer = load("prism-ml/Waldwicht-8B-mlx-1bit")
mx.eval(model.parameters())
report("model loaded")

# Load draft model
draft_model, _ = load("prism-ml/Waldwicht-1.7B-mlx-1bit")
mx.eval(draft_model.parameters())
report("draft loaded")

# Create turbo caches
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model, turbo_kv_bits=4, turbo_fp16_layers=2)
report("cache created (empty)")

print(f"\nCache structure: {len(cache)} layers")
for i, c in enumerate(cache):
    print(f"  layer {i}: {type(c).__name__}")

# Simulate a prefill with ~200 tokens
prompt = "The quick brown fox " * 50  # ~200 tokens
tokens = tokenizer.encode(prompt)
print(f"\nPrefill with {len(tokens)} tokens...")

input_ids = mx.array([tokens])
report("before prefill")

# Run through model
logits = model(input_ids, cache=cache)
mx.eval(logits)
report("after prefill")

# Check cache sizes
total_nbytes = sum(c.nbytes for c in cache)
print(f"Cache reported nbytes: {total_nbytes / 1e6:.1f} MB")

# Check actual memory of decode buffers
deq_buf_bytes = 0
for c in cache:
    if hasattr(c, '_k_deq_buf') and c._k_deq_buf is not None:
        deq_buf_bytes += c._k_deq_buf.nbytes + c._v_deq_buf.nbytes
print(f"Decode buffer actual bytes: {deq_buf_bytes / 1e6:.1f} MB")

# Now simulate decode steps (like generating tokens)
print("\nSimulating 50 decode steps...")
for step in range(50):
    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    logits = model(next_token, cache=cache)
    mx.eval(logits)
    if step % 10 == 9:
        report(f"decode step {step+1}")

total_nbytes = sum(c.nbytes for c in cache)
deq_buf_bytes = 0
for c in cache:
    if hasattr(c, '_k_deq_buf') and c._k_deq_buf is not None:
        deq_buf_bytes += c._k_deq_buf.nbytes + c._v_deq_buf.nbytes
print(f"Cache reported nbytes: {total_nbytes / 1e6:.1f} MB")
print(f"Decode buffer bytes: {deq_buf_bytes / 1e6:.1f} MB")

# Now simulate what the server does: deepcopy and reuse
import copy
print("\n--- Simulating server cache management ---")
report("before deepcopy")

cache2 = copy.deepcopy(cache)
report("after deepcopy")

# Run a second request using the copied cache
prompt2 = "Explain quantum computing in simple terms."
tokens2 = tokenizer.encode(prompt2)
input_ids2 = mx.array([tokens2])
logits2 = model(input_ids2, cache=cache2)
mx.eval(logits2)
report("after 2nd prefill on copy")

# Generate 50 more tokens on the copy
for step in range(50):
    next_token = mx.argmax(logits2[:, -1, :], axis=-1, keepdims=True)
    logits2 = model(next_token, cache=cache2)
    mx.eval(logits2)
report("after 2nd generation (50 steps)")

# Delete the copy and check if memory is freed
del cache2, logits2
gc.collect()
mx.eval(mx.array(0))
report("after deleting copy")

# Make 5 more copies to simulate LRU cache behavior
print("\n--- Simulating LRU with 5 cached copies ---")
copies = []
for i in range(5):
    c = copy.deepcopy(cache)
    copies.append(c)
    report(f"after copy {i+1}")

del copies
gc.collect()
mx.eval(mx.array(0))
report("after deleting all copies")

print("\n--- Checking MLX memory stats ---")
print(f"Active memory: {mx.metal.get_active_memory() / 1e6:.1f} MB")
print(f"Peak memory: {mx.metal.get_peak_memory() / 1e6:.1f} MB")
print(f"Cache memory: {mx.metal.get_cache_memory() / 1e6:.1f} MB")
