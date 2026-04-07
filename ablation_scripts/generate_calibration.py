#!/usr/bin/env python3
"""Generate 64 calibration samples from each optimal model.

Produces a JSON file per model with prompt/response pairs that serve as
the ground-truth baseline for PLE/embedding quantization ablation.

Usage:
    .venv/bin/python optimal/generate_calibration.py
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ── 64 diverse calibration prompts ───────────────────────────────────────────

PROMPTS = [
    # Math / reasoning (8)
    "What is 2+2? Answer with just the number.",
    "Solve: 17 × 23. Show your work step by step.",
    "If a train travels 120 km in 2 hours, what is its average speed in m/s?",
    "What is the derivative of x³ + 2x² - 5x + 3?",
    "A box has 3 red, 5 blue, and 2 green balls. What's the probability of picking blue?",
    "What is the sum of the first 100 positive integers?",
    "Convert 0.375 to a fraction in simplest form.",
    "If f(x) = 2x + 1 and g(x) = x², what is f(g(3))?",

    # Code generation (8)
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python one-liner to flatten a list of lists.",
    "Implement binary search in Python.",
    "Write a Python function to compute the nth Fibonacci number using memoization.",
    "Write a bash command to find all .py files larger than 1MB.",
    "Write a Python function that reverses a linked list.",
    "Write a list comprehension that generates all prime numbers under 50.",
    "Write a Python decorator that measures execution time of a function.",

    # Knowledge / factual (8)
    "What are the three laws of thermodynamics? Be concise.",
    "Name the planets in our solar system in order from the Sun.",
    "What is the capital of Australia?",
    "Who wrote 'Pride and Prejudice' and in what year was it published?",
    "Explain the difference between DNA and RNA in 2 sentences.",
    "What is the speed of light in meters per second?",
    "What are the noble gases? List all of them.",
    "What is photosynthesis? Explain in one sentence.",

    # Translation (4)
    "Translate 'The cat sat on the mat' into French.",
    "Translate 'Good morning, how are you?' into Japanese.",
    "Translate 'I love programming' into German.",
    "Translate 'The weather is beautiful today' into Spanish.",

    # Explanation / teaching (8)
    "Explain quantum entanglement in 2 sentences.",
    "What is a neural network? Explain like I'm 10 years old.",
    "Explain the concept of recursion with a simple example.",
    "What is the difference between TCP and UDP?",
    "Explain how a hash table works in simple terms.",
    "What is the difference between a compiler and an interpreter?",
    "Explain the CAP theorem in distributed systems.",
    "What is gradient descent? Explain briefly.",

    # Creative writing (8)
    "Write a haiku about the ocean.",
    "Complete this sentence creatively: 'The robot looked at the sunset and...'",
    "Write a one-paragraph short story about a time traveler.",
    "Create a limerick about a programmer.",
    "Write a metaphor comparing life to a river.",
    "Describe a futuristic city in 3 sentences.",
    "Write a tongue twister about technology.",
    "Create an acrostic poem using the word PYTHON.",

    # Instruction following (8)
    "List exactly 5 fruits that are red. Number them.",
    "Summarize the concept of machine learning in exactly 20 words.",
    "Give me 3 pros and 3 cons of remote work in bullet points.",
    "Explain the water cycle using only words that start with the letter 'W' or 'S'.",
    "Name 3 programming languages and one unique feature of each.",
    "Create a simple meal plan for one day (breakfast, lunch, dinner).",
    "List the steps to make a peanut butter sandwich, in order.",
    "Give me an acronym for SMART goals and explain each letter.",

    # Logic / puzzles (4)
    "I have a brother. My brother has a brother. How many brothers minimum?",
    "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "What comes next: 2, 6, 12, 20, 30, ?",

    # Edge cases / harder (8)
    "Explain the P vs NP problem in 3 sentences.",
    "What are the key differences between transformers and RNNs in deep learning?",
    "Write a SQL query to find the second highest salary from an employees table.",
    "Explain why 0.1 + 0.2 != 0.3 in floating point arithmetic.",
    "What is the halting problem and why is it undecidable?",
    "Explain big-O notation with examples of O(1), O(n), and O(n²).",
    "What is the difference between concurrency and parallelism?",
    "Describe how public-key cryptography works in simple terms.",
]

assert len(PROMPTS) == 64, f"Expected 64 prompts, got {len(PROMPTS)}"

# ── Models ────────────────────────────────────────────────────────────────────

MODELS = {
    "E2B": "mlx/gemma-4-E2B-q3-g64-optimal",
    "E4B": "mlx/gemma-4-E4B-e4ne4g64-optimal",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "calibration")
MAX_TOKENS = 200


def generate_calibration(model_name: str, model_path: str):
    """Generate responses for all 64 prompts and save to JSON."""
    print(f"\n{'='*80}")
    print(f"  Generating calibration: {model_name}")
    print(f"  Model: {model_path}")
    print(f"{'='*80}\n")

    model, tok = load(model_path)
    sampler = make_sampler(temp=0.0)  # greedy for reproducibility

    results = []
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        t1 = time.time()
        response = generate(model, tok, prompt=chat, max_tokens=MAX_TOKENS,
                            sampler=sampler, verbose=False)
        dt = time.time() - t1

        n_tok = len(tok.encode(response))
        results.append({
            "id": i,
            "prompt": prompt,
            "response": response,
            "tokens": n_tok,
            "time_s": round(dt, 2),
            "tok_per_sec": round(n_tok / dt, 1) if dt > 0 else 0,
        })

        # Progress
        if (i + 1) % 8 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:2d}/64] {elapsed:.0f}s  last: {dt:.1f}s ({n_tok/dt:.0f} tok/s)")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} samples in {elapsed:.0f}s")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"calibration_{model_name.lower()}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "model_path": model_path,
            "n_prompts": len(results),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "samples": results,
        }, f, indent=2)
    print(f"  Saved: {out_path}")

    # Cleanup
    del model, tok
    gc.collect()
    mx.clear_cache()

    return results


def main():
    for name, path in MODELS.items():
        generate_calibration(name, path)
    print("\n✓ Calibration complete for all models.")


if __name__ == "__main__":
    main()
