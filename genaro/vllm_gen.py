"""Batch chat generation with vLLM, for the table-talk teacher (genaro.teacher --engine vllm).

Runs in its own environment (vLLM pins its own torch), so it imports nothing from genaro:

    ~/Work/vllm-env/.venv/bin/python genaro/vllm_gen.py IN.jsonl OUT.jsonl --model google/gemma-4-E4B-it --n 3

IN.jsonl: one {"messages": [...]} per line. OUT.jsonl: one {"outs": [n strings]} per line, same order.
Sampling matches the HF path: temperature 1.0, top-p 0.95, 32 new tokens, n samples per prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    # Unified memory: a big share forces the kernel to evict page cache first (very slow, can OOM).
    ap.add_argument("--gpu-memory", type=float, default=0.3, help="share of the (unified) memory vLLM may take")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from vllm import LLM, SamplingParams

    convs = [json.loads(line)["messages"] for line in open(a.inp)]
    llm = LLM(a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory, max_model_len=4096, seed=a.seed,
              limit_mm_per_prompt={"image": 0, "audio": 0},
              # read the weights into RAM first: copying mmap'd pages to this GPU is pathologically
              # slow (~10 min for Gemma 4 E4B; seconds this way). See train-me's README.
              safetensors_load_strategy="eager")
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
    t = time.perf_counter()
    res = llm.chat(convs, sp)  # same chat template call as the HF path
    dt = time.perf_counter() - t
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps({"outs": [o.text for o in r.outputs]}) + "\n")
    print(f"vllm: {len(convs)} prompts x {a.n} in {dt:.1f}s ({len(convs) / dt:.1f} prompts/s)", flush=True)


if __name__ == "__main__":
    main()
