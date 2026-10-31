"""HF checkpoint -> GGUF (f16) -> Q4_K_M with llama.cpp, the format PokerSI loads.

    uv run python -m genaro.export runs/genaro-0.6b/final out/genaro-poker-0.6b-Q4_K_M.gguf --size 0.6B
    uv run python -m genaro.export Qwen/Qwen3-0.6B out/qwen3-0.6b-Q4_K_M.gguf

With --size, GGUF metadata names the model "Genaro Poker <size>" (what model pickers show)
and records license, author and repo links.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP", Path.home() / "Work" / "llama.cpp"))


def resolve(src: str) -> str:
    if Path(src).exists():
        return src
    from huggingface_hub import snapshot_download
    return snapshot_download(src)


def main() -> None:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--size", help="e.g. 0.6B: name the model Genaro Poker <size> in the GGUF metadata")
    a = ap.parse_args()
    src, out = resolve(a.src), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f16 = out.with_suffix(".f16.gguf")
    env = {**os.environ, "PYTHONPATH": str(LLAMA / "gguf-py")}
    cmd = [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), src, "--outfile", str(f16), "--outtype", "f16"]
    if a.size:
        meta = out.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "general.name": f"Genaro Poker {a.size}", "general.basename": "genaro-poker", "general.size_label": a.size,
            "general.author": "Brad Flaugher", "general.organization": "predbrad", "general.license": "apache-2.0",
            "general.description": "A phone-sized poker model fine-tuned to play PokerSI's opponents",
            "general.url": f"https://huggingface.co/predbrad/genaro-poker-{a.size.lower()}",
            "general.repo_url": "https://github.com/bradflaugher/genaro-poker",
        }))
        cmd += ["--metadata", str(meta)]
    subprocess.run(cmd, check=True, env=env)
    if a.size:
        meta.unlink()
    subprocess.run([str(LLAMA / "build" / "bin" / "llama-quantize"), str(f16), str(out), a.quant], check=True)
    f16.unlink()
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
