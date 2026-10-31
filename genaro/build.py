"""Build the SFT set: PokerSI's exact prompts -> the exact JSON its grammar allows.

    uv run python -m genaro.build --model Qwen/Qwen3-0.6B --extra-quiet 12000

* talk turns: teacher action + the filtered teacher line (genaro.teacher)
* quiet turns: teacher action only, no "say" (as the app's quiet-turn grammar requires)
* the prompt is rendered with the model's own chat template, thinking off, exactly as
  llama.cpp renders it on the phone; loss is on the reply (+ end-of-turn) only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def reply(r: dict, say: str | None) -> str:
    """Byte-identical to what PokerGrammar allows (key order, spacing, list format)."""
    t = r["teacher"]
    if "discard" in t:
        head = '{"discard": [' + ", ".join(str(i) for i in t["discard"]) + "]"
    else:
        head = '{"action": "' + t["action"] + '", "amount": ' + str(t["amount"])
    return head + (', "say": "' + say + '"}' if say else "}")


def prompt_text(tok, r: dict) -> str:
    return tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extra_quiet(n: int, exclude: set[str], seed: int) -> list[dict]:
    """More decision practice for free: quiet turns need no teacher line."""
    by = defaultdict(list)
    for line in open(DATA / "train-decisions.jsonl"):
        r = json.loads(line)
        if not r["speak"] and r["id"] not in exclude:
            by[r["variant"]].append(r)
    rng = random.Random(seed)
    out = []
    for v in sorted(by):
        rng.shuffle(by[v])
        out += by[v][: n // len(by)]
    return out


def relabel(rows: list[dict]) -> list[dict]:
    """Swap the single-sample teacher move for the 9-teacher majority. A talk turn whose line
    was written for a different move is dropped rather than mismatched."""
    voted = {}
    for line in open(DATA / "train-decisions-voted.jsonl"):
        r = json.loads(line)
        if r.get("majority"):
            voted[r["id"]] = {k: v for k, v in r["majority"].items() if k != "votes"}
    out, dropped = [], 0
    for r in rows:
        m = voted[r["id"]]
        if r.get("say") and m.get("action", "discard") != r["teacher"].get("action", "discard"):
            dropped += 1
            continue
        out.append({**r, "teacher": m, "single_teacher": r["teacher"]})
    print(f"relabelled {len(out)} rows with the majority move; dropped {dropped} talk turns whose line fit another move")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--extra-quiet", type=int, default=12000)
    ap.add_argument("--out", default="sft")
    ap.add_argument("--label", choices=["teacher", "majority"], default="majority",
                    help="majority: the consensus of 9 independent teachers (bridge dump --vote-ids)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    end = tok.eos_token if "<|im_end|>" not in tok.get_vocab() else "<|im_end|>"

    rows = [json.loads(line) for line in open(DATA / "train-sft-rows.jsonl")]
    rows += extra_quiet(a.extra_quiet, {r["id"] for r in rows}, seed=1)
    if a.label == "majority":
        rows = relabel(rows)
    ids, labels, kinds = [], [], defaultdict(int)
    for r in rows:
        p = tok(prompt_text(tok, r), add_special_tokens=False)["input_ids"]
        c = tok(reply(r, r.get("say")) + end, add_special_tokens=False)["input_ids"]
        ids.append(p + c)
        labels.append([-100] * len(p) + c)
        kinds[("talk" if r.get("say") else "quiet") + ("/discard" if r["discard"] else "")] += 1
    ds = Dataset.from_dict({"input_ids": ids, "labels": labels}).shuffle(seed=42)
    lens = sorted(map(len, ids))
    print(f"{len(ds)} examples {dict(kinds)} | tokens {sum(lens):,} | median {lens[len(lens) // 2]} | max {lens[-1]}")
    ds.save_to_disk(str(DATA / a.out))
    print("example:\n" + tok.decode(ids[0]))


if __name__ == "__main__":
    main()
