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


LABELS = {  # label source -> (bridge dump with the labels, field)
    "majority": ("train-decisions-voted.jsonl", "majority"),  # 9 House Bot teachers
    "solver": ("train-decisions-solved.jsonl", "solver"),  # search solver (bridge/Solver.kt)
}
MOVE_KEYS = ("action", "amount", "discard")


def labels(source: str, split: str = "train") -> dict[str, dict]:
    path, field = LABELS[source]
    out = {}
    for line in open(DATA / path.replace("train-", f"{split}-")):
        r = json.loads(line)
        if r.get(field):
            out[r["id"]] = {k: v for k, v in r[field].items() if k in MOVE_KEYS}
    return out


def word(move: dict) -> str:
    return move.get("action", "discard")


def relabel(rows: list[dict], source: str = "majority") -> list[dict]:
    """Swap the single-sample teacher move for a better label. A talk turn whose line was
    written for a different move gets the line rewritten for the new move
    (data/train-talk-<source>.jsonl, from genaro.teacher --relabel) or is dropped."""
    new = labels(source)
    f = DATA / f"train-talk-{source}.jsonl"
    rewritten = {r["id"]: r["say"] for r in map(json.loads, open(f))} if f.exists() else {}
    out, dropped, swapped = [], 0, 0
    for r in rows:
        m = new[r["id"]]
        if r.get("say") and word(m) != word(r["teacher"]):
            if r["id"] not in rewritten:
                dropped += 1
                continue
            r = {**r, "say": rewritten[r["id"]]}
            swapped += 1
        out.append({**r, "teacher": m, "single_teacher": r["teacher"]})
    print(f"relabelled {len(out)} rows ({source}); {swapped} talk lines rewritten for the new move, {dropped} dropped")
    return out


def stake(ev_bb: list[dict]) -> float:
    """EV at stake in a decision: the solver's best move minus its second best, in big blinds."""
    evs = sorted((e["ev"] for e in ev_bb), reverse=True)
    return evs[0] - evs[1] if len(evs) > 1 else 0.0


def weight_by_gap(rows: list[dict]) -> list[dict]:
    """Cost-sensitive SFT by repetition: 1 copy, +1 if more than 1 bb is at stake, +1 above
    3 bb, +1 for discards (the weakest spot). Most decisions are near-ties where any move is
    fine; the ones that cost chips should dominate the loss."""
    evs = {}
    for line in open(DATA / "train-decisions-solved.jsonl"):
        r = json.loads(line)
        if r.get("solver"):
            evs[r["id"]] = r["solver"]["ev_bb"]
    out = []
    for r in rows:
        g = stake(evs[r["id"]]) if r["id"] in evs else 0.0
        out += [r] * (1 + (g > 1) + (g > 3) + bool(r["discard"]))
    print(f"weighted by EV at stake: {len(rows)} -> {len(out)} rows")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--extra-quiet", type=int, default=24000)
    ap.add_argument("--out", default="sft")
    ap.add_argument("--weight-by-gap", action="store_true",
                    help="cost-sensitive: repeat decisions with more EV at stake (solver's best vs second-best move)")
    ap.add_argument("--dagger", nargs="*", default=[],
                    help="DAgger rows from genaro.play --collect: states the model itself reached, solver-labelled")
    ap.add_argument("--label", choices=["teacher", "majority", "solver"], default="solver",
                    help="majority: 9 House Bot teachers (dump --vote-ids); solver: bridge/Solver.kt (dump --solve-ids)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    end = tok.eos_token if "<|im_end|>" not in tok.get_vocab() else "<|im_end|>"

    rows = [json.loads(line) for line in open(DATA / "train-sft-rows.jsonl")]
    rows += extra_quiet(a.extra_quiet, {r["id"] for r in rows}, seed=1)
    if a.label != "teacher":
        rows = relabel(rows, a.label)
    if a.weight_by_gap:
        rows = weight_by_gap(rows)
    for path in a.dagger:
        extra = [r for r in map(json.loads, open(path)) if not (r["speak"] and not r.get("say"))]
        print(f"DAgger {path}: {len(extra)} rows (talk turns without a clean, matching line dropped)")
        rows += extra
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
