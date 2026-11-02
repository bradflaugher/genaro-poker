"""Teacher table talk: a larger local model writes each talk-turn line given the app's exact
prompt and the chosen action; lines are filtered by length, grammar, PokerSI's own
HandBlur (no leaks) and TableTalk repeat rules.

    uv run python -m genaro.teacher --split train --n 12000 --batch 128 --k 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from genaro import hw

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from genaro import talk  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
TEACHER = "google/gemma-4-E4B-it"  # 26B-A4B works too; E4B is ~3x faster with the same filter yield


def describe(t: dict) -> str:
    if "discard" in t:
        return f"throw away card position(s) {t['discard']}" if t["discard"] else "keep all your cards (stand pat)"
    a, n = t["action"], t["amount"]
    return {"fold": "fold", "check": "check", "call": f"call {n}", "raise": f"raise to {n}", "allin": f"go all in for {n}"}[a]


def teacher_prompt(r: dict) -> list[dict]:
    extra = (
        f"\n\n(Behind the scenes: you have decided to {describe(r['teacher'])}. Now write ONLY your table talk "
        "for \"say\": one short, funny sentence in your voice, under 12 words, family friendly, following the "
        "\"For \\\"say\\\"\" hint above. Never mention any card, rank, suit, pair, draw, hand name, or how good "
        "your hand is. Don't repeat anything already said. No quotation marks, no emoji. Reply with the sentence only.)"
    )
    return [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"] + extra}]


def sample(split: str, n: int, seed: int) -> list[dict]:
    """Stratified by variant so every game gets an equal share."""
    by = defaultdict(list)
    with open(DATA / f"{split}-decisions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by[r["variant"]].append(r)
    rng = random.Random(seed)
    per = n // len(by)
    out = []
    for v, rows in sorted(by.items()):
        rng.shuffle(rows)
        out += rows[:per]
    rng.shuffle(out)
    return out


@torch.inference_mode()
def generate(model, tok, prompts: list[list[dict]], k: int, batch: int) -> list[list[str]]:
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    outs: list[list[str]] = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s : s + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=32, num_return_sequences=k)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        outs += [dec[i : i + k] for i in range(0, len(dec), k)]
        print(f"  {min(s + batch, len(texts))}/{len(texts)}", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--relabel", choices=["majority", "solver"],
                    help="rewrite the lines of talk turns whose move the new label changes (-> train-talk-<label>.jsonl)")
    a = ap.parse_args()

    if a.relabel:
        from genaro.build import labels, word
        new = labels(a.relabel)
        rows = [{**r, "teacher": new[r["id"]]} for r in map(json.loads, open(DATA / "train-sft-rows.jsonl"))
                if r.get("say") and r["id"] in new and word(new[r["id"]]) != word(r["teacher"])]
    else:
        rows = sample(a.split, a.n, a.seed)
    talky = [r for r in rows if r["speak"]]
    print(f"{len(rows)} decisions, {len(talky)} talk turns")
    hw.setup()
    tok = AutoTokenizer.from_pretrained(a.teacher)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation={"": "sdpa", "text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"})
    cands = generate(model, tok, [teacher_prompt(r) for r in talky], a.k, a.batch)
    with open(DATA / (f"train-teacher-raw-{a.relabel}.jsonl" if a.relabel else f"{a.split}-teacher-raw.jsonl"), "w") as f:
        for r, cs in zip(talky, cands):
            f.write(json.dumps({"id": r["id"], "cands": cs}) + "\n")

    # filter: grammar-safe, short, not a repeat, then no leaks per the app's HandBlur
    flat = []
    for r, cs in zip(talky, cands):
        catch_ok = "catchphrase" in (r["hint"] or "")
        previous = r["mine"] + [o["text"] for o in r["others"]] + ([] if catch_ok else [r["persona"]["catchphrase"]])
        for c in dict.fromkeys(talk.clean(c) for c in cs):
            if talk.grammar_ok(c) and not talk.card_talk(c) and 2 <= talk.words(c) <= 12 and len(c) <= talk.SAY_MAX and not talk.is_repeat(c, previous):
                flat.append((r["id"], c, r))
    flags = talk.leaks([{"text": c, "hole": r["hole"], "face_up": r["face_up"], "board": r["board"], "made_hand": r["made_hand"],
                         "short_deck": r["short_deck"], "board_plays": r["board_plays"]} for _, c, r in flat])
    lines: dict[str, str] = {}
    n_leak = 0
    for (rid, c, _), leak in zip(flat, flags):
        n_leak += leak
        if not leak and rid not in lines:
            lines[rid] = c
    print(f"candidates {sum(len(c) for c in cands)}, passed shape/repeat {len(flat)}, leaked {n_leak}, "
          f"talk turns covered {len(lines)}/{len(talky)}")
    if a.relabel:
        with open(DATA / f"train-talk-{a.relabel}.jsonl", "w") as f:
            for rid, say in lines.items():
                f.write(json.dumps({"id": rid, "say": say}) + "\n")
        return
    with open(DATA / f"{a.split}-sft-rows.jsonl", "w") as f:
        for r in rows:
            if r["speak"] and r["id"] not in lines:
                continue  # no clean line: drop rather than teach a bad one
            f.write(json.dumps({**r, "say": lines.get(r["id"])}) + "\n")


if __name__ == "__main__":
    main()
