"""Rescore every saved eval against the same reference and write results/README.md.

    uv run python -m genaro.report

Decision metrics come from each model's stored raw replies, rescored against the
9-teacher majority move and Monte Carlo equity, so older runs and newer runs are compared
like for like. Live-play results come from genaro.play.
"""

from __future__ import annotations

import json
from pathlib import Path

from genaro.evaluate import judge, parse

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

OFFLINE = [  # (results file, label)
    ("qwen3-0.6b-base", "Qwen3 0.6B (base)"),
    ("qwen3-1.7b-base", "Qwen3 1.7B (base)"),
    ("lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
    ("genaro-poker-0.6b-v1", "Genaro 0.6B v1 (single-sample labels)"),
    ("genaro-poker-1.7b-v1", "Genaro 1.7B v1 (single-sample labels)"),
    ("genaro-poker-1.7b-v2", "Genaro 1.7B v2 (majority labels, 1 epoch)"),
    ("genaro-poker-0.6b", "**Genaro 0.6B** (majority labels, 2 epochs)"),
    ("genaro-poker-1.7b", "**Genaro 1.7B** (majority labels, 2 epochs)"),
]
PLAY = [
    ("housebot-1200", "House Bot (the app's heuristic)"),
    ("advice-1200", "Always follow the prompt's advice"),
    ("lfm2.5-1.2b", "LFM2.5 1.2B"),
    ("genaro-poker-0.6b-v1", "Genaro 0.6B v1"),
    ("genaro-poker-0.6b", "**Genaro 0.6B**"),
    ("genaro-poker-1.7b", "**Genaro 1.7B**"),
]


def rows() -> dict[str, dict]:
    out = {}
    for line in open(ROOT / "data" / "test-decisions-voted.jsonl"):
        r = json.loads(line)
        if r.get("majority"):
            out[r["id"]] = {**r, "teacher": {k: v for k, v in r["majority"].items() if k != "votes"}}
    return out


def rescore(name: str, ref: dict[str, dict]) -> dict | None:
    f = RES / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    s, recs = d["summary"], d["records"]
    tot = {"agree": 0, "bad_fold": 0, "bad_call": 0, "missed_value": 0, "cost_bb": 0.0}
    betting = 0
    for rec in recs:
        r = ref[rec["id"]]
        j = judge(r, parse(rec["raw"]))
        betting += not r["discard"]
        for k in tot:
            tot[k] += j[k]
    n = len(recs)
    return {**s, "agree": tot["agree"] / n, "blunders": 100 * (tot["bad_fold"] + tot["bad_call"]) / betting,
            "bad_fold": 100 * tot["bad_fold"] / betting, "cost": 100 * tot["cost_bb"] / betting}


def main() -> None:
    ref = rows()
    lines = ["# Results", "",
             "## Decisions and table talk (600 held-out PokerSI decisions, all six variants)", "",
             "Every model runs through llama.cpp exactly as PokerSI runs it: the app's prompt, its GBNF grammar and "
             "its sampling (persona temperature 0.6-0.9, top-k 40, top-p 0.95). *Agreement* is with the majority "
             "of 9 House Bot teachers using 1,000-sample equity. A *blunder* is folding with equity at least 10 points "
             "above the pot odds, or calling/raising with equity at least 15 points below them (bluff raises by "
             "bluffing personas excepted). *Leaks* are lines PokerSI's own HandBlur would censor.", "",
             "| model | GGUF | agreement | blunders /100 | bad folds /100 | blunder cost (bb/100 decisions) | leaks | card talk | repeats | >12 words | tokens/reply |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, label in OFFLINE:
        s = rescore(name, ref)
        if s:
            lines.append(f"| {label} | {s['size_mb']} MB | {s['agree']:.1%} | {s['blunders']:.1f} | {s['bad_fold']:.1f} | {s['cost']:.1f} | "
                         f"{s['leak_rate']:.1%} | {s['card_talk_rate']:.1%} | {s['repeat_rate']:.1%} | {s['over_12_words']:.1%} | {s['avg_gen_tokens']} |")
    lines += ["", "## Live play (1,200 hands, identical deals for every row)", "",
              "The model plays one seat of a 4-handed table against three House Bots through PokerSI's real engine "
              "(`bridge/Bridge.kt play`), all six variants in rotation, 100 big blinds deep. Win rate has a large "
              "standard error at this sample size; the operational numbers are the robust ones.", "",
              "| seat 0 played by | bb/100 (± s.e.) | fallback to House Bot | leaks | repeats | words/line | tokens/reply |",
              "|---|---|---|---|---|---|---|"]
    for name, label in PLAY:
        f = RES / f"play-{name}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        model = s["mode"] == "server"
        lines.append(f"| {label} | {s['bb_per_100']:+.0f} ± {s.get('bb_per_100_stderr', 0):.0f} | "
                     + (f"{s['fallback_rate']:.2%} | {s['leak_rate']:.1%} | {s['repeat_rate']:.1%} | {s['words_per_line']} | {s['gen_tokens_per_decision']} |"
                        if model else "— | — | — | — | — |"))
    (RES / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
