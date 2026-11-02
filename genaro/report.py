"""Rescore every saved eval against the same reference and write results/README.md.

    uv run python -m genaro.report

Decision metrics come from each model's stored raw replies, rescored against the
9-teacher majority move and Monte Carlo equity, so older runs and newer runs are compared
like for like. Live-play results come from genaro.play.
"""

from __future__ import annotations

import json
from pathlib import Path

from genaro.evaluate import judge, parse, regret

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

OFFLINE = [  # (results file, label)
    ("qwen3-0.6b-base", "Qwen3 0.6B (base)"),
    ("qwen3-1.7b-base", "Qwen3 1.7B (base)"),
    ("lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
    ("genaro-poker-0.6b-v1", "Genaro 0.6B v1 (one House Bot sample)"),
    ("genaro-poker-0.6b", "Genaro 0.6B v2 (House Bot majority)"),
    ("genaro-poker-1.7b", "Genaro 1.7B v2 (House Bot majority)"),
    ("genaro-poker-0.6b-solver", "Genaro 0.6B v3, plain SFT (solver labels)"),
    ("genaro-poker-0.6b-dagger", "Genaro 0.6B v3 + DAgger (solver labels)"),
    ("genaro-poker-0.6b-weighted", "**Genaro 0.6B v3**, EV-weighted (solver labels, shipped)"),
    ("genaro-poker-1.7b-solver", "**Genaro 1.7B v3** (solver labels, shipped)"),
]
PLAY = [  # (play results file, label); "house" marks the baseline the paired column uses
    ("housebot-3600", "House Bot (the app's heuristic)"),
    ("solver-3600", "The solver itself (the teacher)"),
    ("advice-1200", "Always follow the prompt's advice"),
    ("lfm2.5-1.2b", "LFM2.5 1.2B"),
    ("genaro-poker-0.6b-v1", "Genaro 0.6B v1"),
    ("genaro-poker-0.6b", "Genaro 0.6B v2"),
    ("genaro-poker-1.7b", "Genaro 1.7B v2"),
    ("genaro-poker-0.6b-solver-3600", "Genaro 0.6B v3, plain SFT"),
    ("genaro-poker-0.6b-dagger-3600", "Genaro 0.6B v3 + DAgger"),
    ("genaro-poker-0.6b-weighted-3600", "**Genaro 0.6B v3**, EV-weighted (shipped)"),
    ("genaro-poker-1.7b-solver-3600", "**Genaro 1.7B v3** (shipped)"),
]


def paired(name: str) -> str:
    """Win rate minus House Bot's in the same seat on the same deals: most of the card luck cancels."""
    import statistics as st
    a = json.loads((RES / f"play-{name}.json").read_text()).get("per_hand_bb")
    b = json.loads((RES / "play-housebot-3600.json").read_text()).get("per_hand_bb") if (RES / "play-housebot-3600.json").exists() else None
    if not a or not b or name.startswith("housebot"):
        return "—"
    # worker w played hands w*per .. w*per+per-1; a 1,200-hand run is each worker's first 100 of 300
    per_a, per_b = len(a) // 12, len(b) // 12
    pairs = [(a[w * per_a + k], b[w * per_b + k]) for w in range(12) for k in range(min(per_a, per_b))]
    d = [x - y for x, y in pairs]
    return f"{100 * st.mean(d):+.0f} ± {100 * st.stdev(d) / len(d) ** 0.5:.0f}"


def rows() -> dict[str, dict]:
    out = {}
    for line in open(ROOT / "data" / "test-decisions-voted.jsonl"):
        r = json.loads(line)
        if r.get("majority"):
            out[r["id"]] = {**r, "teacher": {k: v for k, v in r["majority"].items() if k != "votes"}}
    solved = ROOT / "data" / "test-decisions-solved.jsonl"
    if solved.exists():
        for line in open(solved):
            r = json.loads(line)
            if r.get("solver") and r["id"] in out:
                out[r["id"]]["solver"] = r["solver"]
    return out


def rescore(name: str, ref: dict[str, dict]) -> dict | None:
    f = RES / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    s, recs = d["summary"], d["records"]
    tot = {"agree": 0, "bad_fold": 0, "bad_call": 0, "missed_value": 0, "cost_bb": 0.0}
    betting = 0
    reg, solver_agree, big, nsolved = 0.0, 0, 0, 0
    for rec in recs:
        r = ref[rec["id"]]
        out = parse(rec["raw"])
        j = judge(r, out)
        betting += not r["discard"]
        for k in tot:
            tot[k] += j[k]
        if r.get("solver"):
            g, same = regret(r["solver"], out)
            reg += g
            solver_agree += same
            big += g > 2.0
            nsolved += 1
    n = len(recs)
    return {**s, "agree": tot["agree"] / n, "blunders": 100 * (tot["bad_fold"] + tot["bad_call"]) / betting,
            "bad_fold": 100 * tot["bad_fold"] / betting, "cost": 100 * tot["cost_bb"] / betting,
            "regret": reg / max(nsolved, 1), "solver_agree": solver_agree / max(nsolved, 1), "big_mistakes": 100 * big / max(nsolved, 1)}


def main() -> None:
    ref = rows()
    lines = ["# Results", "",
             "## Decisions and table talk (600 held-out PokerSI decisions, all six variants)", "",
             "Every model runs through llama.cpp exactly as PokerSI runs it: the app's prompt, its GBNF grammar and "
             "its sampling (persona temperature 0.6-0.9, top-k 40, top-p 0.95).", "",
             "- *EV regret*: big blinds the move gives up against the solver's best move, from the solver's own EV "
             "estimate of every move (480 rollouts each, `bridge/Solver.kt`). *Big mistakes*: moves that give up more "
             "than 2 bb. Lower is better.",
             "- *Solver / House Bot agreement*: same move word as the solver, or as the majority of 9 House Bots.",
             "- *Leaks*: lines PokerSI's own HandBlur would censor. *Card talk*: any card or hand word at all.", "",
             "| model | GGUF | EV regret (bb/decision) | big mistakes /100 | solver agreement | House Bot agreement | leaks | card talk | repeats | >12 words |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, label in OFFLINE:
        s = rescore(name, ref)
        if s:
            lines.append(f"| {label} | {s['size_mb']} MB | {s['regret']:.2f} | {s['big_mistakes']:.1f} | {s['solver_agree']:.1%} | {s['agree']:.1%} | "
                         f"{s['leak_rate']:.1%} | {s['card_talk_rate']:.1%} | {s['repeat_rate']:.1%} | {s['over_12_words']:.1%} |")
    lines += ["", "## Live play", "",
              "One seat of a 4-handed table against three House Bots through PokerSI's real engine "
              "(`bridge/Bridge.kt play`), all six variants in rotation, 100 big blinds deep, identical deals for every "
              "row. *vs House Bot* is the paired difference against House Bot playing the same seat on the same deals, "
              "which cancels most of the card luck; raw win rates are much noisier.", "",
              "| seat 0 played by | hands | bb/100 (± s.e.) | vs House Bot, paired (bb/100) | leaks | repeats | words/line |",
              "|---|---|---|---|---|---|---|"]
    for name, label in PLAY:
        f = RES / f"play-{name}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        model = s["mode"] == "server"
        lines.append(f"| {label} | {s['hands']:,} | {s['bb_per_100']:+.0f} ± {s.get('bb_per_100_stderr', 0):.0f} | {paired(name)} | "
                     + (f"{s['leak_rate']:.1%} | {s['repeat_rate']:.1%} | {s['words_per_line']} |" if model else "— | — | — |"))
    (RES / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
