"""Fill the results tables into the Hugging Face cards (cards/*.md -> <hf clone>/README.md).

    uv run python -m genaro.cards ~/Work/hf     # expects Genaro-poker/, Genaro-poker-1.7B/, genaro-poker-sft/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from genaro.report import RES, paired, rescore, rows

ROOT = Path(__file__).resolve().parents[1]
CARDS = {  # card -> (HF clone dir, rows: (offline results, live results, label))
    "genaro-poker-0.6b": ("Genaro-poker", [
        ("qwen3-0.6b-base", None, "Qwen3 0.6B (base)"),
        ("lfm2.5-1.2b", "lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
        ("genaro-poker-0.6b", "genaro-poker-0.6b", "Genaro Poker 0.6B v2 (House Bot labels)"),
        ("genaro-poker-0.6b-weighted", "genaro-poker-0.6b-weighted-3600", "**Genaro Poker 0.6B v3**")]),
    "genaro-poker-1.7b": ("Genaro-poker-1.7B", [
        ("qwen3-1.7b-base", None, "Qwen3 1.7B (base)"),
        ("lfm2.5-1.2b", "lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
        ("genaro-poker-1.7b", "genaro-poker-1.7b", "Genaro Poker 1.7B v2 (House Bot labels)"),
        ("genaro-poker-0.6b-weighted", "genaro-poker-0.6b-weighted-3600", "Genaro Poker 0.6B v3"),
        ("genaro-poker-1.7b-solver", "genaro-poker-1.7b-solver-3600", "**Genaro Poker 1.7B v3**")]),
}


def common_deals(live: str) -> str:
    """Win rate on the 1,200 deals every live run played (worker w's first 100 hands)."""
    import statistics as st
    d = json.loads((RES / f"play-{live}.json").read_text())
    per_hand = d.get("per_hand_bb")
    if not per_hand:  # a 1,200-hand run is exactly those deals
        s = d["summary"]
        return f"{s['bb_per_100']:+.0f} ± {s['bb_per_100_stderr']:.0f}"
    per = len(per_hand) // 12
    sub = [per_hand[w * per + k] for w in range(12) for k in range(100)]
    return f"{100 * st.mean(sub):+.0f} ± {100 * st.stdev(sub) / len(sub) ** 0.5:.0f}"


def table(show: list[tuple[str, str | None, str]], ref: dict) -> str:
    out = ["| model | GGUF | EV regret (bb/decision) | big mistakes /100 | leaks | card talk | repeats | >12 words | live bb/100 (same 1,200 deals) | live vs House Bot, paired (3,600 hands) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, live, label in show:
        s = rescore(name, ref)
        if not s:
            continue
        p = RES / f"play-{live}.json" if live else None
        lv = json.loads(p.read_text())["summary"] if p and p.exists() else None
        raw = common_deals(live) if lv else "—"
        out.append(f"| {label} | {s['size_mb']} MB | {s['regret']:.2f} | {s['big_mistakes']:.1f} | {s['leak_rate']:.1%} | "
                   f"{s['card_talk_rate']:.1%} | {s['repeat_rate']:.1%} | {s['over_12_words']:.1%} | {raw} | {paired(live) if lv else '—'} |")
    return "\n".join(out)


def main() -> None:
    hf = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    ref = rows()
    for card, (repo, show) in CARDS.items():
        text = (ROOT / "cards" / f"{card}.md").read_text().replace("<!-- RESULTS -->", table(show, ref))
        if hf:
            (hf / repo / "README.md").write_text(text)
        print(f"== {card}\n{table(show, ref)}\n")
    if hf:
        (hf / "genaro-poker-sft" / "README.md").write_text((ROOT / "cards" / "dataset.md").read_text())


if __name__ == "__main__":
    main()
