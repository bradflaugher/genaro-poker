"""Fill the results tables into the Hugging Face cards (cards/*.md -> <hf clone>/README.md).

    uv run python -m genaro.cards ~/Work/hf     # expects Genaro-poker/, Genaro-poker-1.7B/, genaro-poker-sft/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from genaro.report import RES, rescore, rows

ROOT = Path(__file__).resolve().parents[1]
CARDS = {  # card -> (HF clone dir, rows to show)
    "genaro-poker-0.6b": ("Genaro-poker", [("qwen3-0.6b-base", "Qwen3 0.6B (base)"), ("lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
                                            ("genaro-poker-0.6b", "**Genaro Poker 0.6B**")]),
    "genaro-poker-1.7b": ("Genaro-poker-1.7B", [("qwen3-1.7b-base", "Qwen3 1.7B (base)"), ("lfm2.5-1.2b", "LFM2.5 1.2B (PokerSI's previous default)"),
                                                ("genaro-poker-0.6b", "Genaro Poker 0.6B"), ("genaro-poker-1.7b", "**Genaro Poker 1.7B**")]),
}


def table(show: list[tuple[str, str]], ref: dict) -> str:
    out = ["| model | GGUF | agreement | blunders /100 | leaks | card talk | repeats | >12 words | live play bb/100 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, label in show:
        s = rescore(name, ref)
        if not s:
            continue
        p = RES / f"play-{name}.json"
        live = json.loads(p.read_text())["summary"] if p.exists() else None
        lv = f"{live['bb_per_100']:+.0f} ± {live['bb_per_100_stderr']:.0f}" if live else "—"
        out.append(f"| {label} | {s['size_mb']} MB | {s['agree']:.1%} | {s['blunders']:.1f} | {s['leak_rate']:.1%} | "
                   f"{s['card_talk_rate']:.1%} | {s['repeat_rate']:.1%} | {s['over_12_words']:.1%} | {lv} |")
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
