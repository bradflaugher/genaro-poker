"""Export the training set and the eval set as a Hugging Face dataset (Parquet).

    uv run python -m genaro.dataset    # -> hf/genaro-poker-sft/data/{train,test}.parquet

train: the exact SFT conversations (system + user prompt from PokerSI, assistant = the
       grammar-shaped JSON reply with the 9-teacher majority move), with metadata.
test:  the 600 held-out decisions genaro.evaluate scores, with equity and the majority move.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from genaro.build import DATA, extra_quiet, relabel, reply

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf" / "genaro-poker-sft" / "data"


def row(r: dict, say: str | None) -> dict:
    return {
        "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                     {"role": "assistant", "content": reply(r, say)}],
        "variant": r["variant"], "street": r["street"], "talk_turn": bool(say), "discard": r["discard"],
        "persona": r["persona"]["name"], "temperature": round(r["persona"]["temperature"], 3),
        "equity": r["equity"], "pot": r["pot"], "to_call": r["to_call"], "big_blind": r["bb"],
        "teacher_action": r["teacher"].get("action", "discard"),
        "single_teacher_action": r.get("single_teacher", r["teacher"]).get("action", "discard"),
        "grammar": r["grammar"], "id": r["id"],
        "speak": r["speak"], "hint": r["hint"],  # PokerSI's TalkPlan for this turn (same schema in both splits)
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(DATA / "train-sft-rows.jsonl")]
    rows += extra_quiet(12000, {r["id"] for r in rows}, seed=1)  # same call as genaro.build
    rows = relabel(rows)
    train = Dataset.from_list([row(r, r.get("say")) for r in rows])
    train.to_parquet(str(OUT / "train.parquet"))

    by = defaultdict(list)  # same stratified sample as genaro.evaluate
    for line in open(DATA / "test-decisions-voted.jsonl"):
        r = json.loads(line)
        by[r["variant"]].append(r)
    rng = random.Random(7)
    test = []
    for v in sorted(by):
        rng.shuffle(by[v])
        test += by[v][:100]
    Dataset.from_list([row({**r, "teacher": {k: v for k, v in r["majority"].items() if k != "votes"}, "single_teacher": r["teacher"]}, None)
                       for r in test]).to_parquet(str(OUT / "test.parquet"))
    print(f"train {len(train)} rows, test {len(test)} rows -> {OUT}")


if __name__ == "__main__":
    main()
