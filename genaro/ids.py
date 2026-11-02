"""Write the record ids that get extra labels from the bridge: 9-teacher majority votes
(`make votes`) and solver moves (`make solve`). Train ids are the teacher-talk rows plus the
extra quiet rows, as genaro.build selects them; test ids are the 600 eval rows, as
genaro.evaluate samples them.

    uv run python -m genaro.ids [--extra-quiet 24000]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from genaro.build import DATA, extra_quiet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-quiet", type=int, default=12000)
    ap.add_argument("--name", default="vote", help="vote -> data/vote-ids.txt; solve -> data/solve-ids.txt (+ test-*)")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(DATA / "train-sft-rows.jsonl")]
    train = [r["id"] for r in rows] + [r["id"] for r in extra_quiet(a.extra_quiet, {r["id"] for r in rows}, seed=1)]
    by = defaultdict(list)
    for line in open(DATA / "test-decisions.jsonl"):
        r = json.loads(line)
        by[r["variant"]].append(r["id"])
    rng = random.Random(7)  # same sample as genaro.evaluate
    test = []
    for v in sorted(by):
        rng.shuffle(by[v])
        test += by[v][:100]
    (DATA / f"{a.name}-ids.txt").write_text("\n".join(train) + "\n")
    (DATA / f"test-{a.name}-ids.txt").write_text("\n".join(test) + "\n")
    print(f"{len(train)} train ids, {len(test)} test ids")


if __name__ == "__main__":
    main()
