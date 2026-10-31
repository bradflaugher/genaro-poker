"""Write the record ids that get 9-teacher majority votes (`make votes`): every training row
(the teacher-talk rows plus the extra quiet rows, as genaro.build selects them) and the 600
eval rows (as genaro.evaluate samples them).

    uv run python -m genaro.ids
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from genaro.build import DATA, extra_quiet


def main() -> None:
    rows = [json.loads(line) for line in open(DATA / "train-sft-rows.jsonl")]
    train = [r["id"] for r in rows] + [r["id"] for r in extra_quiet(12000, {r["id"] for r in rows}, seed=1)]
    by = defaultdict(list)
    for line in open(DATA / "test-decisions.jsonl"):
        r = json.loads(line)
        by[r["variant"]].append(r["id"])
    rng = random.Random(7)  # same sample as genaro.evaluate
    test = []
    for v in sorted(by):
        rng.shuffle(by[v])
        test += by[v][:100]
    (DATA / "vote-ids.txt").write_text("\n".join(train) + "\n")
    (DATA / "test-vote-ids.txt").write_text("\n".join(test) + "\n")
    print(f"{len(train)} train ids, {len(test)} test ids")


if __name__ == "__main__":
    main()
