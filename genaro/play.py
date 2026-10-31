"""End-to-end: real PokerSI hands (all six variants), one seat played by the model through a
llama.cpp server exactly like the app's LlmAgent, three House Bots, identical deals for
every model. Baselines need no model: `--mode housebot` (the app's fallback) and
`--mode advice` (always follow the prompt's Advice line).

    uv run python -m genaro.play out/genaro-poker-0.6b-Q4_K_M.gguf --name genaro-0.6b
    uv run python -m genaro.play --mode housebot --name housebot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from genaro.evaluate import server

ROOT = Path(__file__).resolve().parents[1]
JAR = ROOT / "bridge" / "bridge.jar"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="server", choices=["server", "housebot", "advice"])
    ap.add_argument("--hands", type=int, default=600, help="split across --workers")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    srv = server(a.gguf, a.port, a.workers, threads=18) if a.mode == "server" else None
    per = a.hands // a.workers
    tmp = Path(tempfile.mkdtemp())
    try:
        procs = [subprocess.Popen(["mise", "exec", "--", "java", "-jar", str(JAR), "play", "--server", f"http://127.0.0.1:{a.port}",
                                   "--mode", a.mode, "--hands", str(per), "--seed", str(w + 1), "--out", str(tmp / f"{w}.json"),
                                   "--log", str(tmp / f"{w}.log")], cwd=ROOT, stderr=open(tmp / f"{w}.err", "w"))
                 for w in range(a.workers)]
        for p in procs:
            p.wait()
    finally:
        if srv:
            srv.kill()
    missing = [w for w in range(a.workers) if not (tmp / f"{w}.json").exists()]
    if missing:
        err = (tmp / f"{missing[0]}.err").read_text()[-3000:]
        raise SystemExit(f"workers {missing} failed (logs in {tmp}):\n{err}")
    parts = [json.loads((tmp / f"{w}.json").read_text()) for w in range(a.workers)]
    tot = {k: sum(p.get(k, 0.0) for p in parts) for k in set().union(*parts)}
    d = max(tot.get("decisions", 0), 1)
    lines = max(tot.get("lines", 0), 1)
    summary = {
        "name": a.name, "mode": a.mode, "hands": int(tot["hands"]),
        "bb_per_100": round(100 * tot["bb_sum"] / tot["hands"], 1),
        "bb_per_100_stderr": round(100 * ((tot["bb_sq"] / tot["hands"] - (tot["bb_sum"] / tot["hands"]) ** 2) / tot["hands"]) ** 0.5, 1),
        "model_decisions": int(tot.get("decisions", 0)),
        "fallback_rate": round(tot.get("fallbacks", 0) / d, 4),
        "ms_per_decision_parallel": round(tot.get("ms", 0) / d),
        "gen_tokens_per_decision": round(tot.get("completion_tokens", 0) / d, 1),
        "leak_rate": round(tot.get("leaks", 0) / lines, 4),
        "repeat_rate": round(tot.get("repeats", 0) / lines, 4),
        "over_12_words": round(tot.get("long_lines", 0) / lines, 4),
        "words_per_line": round(tot.get("words", 0) / lines, 1),
    }
    out = ROOT / "results" / f"play-{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    logs = []
    for w in range(a.workers):
        f = tmp / f"{w}.log"
        if f.exists():
            logs += f.read_text().splitlines()[:50]
    out.write_text(json.dumps({"summary": summary, "sample_log": [json.loads(x) for x in logs]}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
