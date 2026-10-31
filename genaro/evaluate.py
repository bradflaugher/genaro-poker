"""Evaluate a GGUF exactly as PokerSI runs it: llama.cpp server, the app's prompt, grammar and
sampling (persona temperature, top_k 40, top_p 0.95, max 96 tokens, thinking off).

    uv run python -m genaro.evaluate out/genaro-poker-0.6b-Q4_K_M.gguf --name genaro-0.6b

Scores decision quality against Monte Carlo equity (blunders and their cost in big blinds),
table talk (leaks via PokerSI's own HandBlur, repeats, length), and speed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from genaro import talk

ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path(os.environ.get("LLAMA_CPP", Path.home() / "Work" / "llama.cpp"))


def server(gguf: str, port: int, parallel: int, threads: int):
    p = subprocess.Popen([str(LLAMA / "build/bin/llama-server"), "-m", gguf, "--port", str(port), "-c", str(4096 * parallel),
                          "--parallel", str(parallel), "-t", str(threads), "--jinja", "--no-webui", "-ngl", "0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("llama-server did not start")


def ask(port: int, r: dict, seed: int) -> dict:
    body = {"messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            "grammar": r["grammar"], "temperature": r["persona"]["temperature"], "top_k": 40, "top_p": 0.95,
            "max_tokens": 96, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return {"raw": resp["choices"][0]["message"]["content"], "gen_tokens": resp["usage"]["completion_tokens"],
            "prompt_tokens": resp["usage"]["prompt_tokens"], "s": time.time() - t}


def parse(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'"action":\s*"(\w+)",\s*"amount":\s*(\d+)', raw)
        if m:
            return {"action": m.group(1), "amount": int(m.group(2))}
        m = re.search(r'"discard":\s*\[([\d,\s]*)\]', raw)
        return {"discard": [int(x) for x in m.group(1).split(",") if x.strip()]} if m else None


def judge(r: dict, out: dict | None) -> dict:
    """Blunders vs equity. Facing a bet: fold with equity >= pot odds + 10 pts is a bad fold (costs
    the call's EV); calling/raising with equity <= pot odds - 15 pts is a bad call (unless a
    bluff-raise by a bluffing persona). Unopposed with equity >= 80% and only checking = missed value."""
    res = {"valid": out is not None, "agree": False, "bad_fold": 0, "bad_call": 0, "missed_value": 0, "cost_bb": 0.0}
    if out is None:
        return res
    t = r["teacher"]
    if r["discard"]:
        res["agree"] = sorted(out.get("discard", [])) == sorted(t["discard"])
        return res
    a = out.get("action")
    res["agree"] = a == t["action"]
    eq, pot, call, bb = r["equity"], r["pot"], r["to_call"], r["bb"]
    if call > 0:
        odds = call / (pot + call)
        ev_call = (eq * (pot + call) - call) / bb
        if a == "fold" and eq >= odds + 0.10:
            res["bad_fold"], res["cost_bb"] = 1, ev_call
        if a in ("call", "raise", "allin") and eq <= odds - 0.15 and not (a in ("raise", "allin") and r["persona"]["bluffiness"] > 0.5):
            res["bad_call"], res["cost_bb"] = 1, -ev_call
    elif eq >= 0.8 and a == "check" and any(m["word"] in ("raise", "allin") for m in r["menu"]):
        res["missed_value"] = 1
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--name", required=True)
    ap.add_argument("-n", type=int, default=600)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()

    by = defaultdict(list)
    voted = ROOT / "data" / "test-decisions-voted.jsonl"  # same records + the 9-teacher majority
    for line in open(voted if voted.exists() else ROOT / "data" / "test-decisions.jsonl"):
        r = json.loads(line)
        if r.get("majority"):
            r = {**r, "single_teacher": r["teacher"], "teacher": {k: v for k, v in r["majority"].items() if k != "votes"}}
        by[r["variant"]].append(r)
    rng = random.Random(7)
    rows = []
    for v in sorted(by):
        rng.shuffle(by[v])
        rows += by[v][: a.n // len(by)]

    srv = server(a.gguf, a.port, a.parallel, threads=16)
    try:
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            outs = list(ex.map(lambda ir: ask(a.port, ir[1], 1000 + ir[0]), enumerate(rows)))
    finally:
        srv.kill()

    recs, tot = [], defaultdict(float)
    talk_rows = []
    for r, o in zip(rows, outs):
        p = parse(o["raw"])
        j = judge(r, p)
        say = (p or {}).get("say") if r["speak"] else None
        rec = {"id": r["id"], "variant": r["variant"], "speak": r["speak"], "raw": o["raw"], "teacher": r["teacher"],
               "equity": r["equity"], **j, "gen_tokens": o["gen_tokens"], "say": say}
        recs.append(rec)
        for k in ("valid", "agree", "bad_fold", "bad_call", "missed_value", "cost_bb", "gen_tokens"):
            tot[k] += float(rec[k])
        if r["speak"]:
            tot["talk_turns"] += 1
            if say:
                tot["lines"] += 1
                tot["long"] += talk.words(say) > 12
                tot["card_talk"] += talk.card_talk(say)
                tot["repeat"] += talk.is_repeat(say, r["mine"] + [x["text"] for x in r["others"]])
                talk_rows.append((rec, {"text": say, "hole": r["hole"], "face_up": r["face_up"], "board": r["board"],
                                        "made_hand": r["made_hand"], "short_deck": r["short_deck"], "board_plays": r["board_plays"]}))
    for (rec, _), leak in zip(talk_rows, talk.leaks([x for _, x in talk_rows])):
        rec["leak"] = leak
        tot["leak"] += leak
    n, lines = len(rows), max(tot["lines"], 1)
    betting = sum(1 for r in rows if not r["discard"])
    summary = {
        "name": a.name, "gguf": a.gguf, "size_mb": round(os.path.getsize(a.gguf) / 1e6), "n": n,
        "valid": round(tot["valid"] / n, 4), "teacher_agreement": round(tot["agree"] / n, 4),
        "blunders_per_100": round(100 * (tot["bad_fold"] + tot["bad_call"]) / betting, 2),
        "bad_fold_per_100": round(100 * tot["bad_fold"] / betting, 2), "bad_call_per_100": round(100 * tot["bad_call"] / betting, 2),
        "missed_value_per_100": round(100 * tot["missed_value"] / betting, 2),
        "blunder_cost_bb_per_100": round(100 * tot["cost_bb"] / betting, 1),
        "talk_turn_has_line": round(tot["lines"] / max(tot["talk_turns"], 1), 4),
        "leak_rate": round(tot["leak"] / lines, 4), "card_talk_rate": round(tot["card_talk"] / lines, 4),
        "repeat_rate": round(tot["repeat"] / lines, 4), "over_12_words": round(tot["long"] / lines, 4),
        "avg_gen_tokens": round(tot["gen_tokens"] / n, 1),
    }
    out = ROOT / "results" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": recs}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
