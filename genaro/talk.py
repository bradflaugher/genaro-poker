"""PokerSI's table-talk rules, mirrored from engine/TableTalk.kt and llm/PokerGrammar.kt,
plus a bridge call to the app's real HandBlur for leak checks."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

SAY_MAX = 80  # PokerGrammar.SAY_MAX
SIMILAR = 0.6  # TableTalk.SIMILAR
BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "bridge.jar"

_non_word = re.compile(r"[^a-z0-9' ]+")
_spaces = re.compile(r"\s+")


def normalize(text: str) -> str:
    t = text.lower().replace("\u2019", "'")
    t = _non_word.sub(" ", t).replace("'", "")
    return _spaces.sub(" ", t).strip()


def overlap(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def is_repeat(text: str, previous: list[str]) -> bool:
    w = normalize(text)
    if not w:
        return False
    return any(w == normalize(p) or overlap(w, normalize(p)) >= SIMILAR for p in previous)


def grammar_ok(say: str) -> bool:
    """The app's grammar allows any char except a double quote, backslash or control char; also
    reject leftovers of a JSON/dict reply the teacher wrote instead of a line."""
    return bool(say) and not re.search(r'["\\\x00-\x1f{}]', say) and not re.search(r"\b(action|amount)\b", say)


# Stricter than HandBlur on purpose (training data only): no card, rank or hand vocabulary at
# all, so the model learns to never go near a leak, not to lean on the in-app censor.
_card_talk = re.compile(
    r"\b(aces?|kings?|queens?|jacks?|tens|nines|eights|sevens|sixes|fives|fours|threes|twos|deuces?|"
    r"pairs?|flush(es)?|straights?|suited|pocket|trips|quads|full house|boat|hand|hands|cards?|"
    r"spades?|hearts?|diamonds?|clubs?|draw(ing)?|kicker|nuts|monster)\b", re.I)


def card_talk(say: str) -> bool:
    return bool(_card_talk.search(say))


def clean(raw: str) -> str:
    """Tidy a teacher completion into one candidate line."""
    m = re.search(r'"say"\s*:\s*"([^"]*)"', raw)
    if m:  # the teacher wrote the whole JSON object
        return m.group(1).strip()
    lines = [x.strip() for x in raw.strip().split("\n") if x.strip()]
    s = max(lines, key=len) if lines else ""  # skip stage directions like "Taps table twice."
    s = re.sub(r'^(say|"say"|table talk)\s*:\s*', "", s, flags=re.I).strip()
    s = s.strip('"\u201c\u201d\'` ').strip()
    s = s.replace('"', "'").replace("\\", "")
    return s


def words(s: str) -> int:
    return len(s.split())


def leaks(rows: list[dict]) -> list[bool]:
    """Run PokerSI's HandBlur over [{text, hole, face_up, board, made_hand, short_deck, board_plays}]."""
    if not rows:
        return []
    payload = "\n".join(json.dumps({"id": i, **r}) for i, r in enumerate(rows)) + "\n"
    out = subprocess.run(["mise", "exec", "--", "java", "-jar", str(BRIDGE), "blur"], input=payload,
                         capture_output=True, text=True, check=True, cwd=BRIDGE.parent.parent).stdout
    return [json.loads(line)["leaks"] for line in out.splitlines()]
