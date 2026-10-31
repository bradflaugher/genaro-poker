"""Tiny YAML config loader with dotted CLI overrides.

    python -m wavebend_edge.sft configs/sft-gemma4-e4b.yaml train.max_steps=20 lora.r=8
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def load_config(argv: list[str] | None = None) -> dict:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit("usage: python -m <module> CONFIG.yaml [section.key=value ...]")
    cfg = yaml.safe_load(Path(argv[0]).read_text())
    for item in argv[1:]:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
