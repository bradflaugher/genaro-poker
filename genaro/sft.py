"""Full fine-tune (or LoRA) of a small model on the pre-tokenized Genaro set.

    uv run python -m genaro.sft configs/genaro-0.6b.yaml [section.key=value ...]
"""

from __future__ import annotations

import json
from pathlib import Path

from genaro import hw

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from genaro.config import load_config  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    cfg = load_config()
    hw.setup()
    print(hw.device_summary())
    m, d, t = cfg["model"], cfg.get("data", {}), cfg["train"]
    ds = load_from_disk(str(DATA / d.get("path", "sft"))).train_test_split(test_size=d.get("n_eval", 256), seed=42)
    tok = AutoTokenizer.from_pretrained(m["name"])
    model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    peft = LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", **cfg["lora"]) if cfg.get("lora") else None
    trainer = SFTTrainer(model=model, args=SFTConfig(**t), train_dataset=ds["train"], eval_dataset=ds["test"],
                         processing_class=tok, peft_config=peft)
    trainer.callback_handler.callbacks.insert(0, hw.ThroughputCallback())
    trainer.train()
    final = Path(t["output_dir"]) / "final"
    model = trainer.model.merge_and_unload() if peft else trainer.model
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    (final / "genaro_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved -> {final}")


if __name__ == "__main__":
    main()
