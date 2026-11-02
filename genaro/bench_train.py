"""Training-throughput benchmark: a few steps of the real Genaro recipe under variants.

    uv run python -m genaro.bench_train [--model Qwen/Qwen3-0.6B] [--steps 40] [variant ...]

Reports real (non-padding) tokens per second, measured after warm-up steps, and peak memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

from genaro import hw

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from genaro.sft import DATA  # noqa: E402

FA2 = "kernels-community/flash-attn2"
FLCE = {"rope": False, "rms_norm": False, "swiglu": False, "cross_entropy": False, "fused_linear_cross_entropy": False}  # TRL fuses the LM head itself
BASE = dict(per_device_train_batch_size=16, gradient_accumulation_steps=4, bf16=True, use_liger_kernel=True,
            gradient_checkpointing=False, train_sampling_strategy="group_by_length", max_length=None,
            learning_rate=3e-5, logging_steps=1, report_to="none", save_strategy="no", eval_strategy="no",
            dataloader_num_workers=4, seed=42)
VARIANTS = {  # name -> (attention, SFTConfig overrides)
    "baseline": ("sdpa", {}),
    "fa2": (FA2, {}),
    "fa2-padfree": (FA2, {"padding_free": True}),
    "fa2-padfree-bs32": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2}),
    "fa2-padfree-bs64": (FA2, {"padding_free": True, "per_device_train_batch_size": 64, "gradient_accumulation_steps": 1}),
    "fa2-padfree-compile": (FA2, {"padding_free": True, "torch_compile": True}),
    "fa2-padfree-bs32-compile": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "torch_compile": True}),
    "sdpa-bs32": ("sdpa", {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 2}),
    # _layers: compile each decoder layer (regional compilation); _fp8: torchao float8 (rowwise) linears
    "fa2-padfree-bs32-layers": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "_layers": True}),
    "fa2-padfree-bs32-fp8": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "_layers": True, "_fp8": True}),
    # Liger's fused linear+cross-entropy only (its other kernels break torch.compile), layers compiled
    "fa2-padfree-bs32-layers-flce": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "_layers": True, "liger_kernel_config": FLCE}),
    "fa2-padfree-bs32-fp8-flce": (FA2, {"padding_free": True, "per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "_layers": True, "_fp8": True, "liger_kernel_config": FLCE}),
    "sdpa-bs32-layers-flce": ("sdpa", {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "_layers": True, "liger_kernel_config": FLCE}),
    "sdpa-bs32-compile": ("sdpa", {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 2, "torch_compile": True}),
}


class Clock(TrainerCallback):
    def __init__(self, warmup: int):
        self.warmup, self.t0, self.tok0 = warmup, None, 0

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.warmup:
            torch.cuda.synchronize()
            self.t0, self.tok0 = time.perf_counter(), state.num_input_tokens_seen


def run(model_name: str, data: str, name: str, steps: int, warmup: int) -> dict:
    attn, over = VARIANTS[name]
    over = dict(over)
    fp8, layers = over.pop("_fp8", False), over.pop("_layers", False)
    ds = load_from_disk(str(DATA / data))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn)
    if fp8:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        convert_to_float8_training(model.model, config=Float8LinearConfig.from_recipe_name("rowwise"),
                                   module_filter_fn=lambda mod, fqn: mod.in_features % 16 == 0 and mod.out_features % 16 == 0)
    if layers:
        for layer in model.model.layers:
            layer.compile()
    args = SFTConfig(output_dir=f"/tmp/genaro-bench/{name}", max_steps=steps, include_num_input_tokens_seen="non_padding",
                     **{**BASE, **over})
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    clock = Clock(warmup)
    trainer.add_callback(clock)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    torch.cuda.synchronize()
    dt = time.perf_counter() - clock.t0
    toks = trainer.state.num_input_tokens_seen - clock.tok0
    out = {"variant": name, "tok_per_s": round(toks / dt), "s_per_step": round(dt / (steps - warmup), 2),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "loss": round(trainer.state.log_history[-2].get("loss", 0), 4)}
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default="sft-qwen3-solver")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    a = ap.parse_args()
    hw.setup()
    for v in a.variants:
        try:
            r = run(a.model, a.data, v, a.steps, a.warmup)
        except Exception as e:  # report and keep going
            r = {"variant": v, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("RESULT", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
