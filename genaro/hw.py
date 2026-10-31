"""Hardware setup and telemetry for the RTX Spark N1X (Blackwell sm_121, unified LPDDR5X)."""

from __future__ import annotations

import os
import sys
import time

# metrics show up immediately in `tee`/`nohup` logs instead of when the buffer fills
sys.stdout.reconfigure(line_buffering=True)

# Must be set before CUDA initialises. Expandable segments avoid fragmentation when
# sequence lengths vary batch to batch (no packing -> lots of different shapes).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402


def _use_pread_for_safetensors() -> None:
    """N1X driver quirk: cudaMemcpy *from file-backed mmap pages* is pathologically slow
    (~5 MB/s, CPU-bound in the driver), while copying from ordinary heap memory runs at
    full speed. transformers mmaps checkpoints by default, so loading Gemma 4 E4B took
    10 minutes. Forcing safetensors' pread backend (read each tensor into heap, then copy)
    brings it down to ~7 s, without reading whole shards into RAM like disable_mmap does."""
    import transformers.modeling_utils as mu

    if getattr(mu.safe_open, "_genaro_pread", False):
        return
    orig = mu.safe_open

    def safe_open_pread(*args, backend=None, **kwargs):
        return orig(*args, backend="pread", **kwargs)

    safe_open_pread._genaro_pread = True
    mu.safe_open = safe_open_pread


def setup() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _use_pread_for_safetensors()
    # Unified memory: GPU allocations come out of system RAM but are NOT charged to the
    # process's cgroup, so `systemd-run -p MemoryMax=` can't stop a runaway GPU job from
    # starving the desktop. Cap PyTorch's allocator instead: over-allocation becomes a clean
    # CUDA OOM in this process. Default 0.75 of 120 GiB = ~90 GiB.
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GENARO_GPU_MEM_FRACTION", "0.75")))


def device_summary() -> str:
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return (
        f"{p.name} | sm_{p.major}{p.minor} | {p.multi_processor_count} SMs | "
        f"{total / 2**30:.0f} GiB unified ({free / 2**30:.0f} GiB free) | "
        f"torch {torch.__version__} / CUDA {torch.version.cuda}"
    )


def telemetry() -> dict:
    out = {"mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    try:
        out["gpu_watts"] = round(torch.cuda.power_draw() / 1000, 1)
        out["gpu_temp_c"] = torch.cuda.temperature()
    except Exception:
        pass
    return out


class ThroughputCallback(TrainerCallback):
    """Adds tokens/sec, peak memory, power and temperature to every log line."""

    def __init__(self) -> None:
        self._t = None
        self._tokens = 0.0

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()
        self._t = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None or "loss" not in logs:
            return
        now = time.perf_counter()
        tokens = logs.get("num_tokens") or getattr(state, "num_input_tokens_seen", 0)
        if tokens and self._t:
            logs["tok_per_s"] = round((tokens - self._tokens) / (now - self._t))
            self._tokens = tokens
        self._t = now
        logs.update(telemetry())
