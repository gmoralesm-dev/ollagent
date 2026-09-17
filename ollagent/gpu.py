"""GPU awareness for Ollagent.

Ollama performs the inference, so "GPU support" in Ollagent means making the
CLI aware of *where* that inference actually happens:

* VRAM capacity and usage, read from ``nvidia-smi`` when it is available;
* how much of each loaded model really sits in VRAM — Ollama's ``/api/ps``
  reports ``size`` and ``size_vram``, and ``size_vram / size`` is exactly the
  CPU/GPU split that ``ollama ps`` prints in its PROCESSOR column;
* whether an installed model can fit the card at all, and which context size
  keeps it resident (a model that does not fit silently runs 100% on CPU);
* real tokens/second, so estimates can be replaced by a measurement.

Standard library only, like the rest of Ollagent.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .client import Ollama
from .config import Config

# ----------------------------------------------------------------------
# Heuristics
# ----------------------------------------------------------------------

# A model's resident footprint is consistently larger than its download
# size, because Ollama also allocates KV cache and compute buffers. Measured
# on a 3B Q4_K_M model: 1.93 GB on disk -> 2.31 GB resident (1.20x).
_LOAD_OVERHEAD = 1.2

# Roughly how much extra memory each token of context costs, in MB. Derived
# from the same measurement: ~0.38 GB of overhead over 2048 tokens.
_KV_MB_PER_TOKEN = 0.2

# Below this context a model is not worth running at all; Ollama itself
# enforces a 2048 floor on num_ctx for most models.
_MIN_CTX = 2048

# Cap for the suggested context — past this you are better off picking a
# smaller model than squeezing more window into the card.
_MAX_CTX = 32768


@dataclass
class Gpu:
    """One physical GPU as reported by the vendor tool."""

    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int


@dataclass
class LoadedModel:
    """A model currently resident in the Ollama server."""

    name: str
    size: int
    size_vram: int
    context: int = 0
    expires_at: str = ""

    @property
    def offload_pct(self) -> float:
        """Share of the model that lives in VRAM (0.0 = pure CPU)."""
        if not self.size:
            return 0.0
        return 100.0 * self.size_vram / self.size


# ----------------------------------------------------------------------
# Hardware + server probing
# ----------------------------------------------------------------------

def probe_gpus() -> List[Gpu]:
    """Return the installed NVIDIA GPUs, or [] when none can be detected.

    Only ``nvidia-smi`` is used: it ships with the driver and prints one CSV
    line per card, which is far more robust to parse than vendor libraries.
    AMD/ROCm and Apple silicon therefore report as "no GPU detected" — the
    CPU path still works, it just cannot be measured from here.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    cmd = [exe,
           "--query-gpu=index,name,memory.total,memory.used,memory.free",
           "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    gpus: List[Gpu] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(Gpu(
                index=int(parts[0]),
                name=parts[1],
                total_mb=int(float(parts[2])),
                used_mb=int(float(parts[3])),
                free_mb=int(float(parts[4])),
            ))
        except ValueError:
            continue
    return gpus


def usable_vram_mb(gpus: Sequence[Gpu], reserve_mb: int) -> int:
    """VRAM budget for a model, based on card *capacity*.

    Deliberately capacity-based: ``free`` memory fluctuates with whatever
    Ollama itself already has resident (it evicts its own models happily) and
    with driver/CUDA context overhead, which would otherwise make a model that
    is running fine right now look like it does not fit. Whatever else the
    desktop is doing is accounted for by ``reserve_mb``.
    """
    if not gpus:
        return 0
    total = max(g.total_mb for g in gpus)
    return max(total - reserve_mb, 0)


def loaded_models(llm: Ollama) -> List[LoadedModel]:
    """Ask Ollama which models are resident and how much is on the GPU."""
    out: List[LoadedModel] = []
    for raw in llm.running_models():
        try:
            out.append(LoadedModel(
                name=str(raw.get("name") or raw.get("model") or "?"),
                size=int(raw.get("size") or 0),
                size_vram=int(raw.get("size_vram") or 0),
                context=int(raw.get("context_length") or 0),
                expires_at=str(raw.get("expires_at") or ""),
            ))
        except (TypeError, ValueError):
            continue
    return out


def predict_resident_mb(download_bytes: int) -> float:
    """Estimate the VRAM a model needs, from its download size."""
    return (download_bytes / 1e6) * _LOAD_OVERHEAD


# ----------------------------------------------------------------------
# "Will it fit?" analysis
# ----------------------------------------------------------------------

def fit_verdict(resident_mb: float, usable_mb: int) -> Tuple[str, int]:
    """Classify a model against the VRAM budget.

    Returns ``(verdict, suggested_ctx)`` where verdict is one of:

    ``gpu``      fits with room for a large context;
    ``tight``    fits only with a small context — shrink --ctx to stay on GPU;
    ``too-big``  cannot be offloaded at all, it will run 100% on the CPU;
    ``unknown``  no GPU information available (no driver / no VRAM reading).

    Ollama enforces a context floor (~2048 tokens), so a model that only
    leaves room for less than that behaves like ``too-big`` whatever
    ``--ctx`` says.
    """
    if usable_mb <= 0:
        return "unknown", 0
    if resident_mb >= usable_mb:
        return "too-big", 0

    spare_mb = usable_mb - resident_mb
    ctx = int(spare_mb / _KV_MB_PER_TOKEN)
    ctx = (ctx // 512) * 512          # round down to a tidy step
    if ctx < _MIN_CTX:
        return "too-big", 0
    if ctx < 4096:
        return "tight", ctx
    return "gpu", min(ctx, _MAX_CTX)


def suggest_ctx(cfg: Config, llm: Ollama,
                model: Optional[str] = None) -> Optional[int]:
    """Context size that should let ``model`` stay on the GPU, if any.

    Returns None when there is nothing to suggest (no GPU, unknown model, or
    a model that simply does not fit).
    """
    usable = usable_vram_mb(probe_gpus(), cfg.reserve_vram_mb)
    if usable <= 0:
        return None
    size = _download_size(llm, model or cfg.model)
    if not size:
        return None
    verdict, ctx = fit_verdict(predict_resident_mb(size), usable)
    if verdict in ("gpu", "tight") and ctx:
        return ctx
    return None


def _download_size(llm: Ollama, model: str) -> int:
    """Download size of ``model`` in bytes, or 0 when it is not installed.

    Cloud models (``:cloud`` tags) have no local size — they run on Ollama's
    servers, so their VRAM is somebody else's problem.
    """
    for entry in llm.list_models():
        name = str(entry.get("name") or "")
        if name != model and not name.startswith(model):
            continue
        try:
            return int(entry.get("size") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _fit_rows(cfg: Config, llm: Ollama,
              usable_mb: int) -> List[Tuple[str, str, str]]:
    """(model, download size, verdict text) for every installed model."""
    rows: List[Tuple[str, str, str]] = []
    for entry in llm.list_models():
        name = str(entry.get("name") or "?")
        try:
            size = int(entry.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size == 0 or "cloud" in name.lower():
            rows.append((name, "cloud", "remote GPU (no local VRAM needed)"))
            continue
        verdict, ctx = fit_verdict(predict_resident_mb(size), usable_mb)
        if verdict == "gpu":
            note = f"fits — up to ~{ctx} ctx on GPU"
        elif verdict == "tight":
            note = f"tight — use --ctx {ctx} to stay on GPU"
        elif verdict == "too-big":
            note = "too big — will run on the CPU"
        else:
            note = "unknown (no VRAM reading)"
        rows.append((name, f"{size / 1e9:.2f} GB", note))
    rows.sort(key=lambda r: (r[2].startswith("too big"), r[0]))
    return rows


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def report(cfg: Config, llm: Ollama) -> str:
    """Full GPU / offload / model-fit report (the ``--gpu`` output)."""
    gpus = probe_gpus()
    usable = usable_vram_mb(gpus, cfg.reserve_vram_mb)
    lines: List[str] = ["", "GPU", "-" * 58]
    if not gpus:
        lines.append("  none detected (no NVIDIA card, or the driver is missing)")
        lines.append("  Ollama will run inference on the CPU.")
        lines.append("  Install the vendor driver, then check `nvidia-smi`.")
    else:
        for g in gpus:
            lines.append(f"  [{g.index}] {g.name}")
            lines.append(f"      VRAM  : {g.total_mb / 1024:.1f} GB total, "
                         f"{g.used_mb / 1024:.1f} GB used, "
                         f"{g.free_mb / 1024:.1f} GB free")
        lines.append(f"      budget: {usable / 1024:.1f} GB usable for a model "
                     f"(capacity minus {cfg.reserve_vram_mb} MB reserve)")

    lines += ["", "LOADED NOW (offload)", "-" * 58]
    loaded = loaded_models(llm)
    if not loaded:
        lines.append("  (no model resident — one loads on the next request)")
    for m in loaded:
        state = "100% CPU" if m.size_vram == 0 else f"{m.offload_pct:.0f}% GPU"
        where = (f"{m.size_vram / 1e9:.2f} of {m.size / 1e9:.2f} GB in VRAM"
                 if m.size_vram else "nothing in VRAM")
        ctx = f", ctx {m.context}" if m.context else ""
        lines.append(f"  {m.name}")
        lines.append(f"      {state} — {where}{ctx}")

    lines += ["", "INSTALLED MODELS vs VRAM BUDGET", "-" * 58]
    rows = _fit_rows(cfg, llm, usable)
    if not rows:
        lines.append("  (no models installed — try `ollama pull qwen2.5-coder:3b`)")
    width = max((len(r[0]) for r in rows), default=10)
    for name, size, note in rows:
        lines.append(f"  {name:<{width}}  {size:>9}  {note}")

    lines += ["", "NEXT STEPS", "-" * 58]
    if not gpus:
        lines.append("  • driver missing: nothing here can be GPU accelerated yet")
    else:
        lines.append("  • --auto-fit    shrinks num_ctx so the model fits VRAM")
        lines.append("  • --benchmark   measures real tokens/second on the card")
        lines.append("  • --keepalive 30m  keeps an offloaded model hot")
    lines.append("")
    return "\n".join(lines)


def startup_warning(cfg: Config, llm: Ollama,
                    model: Optional[str] = None) -> Optional[str]:
    """One-line warning when the request is about to run on the CPU.

    Two cases are covered:

    1. a model is already resident with nothing in VRAM — hard evidence that
       the server gave up on offloading it;
    2. the requested model is too large for the card, so the CPU is certain —
       caught *before* the slow load, which is the useful moment to speak up.
    """
    target = model or cfg.model
    gpus = probe_gpus()
    if not gpus:
        return ("no NVIDIA GPU detected — inference will run on the CPU. "
                "See `--gpu` once the driver is installed.")

    for m in loaded_models(llm):
        matches = m.name == target or m.name.startswith(target)
        if matches and m.size and m.size_vram == 0:
            return (f"'{m.name}' is resident with 0% offload (100% CPU). "
                    f"Try a smaller --ctx or a smaller model — see `--gpu`.")

    size = _download_size(llm, target)
    if size:
        usable = usable_vram_mb(gpus, cfg.reserve_vram_mb)
        verdict, ctx = fit_verdict(predict_resident_mb(size), usable)
        if verdict == "too-big":
            return (f"'{target}' needs ~{predict_resident_mb(size) / 1024:.1f} GB "
                    f"but only {usable / 1024:.1f} GB of VRAM is usable here — "
                    f"it will run on the CPU. See `--gpu` for models that fit.")
        if verdict == "tight":
            return (f"'{target}' only fits with a small context — "
                    f"run with `--ctx {ctx}` to stay on the GPU.")
        # A context that is too big for the card makes Ollama's runner abort
        # outright ("llama runner process has terminated"), so flag it before
        # the request instead of after the crash.
        if verdict in ("gpu", "tight") and ctx and cfg.num_ctx > ctx:
            return (f"--ctx {cfg.num_ctx} exceeds what fits in VRAM for "
                    f"'{target}' (~{ctx} tokens on this card) — the runner may "
                    f"abort. Try --auto-fit or `--ctx {ctx}`.")
    return None


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------

# A prompt that is just long enough to exercise prompt processing (the
# prefill phase) without pulling in any specific knowledge.
_BENCH_PROMPT = (
    "Explain in about three sentences what a GPU does differently from a CPU."
)


def benchmark(cfg: Config, llm: Ollama, model: Optional[str] = None,
              prompt: Optional[str] = None, num_predict: int = 96,
              num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Measure real generation speed for ``model`` on this machine.

    Runs one /api/generate request and then reads /api/ps to find out how much
    of the model actually ended up in VRAM, so a slow result can be explained
    (CPU fallback) rather than guessed at.

    ``num_ctx`` defaults to whatever fits in VRAM (capped at 4096): a context
    that is too large for the card makes Ollama's runner abort outright, and a
    benchmark that crashes measures nothing.
    """
    target = model or cfg.model
    if num_ctx is None:
        fitted = suggest_ctx(cfg, llm, target)
        num_ctx = min(fitted, 4096) if fitted else min(cfg.num_ctx, 4096)
    data = llm.generate(
        prompt or _BENCH_PROMPT,
        model=target,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=cfg.temperature,
        keep_alive=cfg.keep_alive,
    )

    def _rate(count: Any, duration: Any) -> float:
        try:
            count_i, dur_i = int(count or 0), int(duration or 0)
        except (TypeError, ValueError):
            return 0.0
        return (count_i / (dur_i / 1e9)) if dur_i else 0.0

    offload = 0.0
    size_vram = size = 0
    context = 0
    for m in loaded_models(llm):
        if m.name == target or m.name.startswith(target):
            offload, size_vram, size, context = (
                m.offload_pct, m.size_vram, m.size, m.context)
            break

    gpus = probe_gpus()
    return {
        "model": target,
        "gpu": gpus[0].name if gpus else "",
        "context": context or num_ctx,
        "load_s": float(int(data.get("load_duration") or 0)) / 1e9,
        "prompt_tokens": int(data.get("prompt_eval_count") or 0),
        "prompt_rate": _rate(data.get("prompt_eval_count"),
                             data.get("prompt_eval_duration")),
        "gen_tokens": int(data.get("eval_count") or 0),
        "gen_rate": _rate(data.get("eval_count"), data.get("eval_duration")),
        "offload_pct": offload,
        "size_vram_gb": size_vram / 1e9,
        "size_gb": size / 1e9,
        "done_reason": str(data.get("done_reason") or ""),
    }


def format_benchmark(res: Mapping[str, Any]) -> str:
    """Human-readable rendering of :func:`benchmark`."""
    lines = ["", f"Benchmark — {res['model']}", "-" * 58]
    if res.get("gpu"):
        lines.append(f"  gpu        : {res['gpu']}")
    lines.append(f"  context    : {int(res.get('context') or 0)} tokens requested")
    lines.append(f"  load       : {res.get('load_s', 0.0):.1f} s")
    lines.append(f"  prompt     : {res.get('prompt_tokens', 0)} tok "
                 f"= {res.get('prompt_rate', 0.0):.1f} tok/s")
    lines.append(f"  generation : {res.get('gen_tokens', 0)} tok "
                 f"= {res.get('gen_rate', 0.0):.2f} tok/s")
    offload = float(res.get("offload_pct") or 0.0)
    if offload <= 0.0:
        lines.append("  offload    : 0% — this ran entirely on the CPU")
        lines.append("               (too big for VRAM: see `--gpu`, or try "
                     "`--auto-fit`)")
    else:
        lines.append(f"  offload    : {offload:.0f}% GPU "
                     f"({res.get('size_vram_gb', 0.0):.2f} of "
                     f"{res.get('size_gb', 0.0):.2f} GB in VRAM)")
    if res.get("done_reason") == "length":
        lines.append("  note       : generation hit num_predict and was cut "
                     "short — the rate above is still valid")
    lines.append("")
    return "\n".join(lines)
