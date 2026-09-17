"""Central configuration for Ollagent.

All connection / model settings live here so they can be overridden from the
CLI or environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# Ollama permission / tuning env vars you may want to set on the server:
#   OLLAMA_ORIGINS="*"           allow browser/Electron clients (CORS)
#   OLLAMA_HOST=0.0.0.0:11434   accept remote connections
#   OLLAMA_KEEP_ALIVE=-1        keep model loaded in VRAM between calls
_DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Reasonable default for the huihui_ai/qwen3.5-abliterated:9b class of models.
_DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"


@dataclass
class Config:
    ollama_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    )
    model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
    )
    temperature: float = 0.1
    # Context window in tokens. Small local models need realistic numbers.
    num_ctx: int = 16000
    # Hard cap on how long a single shell command may run (seconds).
    command_timeout: int = 120
    # How long to keep the loop alive (tool-call iterations) before stopping.
    max_iterations: int = 25
    # YOLO mode: auto-approve all file writes / shell commands.
    auto_approve: bool = False
    # Disable the model's reasoning/"thinking" phase (qwen3.5-class models).
    # Off by default: small local models burn their context window thinking
    # and can return an empty answer when it runs out mid-thought.
    think: bool = False
    # Restrict file access to the workspace directory tree.
    sandbox: bool = True
    # Force purely conversational mode (no tools) even for tasks.
    chat: bool = False
    workspace: str = field(default_factory=os.getcwd)

    # -- GPU awareness ---------------------------------------------------
    # Warn before a request when the model cannot fit in VRAM and will
    # therefore run on the CPU (3-4x slower for small models). Disable with
    # --no-gpu-check, or OLLAGENT_GPU_CHECK=0.
    gpu_check: bool = True
    # Automatically shrink num_ctx until the model should fit in VRAM.
    auto_fit: bool = False
    # Keep the model resident between runs (e.g. "30m"); "" = server default.
    # Useful on machines where a GPU-offloaded model is expensive to reload.
    keep_alive: str = ""
    # VRAM left untouched for the desktop/compositor, in MB, so offloading a
    # model does not starve the display server.
    reserve_vram_mb: int = 400


def from_env() -> Config:
    """Build a Config from environment variables."""
    cfg = Config()
    if os.environ.get("OLLAGENT_MODEL"):
        cfg.model = os.environ["OLLAGENT_MODEL"]
    if os.environ.get("OLLAGENT_URL"):
        cfg.ollama_url = os.environ["OLLAGENT_URL"]
    if os.environ.get("OLLAGENT_KEEP_ALIVE"):
        cfg.keep_alive = os.environ["OLLAGENT_KEEP_ALIVE"]
    if os.environ.get("OLLAGENT_GPU_CHECK", "").lower() in ("0", "false", "no", "off"):
        cfg.gpu_check = False
    return cfg