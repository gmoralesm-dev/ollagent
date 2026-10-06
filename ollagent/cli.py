"""Command-line entry point for Ollagent.

Two modes:
  ollagent "task text"                    # one-shot task
  ollagent                                # interactive session

Stop & resume (like Agy / Cline):
  Ctrl+C            stops the current task (analogous to their Stop button)
  'exit' / 'quit'   exits and auto-saves the conversation so you can resume
  run ollagent again in the same folder to pick the session back up
  'new'             starts a fresh conversation

Use --yolo to fully auto-approve writes/commands (throwaway folders only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from . import gpu
from .agent import Agent
from .client import Ollama, OllamaError
from .config import Config, from_env

_DEFAULT_SESSION = ".ollagent_session.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollagent",
        description="A local terminal agent for your Ollama models.",
    )
    p.add_argument("task", nargs="*",
                   help="Task to run. If omitted, enters interactive session.")
    p.add_argument("--model", default=None,
                   help=f"Ollama model name (default: {Config().model}).")
    p.add_argument("--url", default=None,
                   help=f"Ollama server URL (default: {Config().ollama_url}).")
    p.add_argument("--ctx", type=int, default=None,
                   help="Context window in tokens.")
    p.add_argument("--temp", type=float, default=None,
                   help="Sampling temperature (default: 0.1).")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Cap on tool-call loop iterations.")
    p.add_argument("--think", dest="think", action="store_true", default=False,
                   help="Enable the model's thinking/reasoning phase "
                        "(default: off — small models can exhaust the "
                        "context window thinking).")
    p.add_argument("--yolo", action="store_true",
                   help="Auto-approve all file writes and shell commands.")
    p.add_argument("--no-sandbox", action="store_true",
                   help="Allow file access outside the workspace.")
    p.add_argument("--cwd", default=None,
                   help="Workspace directory to operate in (default: current).")
    p.add_argument("--session", default=None,
                   help="Session file to load/save "
                        "(default: <workspace>/.ollagent_session.json).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any saved session; start fresh.")
    p.add_argument("--chat", action="store_true",
                   help="Force conversational mode (no tools) even for tasks. "
                        "Questions are auto-detected as chat without this flag.")
    p.add_argument("--list-models", action="store_true",
                   help="List models on the Ollama server and exit.")
    p.add_argument("--gpu", action="store_true",
                   help="Report GPU/VRAM status, how much of each loaded model "
                        "is offloaded, and which installed models fit in VRAM, "
                        "then exit.")
    p.add_argument("--benchmark", nargs="?", const="", default=None,
                   metavar="MODEL",
                   help="Measure real tokens/second and the GPU offload ratio "
                        "for MODEL (default: the configured model), then exit.")
    p.add_argument("--auto-fit", action="store_true",
                   help="Shrink --ctx automatically so the model fits in VRAM "
                        "and stays on the GPU.")
    p.add_argument("--no-gpu-check", action="store_true",
                   help="Skip the startup warning about CPU-bound inference.")
    p.add_argument("--keepalive", default=None, metavar="DURATION",
                   help="Keep the model loaded between runs (e.g. 30m, or -1 "
                        "for forever). Default: the server's own setting.")
    return p


# Common coding/tool-requiring verbs. If any appear, treat as a task even if
# it happens to end in '?'. Keeps chat auto-detection conservative.
_ACTION_HINTS = (
    "create", "write", "make", "build", "fix", "repair", "edit", "read",
    "search", "find", "run", "delete", "remove", "add", "install", "test",
    "refactor", "update", "change", "generate", "list", "show", "check",
)

_QUESTION_HINTS = (
    "what", "why", "who", "how", "when", "where", "which", "explain",
    "define", "describe", "tell me", "meaning of", "what is", "what does",
    "difference between", "is it", "how do i", "can you explain",
)


def _looks_like_question(text: str) -> bool:
    """Heuristic: is this message a casual question rather than a task?"""
    t = text.strip().lower()
    if not t:
        return False
    # Explicit action words → treat as a coding task (even if phrased as a Q).
    if any(h in t for h in _ACTION_HINTS):
        return False
    # Starts with a question word, or ends with a '?'.
    if t.endswith("?"):
        return True
    first = t.split()[0] if t.split() else ""
    return any(first.startswith(q) or t.startswith(q) for q in _QUESTION_HINTS)


def _apply_args(cfg: Config, args: argparse.Namespace) -> None:
    if args.model:
        cfg.model = args.model
    if args.url:
        cfg.ollama_url = args.url
    if args.ctx:
        cfg.num_ctx = args.ctx
    if args.temp is not None:
        cfg.temperature = args.temp
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations
    if args.yolo:
        cfg.auto_approve = True
    if args.no_sandbox:
        cfg.sandbox = False
    if args.cwd:
        raw = args.cwd.strip()
        cfg.workspace = os.path.abspath(raw)
        looks_like_word = (
            os.sep not in raw
            and not raw.startswith(".")
            and not os.path.isdir(raw)
        )
        if looks_like_word and args.task:
            # `--cwd what is a script?` → argparse ate "what" as the folder,
            # leaving "is a script?" as the task. Flag it so the user isn't
            # confused about the split / the folder it just created.
            print(
                f"\033[33m[tip] '--cwd {raw}' does not look like an existing folder "
                f"and is being treated as the workspace.\n"
                f"      If you meant '{raw}' as the start of your task instead, "
                f"drop --cwd or quote the whole task.\033[0m",
                file=sys.stderr,
            )
        os.makedirs(cfg.workspace, exist_ok=True)
    if args.chat:
        cfg.chat = True
    if args.think:
        cfg.think = args.think
    if args.auto_fit:
        cfg.auto_fit = True
    if args.no_gpu_check:
        cfg.gpu_check = False
    if args.keepalive:
        cfg.keep_alive = args.keepalive
# -- session persistence ------------------------------------------------

def _load_session(path: str) -> list:
    """Return saved messages from a session file, or [] if none/invalid."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return []
        return [m for m in msgs if isinstance(m, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def _save_session(path: str, cfg: Config, messages) -> None:
    try:
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    except OSError:
        pass
    payload = {
        "version": __version__,
        "model": cfg.model,
        "url": cfg.ollama_url,
        "workspace": cfg.workspace,
        "messages": messages,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"\033[31m[could not save session: {exc}]\033[0m")


def _banner(cfg: Config, session_file: str, resumed: bool) -> None:
    print("=" * 58)
    print(" Ollagent — local terminal agent for Ollama")
    print("=" * 58)
    print(f" model     : {cfg.model}")
    print(f" server    : {cfg.ollama_url}")
    print(f" workspace : {cfg.workspace}")
    print(f" session   : {session_file} ({'resumed' if resumed else 'new'})")
    print(f" context   : {cfg.num_ctx} tokens | temp {cfg.temperature} "
          f"| think {'on' if cfg.think else 'off'}")
    gpus = gpu.probe_gpus()
    if gpus:
        g = gpus[0]
        print(f" gpu       : {g.name} ({g.free_mb / 1024:.1f} GB free "
              f"of {g.total_mb / 1024:.1f} GB)")
    else:
        print(" gpu       : none detected (inference runs on the CPU)")
    print(f" approve   : {'auto (--yolo)' if cfg.auto_approve else 'prompted'}")
    print(" Ctrl+C stops the current task; 'exit' quits & saves for later.")
    print(" Type a task, then Enter. 'new' clears the conversation.")
    print()


def _run_interactive(agent: Agent, cfg: Config, session_file: str,
                     resumed: bool) -> None:
    _banner(cfg, session_file, resumed)
    while True:
        try:
            task = input("\n\033[1mollagent> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye — session saved for later.")
            _save_session(session_file, cfg, agent.history())
            break
        if not task:
            continue
        low = task.lower()
        if low in ("exit", "quit", "q", "stop"):
            print("Bye — session saved. Run ollagent here again to resume.")
            _save_session(session_file, cfg, agent.history())
            break
        if low == "new":
            agent.new()
            print("Fresh conversation started.")
            continue

        # Auto-route: casual questions → chat (no tools), else agent mode.
        agent.chat_mode = cfg.chat or _looks_like_question(task)

        try:
            final = agent.run(task)
        except OllamaError as exc:
            print(f"\033[31m{exc}\033[0m")
            continue

        if agent.last_stopped:
            print("\n[task stopped] say a new instruction to resume, "
                  "or 'exit' to quit.\n")
            agent.last_stopped = False
        elif final:
            print(f"\n{final}")

        _save_session(session_file, cfg, agent.history())


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = from_env()
    _apply_args(cfg, args)

    llm = Ollama(cfg)

    if args.list_models:
        models = llm.list_models()
        if models:
            print("Available models:")
            for m in models:
                print(f"  {m}")
        else:
            print("Could not list models (is Ollama running?).")
        return 0

    if not llm.ping():
        print(
            f"\033[31mCould not reach Ollama at {cfg.ollama_url}.\033[0m\n"
            "  • Start it with `ollama start` (or `ollama serve`).\n"
            "  • If hosted elsewhere, pass --url and --model.\n"
            "  • Verify the model name with `ollama list`."
        )
        return 1

    if args.gpu:
        print(gpu.report(cfg, llm))
        return 0

    if args.benchmark is not None:
        bench_model = args.benchmark or cfg.model
        ctx_note = f"num_ctx={args.ctx}" if args.ctx else "auto num_ctx"
        print(f"\033[90m[benchmark] {bench_model} - generating up to 96 "
              f"tokens ({ctx_note})...\033[0m")
        try:
            result = gpu.benchmark(cfg, llm, model=bench_model,
                                   num_ctx=args.ctx)
        except OllamaError as exc:
            print(f"\033[31m{exc}\033[0m")
            return 1
        print(gpu.format_benchmark(result))
        return 0

    # GPU-aware defaults: shrink the context until the model fits in VRAM,
    # and say so up front when the request is going to be CPU-bound anyway.
    fitted = gpu.suggest_ctx(cfg, llm)
    if fitted is None:
        if cfg.auto_fit:
            print("\033[33m[auto-fit] no VRAM budget for this model - "
                  "keeping --ctx as it is.\033[0m")
    elif fitted != cfg.num_ctx:
        print(f"\033[90m[auto-fit] num_ctx {cfg.num_ctx} -> {fitted} "
              f"(so the model fits in VRAM)\033[0m")
        cfg.num_ctx = fitted

    if cfg.gpu_check:
        warning = gpu.startup_warning(cfg, llm)
        if warning:
            print(f"\033[33m[gpu] {warning}\033[0m")

    interactive = not bool(args.task)
    session_file = None
    if args.session:
        session_file = args.session
    elif interactive:
        session_file = os.path.join(cfg.workspace, _DEFAULT_SESSION)

    loaded_messages = []
    if session_file and not args.fresh:
        loaded_messages = _load_session(session_file)
        if loaded_messages:
            print(f"\033[90m[resumed session: {len(loaded_messages)} messages "
                  f"from {session_file}]\033[0m")

    agent = Agent(cfg, history=loaded_messages)

    if args.task:
        task = " ".join(args.task)
        agent.chat_mode = cfg.chat or _looks_like_question(task)
        print(f"\033[90m[task] {task}\033[0m")
        try:
            final = agent.run(task)
        except OllamaError as exc:
            print(f"\033[31m{exc}\033[0m")
            return 1
        if final and not agent.last_stopped:
            print(f"\n\033[92m==\033[0m{final}")
        elif agent.last_stopped:
            print("\n[task stopped mid-way]")
        if session_file:
            _save_session(session_file, cfg, agent.history())
            print(f"\033[90m[session saved to {session_file}]\033[0m")
        return 0

    _run_interactive(agent, cfg, session_file, bool(loaded_messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())