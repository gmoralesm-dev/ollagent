"""Tool implementations + registry for the agent.

These are the "hands" of the agent. Ollama (the brain) proposes JSON tool
calls; these functions do the actual filesystem + shell work and return
text observations back to the model.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List

from .config import Config

# JSON schema for each tool (also used to build the model prompt).
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a text file (optionally a line range). Returns file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-based first line."},
                "end_line": {"type": "integer", "description": "1-based last line."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing one.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string within a file. Prefer over rewriting whole files. replace_all replaces every occurrence.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory. Recursive by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "depth": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "search_files",
        "description": "Regex search over files. Uses ripgrep if available, else a built-in walker.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "file_pattern": {"type": "string", "description": "Optional glob, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and capture stdout/stderr. For builds, tests, git, scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (default 60)."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "done",
        "description": "Signal the task is finished, with a short summary.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]
class ToolResult:
    """Result of executing a tool: a text observation plus a stop/conversation flag."""
    __slots__ = ("text", "should_stop", "conversational")

    def __init__(self, text: str, should_stop: bool = False,
                 conversational: bool = False):
        self.text = text
        self.should_stop = should_stop
        # True when this is a plain conversational answer (no tool ran).
        self.conversational = conversational


class ToolError(Exception):
    """Raised when a tool call is malformed or cannot execute."""


# Small models often emit typographic ("smart") quotes instead of ASCII ones,
# which breaks code. Normalize them in written/edited file content.
_SMART_QUOTES = {
    "\u2018": "'",   # ‘
    "\u2019": "'",   # ’
    "\u201c": '"',   # “
    "\u201d": '"',   # ”
    "\u0060": "'",   # ` (backtick used as quote)
    "\u00b4": "'",   # ´
}


def _sanitize_code_text(text: str, path: str) -> str:
    """Replace smart quotes with ASCII ones for code-ish files."""
    code_exts = (
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".json", ".yml",
        ".yaml", ".toml", ".html", ".css", ".sql", ".md",
    )
    lower = path.lower()
    if lower.endswith(code_exts) and any(q in text for q in _SMART_QUOTES):
        for smart, ascii_q in _SMART_QUOTES.items():
            text = text.replace(smart, ascii_q)
    return text


class ToolContext:
    """Shared state passed to every tool: config, workspace root, cwd."""

    def __init__(self, cfg: Config, workspace: str, cwd: str):
        self.cfg = cfg
        self.workspace = os.path.realpath(workspace)
        self.cwd = cwd
        self.auto_approve = cfg.auto_approve
        self.sandbox = cfg.sandbox
        self.command_timeout = cfg.command_timeout

    def resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the cwd (workspace)."""
        p = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isabs(p):
            p = os.path.join(self.cwd, p)
        return os.path.realpath(p)

    def check_sandbox(self, path: str) -> str:
        """If sandboxing, forbid paths that escape the workspace root."""
        real = self.resolve(path)
        if self.sandbox:
            try:
                os.path.commonpath([real, self.workspace])
            except ValueError:
                raise ToolError(
                    f"Path outside workspace sandbox: {real}\n"
                    f"Workspace is {self.workspace}. "
                    f"Re-run with --no-sandbox to allow it."
                )
        return real


def _approve(ctx: ToolContext, action: str) -> None:
    """Prompt for human approval unless in auto-approve mode."""
    if ctx.auto_approve:
        return
    print(f"\n\033[33m[approval] {action}\033[0m")
    try:
        answer = input("  Proceed? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise ToolError("Action declined by user.")
# ----------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------

def _read_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = ctx.check_sandbox(args.get("path", ""))
    if not os.path.isfile(path):
        raise ToolError(f"Not a file or does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise ToolError(f"Could not read {path}: {exc}")

    start = int(args.get("start_line") or 1)
    end = int(args.get("end_line") or len(lines))
    start = max(1, min(start, len(lines)))
    end = max(start, min(end, len(lines)))
    selected = lines[start - 1:end]

    MAX_LINES = 400
    truncated = len(selected) > MAX_LINES
    if truncated:
        selected = selected[:MAX_LINES]

    body = "".join(selected).rstrip("\n")
    out = f"--- {path} (lines {start}-{end}) ---\n{body}"
    if truncated:
        out += f"\n... [truncated; file has {len(lines)} lines]"
    out += f"\n--- end of {path} ---"
    return ToolResult(out)


def _write_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = ctx.check_sandbox(args.get("path", ""))
    content = str(args.get("content", ""))
    content = _sanitize_code_text(content, path)
    _approve(ctx, f"WRITE FILE {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return ToolResult(f"Wrote {len(content)} bytes to {path}")


def _edit_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = ctx.check_sandbox(args.get("path", ""))
    old = _sanitize_code_text(str(args.get("old_string", "")), path)
    new = _sanitize_code_text(str(args.get("new_string", "")), path)
    if not old:
        raise ToolError("edit_file requires non-empty old_string.")
    if not os.path.isfile(path):
        raise ToolError(f"Not a file or does not exist: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    replace_all = bool(args.get("replace_all", False))
    if replace_all:
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string not found in {path}.")
        text = text.replace(old, new)
    else:
        count = 1
        idx = text.find(old)
        if idx == -1:
            raise ToolError(f"old_string not found in {path}. Match indentation exactly.")
        text = text[:idx] + new + text[idx + len(old):]

    _approve(ctx, f"EDIT FILE {path} ({count} replacement(s))")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return ToolResult(f"Applied {count} replacement(s) in {path}")
def _list_files(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = ctx.check_sandbox(args.get("path") or ".")
    recursive = bool(args.get("recursive", True))
    depth = int(args.get("depth") or (10 if recursive else 1))
    if not os.path.isdir(path):
        raise ToolError(f"Not a directory: {path}")

    skip = {".git", "__pycache__", ".venv", "venv", "node_modules", ".next"}
    rows: List[str] = []
    base_level = path.rstrip("/").count("/")

    def walk(dirpath: str, level: int) -> None:
        try:
            entries = sorted(os.listdir(dirpath))
        except OSError as exc:
            rows.append(f"{'  ' * level}[error: {exc}]")
            return
        for name in entries:
            full = os.path.join(dirpath, name)
            if name in skip:
                continue
            indent = "  " * (dirpath.count("/") - base_level)
            if os.path.isdir(full) and not os.path.islink(full):
                rows.append(f"{indent}{name}/")
                if recursive and level < depth:
                    walk(full, level + 1)
            else:
                rows.append(f"{indent}{name}")

    walk(path, 0)
    return ToolResult("\n".join([f"--- {path} ---"] + rows))


def _search_files(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    pattern = str(args.get("pattern", ""))
    if not pattern:
        raise ToolError("search_files requires a pattern.")
    path = ctx.check_sandbox(args.get("path") or ".")
    file_pat = args.get("file_pattern")
    if not os.path.isdir(path):
        raise ToolError(f"Not a directory: {path}")

    regex = re.compile(pattern)
    rg = shutil.which("rg")

    if rg:
        cmd = [rg, "--line-number", "--no-heading", "-S", pattern, path]
        if file_pat:
            cmd[1:1] = ["--glob", file_pat]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              cwd=ctx.workspace)
        out = proc.stdout or proc.stderr
        if not out.strip():
            return ToolResult("No matches found.")
        return ToolResult("Matches:\n" + out[:6000])

    import fnmatch
    skip = {".git", "__pycache__", ".venv", "venv", "node_modules", ".next"}
    matches: List[str] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if file_pat and not fnmatch.fnmatch(fn, file_pat):
                continue
            full = os.path.join(root, fn)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if regex.search(line):
                            matches.append(f"{full}:{i}:{line.rstrip()}")
                            if len(matches) >= 100:
                                break
            except OSError:
                continue
        if len(matches) >= 100:
            break
    if not matches:
        return ToolResult("No matches found.")
    return ToolResult("Matches:\n" + "\n".join(matches))


def _run_command(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    command = str(args.get("command", ""))
    if not command.strip():
        raise ToolError("run_command requires a command.")
    timeout = int(args.get("timeout") or ctx.command_timeout)

    _approve(ctx, f"RUN COMMAND: {command}")

    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=timeout, cwd=ctx.cwd)
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            f"Command timed out after {timeout}s. Partial output:\n"
            f"{exc.stdout or ''}\n{exc.stderr or ''}"
        )
    except OSError as exc:
        raise ToolError(f"Could not run command: {exc}")

    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout)
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr)
    out = "\n".join(parts).strip() or "(no output)"
    out = f"[exit code {proc.returncode}]\n{out}"
    return ToolResult(out[:8000])


def _done(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    summary = str(args.get("summary", ""))
    return ToolResult(f"TASK COMPLETE. Summary: {summary}", should_stop=True)


# ----------------------------------------------------------------------
# Registry / dispatch
# ----------------------------------------------------------------------

TOOLS: Dict[str, Callable[[ToolContext, Dict[str, Any]], ToolResult]] = {
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "list_files": _list_files,
    "search_files": _search_files,
    "run_command": _run_command,
    "done": _done,
}


def dispatch(ctx: ToolContext, name: str, args: Dict[str, Any]) -> ToolResult:
    if name not in TOOLS:
        raise ToolError(
            f"Unknown tool '{name}'. Available tools: {', '.join(sorted(TOOLS))}"
        )
    return TOOLS[name](ctx, args)