"""The Ollagent agent loop.

Owns the conversation with the model, builds the system prompt describing
the available tools, parses the model's JSON replies, dispatches tool calls,
and feeds observations back until the model calls `done`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional

from .client import Ollama, OllamaError
from .config import Config
from .tools import TOOL_SCHEMAS, ToolContext, ToolError, ToolResult, dispatch

_TAG_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


def _schema_params(schema: Mapping[str, Any]) -> str:
    props = schema.get("parameters", {}).get("properties", {})
    required = schema.get("parameters", {}).get("required", [])
    parts = []
    for name in props:
        marker = "*" if name in required else ""
        parts.append(f"{name}{marker}")
    return ", ".join(parts)


def build_system_prompt(workspace: str) -> str:
    """Describe the environment, tool protocol, and JSON call format."""
    tools_desc = "\n".join(
        f"- {t['name']}({_schema_params(t)}): {t['description']}"
        for t in TOOL_SCHEMAS
    )
    return f"""You are Ollagent, an autonomous coding agent that works inside a terminal.
You help complete coding tasks by interacting with the local filesystem and shell.

WORKSPACE
The user's project root is: {workspace}
All work should happen inside this directory unless asked otherwise.

AVAILABLE TOOLS
{tools_desc}

PROTOCOL
Respond with EXACTLY ONE valid JSON object. No markdown, no prose outside JSON.
There are exactly two allowed response shapes:

1) A plain conversational answer, when the user is just chatting or asking a
   question that does NOT need reading or changing the workspace:
   {{"type": "answer", "text": "<your natural-language reply>"}}

2) A tool call when the task actually requires the filesystem or shell:
   {{"tool": "<tool_name>", "arguments": {{ ... }}}}
   ...keep calling tools until the job is really done, then finish with:
   {{"done": true, "answer": "<plain-text summary for the user>"}}

RULES
- If the user is chatting or asking something general and a tool is NOT needed,
  reply directly with {{"type": "answer", "text": "..."}}. Do NOT call a tool
  just to be helpful — call one only when you truly need the environment.
- To inspect code first use read_file / list_files / search_files, then edit or
  run commands as needed.
- Prefer edit_file with an exact old_string over write_file for small changes.
- If a tool reports an error, read the error, adapt, and retry.
- Run commands (builds/tests) to verify your work and feed output back.
- Never claim completion without verifying. Keep it concise.
- Output the JSON only."""


def build_chat_prompt() -> str:
    """A pure conversational system prompt — no tools, no JSON.

    Used when the user is just chatting / asking questions, so the model
    answers naturally instead of reaching for tools it doesn't need.
    """
    return (
        "You are Ollagent, a helpful and knowledgeable assistant. "
        "Answer the user conversationally and clearly. You have no tools "
        "and no filesystem access in this mode, so reply in plain natural "
        "language. Be concise but complete. If the user starts asking you "
        "to write/edit files or run commands, just say that a coding task "
        "was detected and they should switch modes."
    )


class Agent:
    def __init__(self, cfg: Config, history: Optional[List[Mapping]] = None):
        self.cfg = cfg
        self.llm = Ollama(cfg)
        self.ctx = ToolContext(cfg, workspace=cfg.workspace, cwd=cfg.workspace)
        self.sys_prompt = build_system_prompt(cfg.workspace)
        # If True, run() answers conversationally (no tools, no JSON).
        self.chat_mode = cfg.chat
        # The full in-memory conversation (system message stays at index 0).
        self.messages: List[Mapping[str, str]] = [
            {"role": "system", "content": self.sys_prompt}
        ]
        if history:
            self.messages.extend(list(history))
        self.last_answer = ""
        self.last_stopped = False

    # -- session / resume helpers --------------------------------------
    def history(self) -> List[Mapping[str, str]]:
        """Saveable conversation (everything except the system prompt)."""
        return list(self.messages[1:])

    def new(self) -> None:
        """Drop the conversation but keep the system prompt."""
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        self.last_answer = ""
        self.last_stopped = False

    # -- public entry point ------------------------------------------
    def run(self, task: str) -> str:
        """Run the loop for one task and return the final answer string.

        If `chat_mode` is on, this is a plain conversational turn (no tools)
        instead of the agent tool-loop. The conversation accumulates on
        `self.messages` so you can resume with another `run()` call.
        Press Ctrl+C (`KeyboardInterrupt`) to stop the current turn.
        """
        if self.chat_mode:
            return self._chat_run(task)
        self.messages.append({"role": "user", "content": task})
        final = ""
        try:
            for _ in range(self.cfg.max_iterations):
                reply = self._request(self.messages)
                decision = self._parse(reply, self.messages)
                if isinstance(decision, ToolResult):
                    if decision.conversational:
                        # A plain conversational reply — no tool ran.
                        self.messages.append({"role": "assistant",
                                              "content": decision.text})
                        print(f"\n{decision.text}")
                        final = decision.text
                        break
                    self.messages.append({"role": "assistant", "content": reply})
                    self.messages.append({"role": "user", "content": (
                        f"[tool observation]\n{decision.text}")})
                    first = decision.text.splitlines()[0] if decision.text else ""
                    print(f"\n\033[90m[agent] {reply[:200]}\033[0m")
                    print(f"\033[90m[{first[:120]}]\033[0m")
                    if decision.should_stop:
                        final = decision.text
                        break
                elif isinstance(decision, str):
                    self.messages.append({"role": "assistant", "content": reply})
                    self.messages.append({"role": "user", "content": (
                        f"[parse error]\n{decision}")})
        except KeyboardInterrupt:
            # Graceful stop (like Agy/Cline's stop button). We keep the
            # conversation so far so the user can resume or give a new task.
            self.last_stopped = True
            print("\n\033[33m[stop] — task interrupted. "
                  "Type a new instruction to continue, or 'exit' to quit.\033[0m")
        self.last_answer = final
        return final

    def _chat_run(self, task: str) -> str:
        """Pure conversational turn: no JSON, no tools, natural reply."""
        # Swap the system prompt for the conversational (no-tools) one.
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": build_chat_prompt()}
        self.messages.append({"role": "user", "content": task})
        try:
            reply = self._chat_raw(self.messages, format_json=False).strip()
        except KeyboardInterrupt:
            self.last_stopped = True
            print("\n\033[33m[stop] — interrupted.\033[0m")
            self.last_answer = ""
            return ""
        except OllamaError as exc:
            raise
        self.messages.append({"role": "assistant", "content": reply})
        self.last_answer = reply
        return reply

    # -- ollama interaction ------------------------------------------

    # -- ollama interaction ------------------------------------------
    def _chat_raw(self, messages: List[Mapping], format_json: bool) -> str:
        """Call the model, retrying with a smaller context on empty replies.

        Small local models (especially 4B-class on CPU-only machines) can
        return an *empty* response when the requested context window is too
        large for available RAM. Halving the window usually fixes it.
        """
        ctx = self.cfg.num_ctx
        for _ in range(3):
            reply = self.llm.chat(
                messages,
                temperature=self.cfg.temperature,
                num_ctx=ctx,
                format_json=format_json,
            )
            if reply.strip():
                return reply
            smaller = max(2048, ctx // 2)
            if smaller == ctx:
                break
            print(f"\033[33m[warn] empty model response — "
                  f"retrying with num_ctx={smaller}\033[0m")
            ctx = smaller
        raise OllamaError(
            "The model returned an empty response even after lowering the "
            f"context window to {ctx} tokens.\n"
            "  • Try a smaller --ctx (e.g. --ctx 4096).\n"
            "  • Or use a lighter model (ollama list to see options)."
        )

    def _request(self, messages: List[Mapping]) -> str:
        try:
            return self._chat_raw(messages, format_json=True)
        except OllamaError as exc:
            print(f"\033[31m[network error] {exc}\033[0m")
            raise
# -- JSON parsing with recovery -----------------------------------
    def _parse(self, reply: str, messages: List[Mapping]) -> Any:
        """Try to parse `reply` into a decision.

        Returns:
          - ToolResult when we should execute a tool (or stop via done),
          - str error message when the JSON was malformed / call invalid.
        """
        obj = self._try_parse(reply)
        if obj is None:
            return ("Your previous reply was not valid JSON. "
                    "Respond with ONLY a JSON object: "
                    "{'tool': name, 'arguments': {...}}, or "
                    "{'type': 'answer', 'text': '...'}, or "
                    "{'done': true, 'answer': ...}.")

        if obj.get("type") == "answer" or ("answer" in obj and "tool" not in obj):
            # A plain conversational reply — no tool ran.
            text = str(obj.get("text") or obj.get("answer") or "").strip()
            if text:
                return ToolResult(text, should_stop=True, conversational=True)
            # Fall through to tool handling if empty.

        if obj.get("done"):
            return ToolResult(f"DONE: {obj.get('answer', '')}", should_stop=True)

        tool = obj.get("tool")
        if not tool:
            return ("Your JSON must include a 'tool' key, a 'done' key, or a "
                    f"'type': 'answer' object. Got keys: {list(obj.keys())}")

        args = obj.get("arguments")
        if not isinstance(args, dict):
            args = {}

        try:
            return dispatch(self.ctx, str(tool), args)
        except ToolError as exc:
            return f"Tool error: {exc}"

    def _try_parse(self, reply: str) -> Optional[Dict[str, Any]]:
        """Strongly attempt to extract a JSON object from the reply."""
        text = reply.strip()
        # Direct parse first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # First balanced {...} block (handles stray prose around the JSON).
        m = _TAG_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Fenced code block.
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        return None