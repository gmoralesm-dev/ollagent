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


def build_system_prompt(workspace: str,
                        use_native_tools: bool = False) -> str:
    """Describe the environment and (for legacy mode) the JSON call format."""
    if use_native_tools:
        # With Ollama native tool calling, the server injects the tool
        # schemas and enforces the format — no JSON protocol needed here.
        return f"""You are Ollagent, an autonomous coding agent that works inside a terminal.
You help complete coding tasks by interacting with the local filesystem and shell.

WORKSPACE
The user's project root is: {workspace}
All work should happen inside this directory unless asked otherwise.

RULES
- Call tools only when the task truly needs the filesystem or shell; otherwise
  just answer directly in plain text.
- To inspect code first use read_file / list_files / search_files, then edit or
  run commands as needed.
- Prefer edit_file over write_file for small changes.
- If a tool reports an error, read the error, adapt, and retry.
- Verify your work (run builds/tests) before claiming completion. Be concise."""
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


def build_ollama_tools() -> List[Dict[str, Any]]:
    """Convert TOOL_SCHEMAS into Ollama's native tool-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


class Agent:
    def __init__(self, cfg: Config, history: Optional[List[Mapping]] = None):
        self.cfg = cfg
        self.llm = Ollama(cfg)
        self.ctx = ToolContext(cfg, workspace=cfg.workspace, cwd=cfg.workspace)
        # Use Ollama's native tool calling when the model supports it;
        # otherwise fall back to the prompt-based JSON protocol.
        self.native_tools = "tools" in self.llm.capabilities()
        self.ollama_tools = build_ollama_tools() if self.native_tools else []
        self.sys_prompt = build_system_prompt(
            cfg.workspace, use_native_tools=self.native_tools)
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
        if not self.native_tools:
            print("\033[90m[model has no native tool support — "
                  "using JSON-prompt fallback]\033[0m")

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
                reply, tool_calls = self._request(self.messages)
                if self.native_tools:
                    if tool_calls:
                        self.messages.append({
                            "role": "assistant",
                            "content": reply,
                            "tool_calls": tool_calls,
                        })
                        stop = self._run_tool_calls(tool_calls)
                        if stop is not None:
                            final = stop
                            break
                        continue
                    if reply.strip():
                        # No tool call → the model's plain conversational answer.
                        self.messages.append({"role": "assistant",
                                              "content": reply})
                        final = reply
                        break
                    continue  # empty reply — _chat_raw already retried
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
            reply, _tool_calls = self._chat_raw(
                self.messages, format_json=False)
            reply = reply.strip()
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

    def _run_tool_calls(
        self, tool_calls: List[Mapping]
    ) -> Optional[str]:
        """Execute native tool calls and feed observations back to the model.

        Returns a final answer string when the loop should stop (a `done`
        tool call), else None to keep iterating.
        """
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if not isinstance(args, dict):
                args = {}
            print(f"\n\033[90m[agent] {name}({json.dumps(args)[:150]})\033[0m")
            try:
                result = dispatch(self.ctx, name, args)
            except ToolError as exc:
                result = ToolResult(f"Tool error: {exc}")
            first = result.text.splitlines()[0] if result.text else ""
            print(f"\033[90m[{first[:120]}]\033[0m")
            self.messages.append({
                "role": "tool",
                "tool_name": name,
                "content": result.text,
            })
            if result.should_stop:
                return result.text
        return None

    # -- ollama interaction ------------------------------------------
    def _chat_raw(self, messages: List[Mapping], format_json: bool) -> tuple:
        """Call the model, retrying with a larger context on truncated replies.

        Small local models (especially reasoning ones like qwen3.5) can spend
        the whole context window thinking and stop with done_reason="length"
        and an *empty* content. Shrinking the window makes that worse, so on
        an empty reply we RAISE the window instead (up to a sane cap).
        """
        ctx = self.cfg.num_ctx
        for _ in range(3):
            reply, tool_calls, done_reason = self.llm.chat(
                messages,
                temperature=self.cfg.temperature,
                num_ctx=ctx,
                format_json=format_json,
                tools=self.ollama_tools or None,
            )
            if reply.strip() or tool_calls:
                return reply, tool_calls
            bigger = min(ctx * 2, 32768)
            if bigger == ctx:
                break
            why = "response truncated by the context limit" \
                if done_reason == "length" else "empty model response"
            print(f"\033[33m[warn] {why} — "
                  f"retrying with num_ctx={bigger}\033[0m")
            ctx = bigger
        raise OllamaError(
            "The model returned an empty response even after raising the "
            f"context window to {ctx} tokens.\n"
            "  • Try a larger --ctx (e.g. --ctx 8192).\n"
            "  • Reasoning models can burn the window 'thinking'; run with\n"
            "    --think off (default) or use a non-reasoning model."
        )

    def _request(self, messages: List[Mapping]) -> tuple:
        try:
            return self._chat_raw(messages, format_json=not self.native_tools)
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