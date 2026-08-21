<div align="center">

[![Version](https://img.shields.io/badge/version-0.4.0-8B5CF6?style=for-the-badge)](https://github.com/gmoralesm-dev/ollagent/releases)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/gmoralesm-dev/ollagent?style=for-the-badge&color=yellow)](https://github.com/gmoralesm-dev/ollagent/stargazers)

[![LAZY-LOL](https://img.shields.io/badge/LAZY--LOL-Methodology-8B5CF6?style=for-the-badge&logo=book&logoColor=white)](https://github.com/gmoralesm-dev/lazy-lol-book)

</div>

# Ollagent 🦙

> **A local terminal coding agent for your own Ollama models.**
> Reads code, edits files, runs shell commands, verifies its work — looping until the task is done. No cloud, no API keys, no browser UI.

Ollagent brings the Cline / Aider / Agy agentic workflow to models running fully on your machine:

```
you give a task
      ↓
ollama proposes a tool call        read_file(path="main.py")   ← native function calling
      ↓
ollagent executes it for real      (reads/writes files, runs bash)
      ↓
result feeds back as observation   [exit code 0] ...
      ↓
repeat until done                  done(summary="...")
```

It also knows when **not** to use tools — casual questions get plain conversational answers, coding tasks get the full tool loop.

---

## ✨ Highlights

| | |
|---|---|
| 🔧 **7 built-in tools** | `read_file` · `write_file` · `edit_file` · `list_files` · `search_files` · `run_command` · `done` |
| 💬 **Chat + agent modes** | questions answered conversationally; tasks drive the tool loop |
| ⏹️ **Stop anytime** | `Ctrl+C` halts mid-task cleanly — no broken state |
| 💾 **Resumable sessions** | conversation auto-saves; pick it back up later |
| 🔒 **Sandboxed by default** | file access confined to your workspace; approval prompts before writes/commands |
| 🛡️ **Small-model guardrails** | smart-quote sanitizing, truncated-response retry with automatic context growth |
| 🧩 **Zero dependencies** | pure Python stdlib — nothing to pip install |

## 🚀 Quick start

```bash
git clone https://github.com/gmoralesm-dev/ollagent.git
cd ollagent

# conversational
python3 -m ollagent --model qwen2.5-coder:7b "what is a script?"

# agentic coding task
python3 -m ollagent --model qwen2.5-coder:7b --ctx 4096 \
    --yolo --cwd ./myproject "add input validation to main.py"
```

Interactive session (with stop/resume):

```bash
python3 -m ollagent            # Ctrl+C stops a task, 'exit' saves & quits,
                               # run again in the same folder to resume
```

> **CPU-only tip:** use `--ctx 4096` and leave thinking off (the default) — reasoning models can burn the whole window thinking. If a reply comes back empty, Ollagent auto-retries with a larger context.
## 🧠 Chat vs Agent — how routing works

Every message is routed before the model sees it:

| Message | Route | Behavior |
|---|---|---|
| *"what is a script?"* | 💬 Chat | plain natural-language answer, no tools |
| *"add validation to main.py"* | 🔧 Agent | tool loop: read → edit → run → verify → done |

- Detection is conservative: messages containing action verbs (`create`, `fix`, `run`, …) always go to agent mode.
- Force chat for everything with `--chat`.

## ⏹️ Stop & resume

```
Ctrl+C          stops the current task (like Agy/Cline's Stop button)
exit / quit     saves the session and leaves
new             clears the conversation
--session FILE  keep named sessions (one per feature branch)
--fresh         ignore the saved session
```

Run `ollagent` again in the same folder and it picks up exactly where you left off — the model keeps its full prior context.

## 📦 CLI reference

```
python3 -m ollagent --help
  --model MODEL          Ollama model to use
  --url URL              Ollama server URL (default http://localhost:11434)
  --ctx N                context window in tokens
  --temp F               sampling temperature (default 0.1)
  --max-iterations N     cap on tool-call loop iterations
  --think                enable the model's thinking/reasoning phase (default off)
  --cwd DIR              workspace directory
  --session FILE         session file to load/save
  --fresh                start a fresh conversation
  --chat                 force conversational mode (no tools)
  --yolo                 auto-approve writes & commands (use scratch dirs!)
  --no-sandbox           allow access outside the workspace
  --list-models          list models on the server
```

## 🏗️ Architecture

```
ollagent/
├── cli.py      # argparse, chat/agent routing, sessions, REPL
├── agent.py    # the loop: system prompt, JSON parsing, retry logic
├── tools.py    # tool implementations, sandboxing, approvals, sanitizer
├── client.py   # stdlib-only Ollama HTTP client
└── config.py   # settings & env overrides
```

Design decisions worth knowing:

- **Native tool calling with JSON-prompt fallback** — models that advertise the `tools` capability use Ollama's native function calling (`message.tool_calls`), which removes fragile JSON parsing entirely. Models without it automatically fall back to the prompt-based JSON protocol.
- **Thinking off by default (`--think` to enable)** — reasoning models like qwen3.5 can spend the whole context window thinking and return an empty answer; Ollagent disables the thinking phase unless you ask for it.
- **Smart-quote sanitizer** — small models love writing `’` instead of `'`, which breaks code. Ollagent normalizes typographic quotes on every write/edit to code files.
- **Empty-response retry grows the context** — when a model stops with an empty reply (typically `done_reason: "length"` after a long thinking phase), shrinking the window makes it worse, so Ollagent doubles `num_ctx` (up to 32k) and retries instead.

## 🔧 Ollama troubleshooting

| Symptom | Fix |
|---|---|
| Empty / blank responses | leave thinking off; Ollagent auto-retries with a larger `--ctx` |
| Model reloads every run | `export OLLAMA_KEEP_ALIVE=-1` on the server |
| Browser apps get CORS errors | `export OLLAMA_ORIGINS="*"` (not needed for this CLI) |
| Remote Ollama | `export OLLAMA_HOST=0.0.0.0:11434`, then `--url http://host:11434` |
| GPU not used | install the vendor driver; verify with `nvidia-smi`, check `ollama ps` |

## 📚 Built with the LAZY-LOL methodology

Ollagent was designed and documented following **LAZY-LOL**, my open methodology for structuring software projects — Lazy scaffolding, then Layered, Observable, Loosely-coupled builds.

👉 **Read the book:** [gmoralesm-dev/lazy-lol-book](https://github.com/gmoralesm-dev/lazy-lol-book)

## 🗺️ Roadmap

- [ ] Native Ollama tool-calling (`tools` array) for models that support it
- [ ] Git auto-commit checkpoints around every edit
- [ ] Aider-style repo map (functions/classes/imports) in the system prompt
- [ ] Persistent PTY shell so `cd`/env survive between commands
- [ ] Streaming output during generation

## 📄 License

MIT — see [LICENSE](LICENSE).

