"""
ollagent - A local terminal agent for Ollama models.

Runs an agentic loop: the model proposes JSON tool calls, the CLI executes
them (file reads/writes, searches, shell commands), and feeds results back
until the task is complete. No browser, no UI - just the terminal.
"""

__version__ = "0.3.0"
__all__ = ["client", "tools", "agent", "cli", "config"]