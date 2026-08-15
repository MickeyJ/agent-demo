# agent-playtime — stage notes

A minimal, staged playground for learning:
local models via LiteLLM, LangGraph, context/memory management, tool contracts,
guardrails, sub-agents, and a toy RAG tool. Each stage is small enough to read
end to end.

## Roadmap

1. **Hello, local model** — skeleton, `make_llm(role)`, smoke test, stateless REPL ← *you are here*
2. **The ReAct loop** — LangGraph state/nodes/edges, checkpointer, two raw tools
3. **Tool contract + guardrails** — ToolSpec, validation, timeouts, loop breaker, `interrupt()` approval
4. **Context & memory** — token budgets, window trimming, rolling summary, pinned facts, SQLite persistence
5. **Sub-agents as tools** — stylist + critic agents wrapped as tools of the master agent
6. **Mini RAG** — mock wiki docs, proxy embeddings, `search_docs` tool

## Stage 1 — Hello, local model

Files: `settings.py`, `llm.py`, `main.py`, `scripts/smoke_llm.py`.

What to internalize here:

- **The proxy is the seam.** `llm.py` is the only file that knows a base URL or
  imports `langchain_openai`. The `qwen-chat` alias is whatever the router
  serves — local vLLM when enabled, cloud failover otherwise. The REPL prints
  `bot (<model>)` so you always know who actually answered.
- **Roles, not models.** Code asks for `"agent"` or `"utility"`; `.env` decides
  what backs each. A common production split is a hosted brain + local utility.
- **An LLM call is stateless.** The REPL sends only `[system, current message]`.
  Tell it your name, then ask for it — it can't know. "Memory" is always just
  context you re-send; the rest of this project is about assembling that
  context well.
- **The smoke test is load-bearing** (rule: run it after any
  model change). It proves the *native tool-calling round trip*: model emits
  `tool_calls` → you run the tool → you append an `AIMessage` + `ToolMessage`
  pair → model produces a final answer. That exact message shape is the inner
  loop of every agent framework; LangGraph just automates it.

Try:

```bash
uv sync
uv run python scripts/smoke_llm.py   # must pass before anything else matters
uv run playtime                      # chat; observe the amnesia
```
