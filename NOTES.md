# agent-playtime — stage notes

A minimal, staged playground for learning:
local models via LiteLLM, LangGraph, context/memory management, tool contracts,
guardrails, sub-agents, and a toy RAG tool. Each stage is small enough to read
end to end.

## Roadmap

1. **Hello, local model** — skeleton, `make_llm(role)`, smoke test, stateless REPL
2. **The ReAct loop** — LangGraph state/nodes/edges, checkpointer, two raw tools ← *you are here*
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

## Stage 2 — The ReAct loop

Files: `graph.py`, `tools.py` (new), `main.py` (rewritten).

What to internalize here:

- **Nodes emit updates; reducers merge them.** `AgentState.messages` is
  annotated with `add_messages`, so what a node returns gets *appended*, never
  assigned. This is the core LangGraph mental model.
- **The loop is three pieces**: `agent` (LLM with tools bound), a conditional
  edge (tool calls? → `tools`, else END), and `tools` (execute calls, return
  `ToolMessage`s) looping back to `agent`. It's `smoke_llm.py`'s manual round
  trip, automated. We hand-write the `tools` node instead of using LangGraph's
  prebuilt `ToolNode`/`create_react_agent` so nothing is hidden.
- **Memory = checkpointer, nothing else.** The graph saves state after every
  node, keyed by `thread_id`. The LLM stays stateless; history is re-sent each
  turn. `MemorySaver` is in-process → restart forgets everything (stage 4).
- **One ToolMessage per tool_call id, always** — even for failures, which is
  why the tools node returns errors as results instead of raising. Models
  break on orphaned tool calls; the error-as-result lets them self-correct.

Try:

```bash
uv run playtime
# tell it your name → ask for it (memory works now)
# "roll 3 dice", "shout hello for me" → watch ⚙ call / → result lines
# /new → ask your name again (fresh thread knows nothing)
# /thread t1 → it remembers again; restart the process → all gone
```

Planted flaws to find (stage 3 fixes them): tools have no timeout, no arg
limits, no result caps — "roll a million dice" hangs the whole turn.
