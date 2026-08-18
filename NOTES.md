# agent-playtime — stage notes

A minimal, staged playground for learning:
local models via LiteLLM, LangGraph, context/memory management, tool contracts,
guardrails, sub-agents, and a toy RAG tool. Each stage is small enough to read
end to end.

## Roadmap

1. **Hello, local model** — skeleton, `make_llm(role)`, smoke test, stateless REPL
2. **The ReAct loop** — LangGraph state/nodes/edges, checkpointer, two raw tools
3. **Tool contract + guardrails** — ToolSpec, validation, timeouts, loop breaker, `interrupt()` approval
4. **Context & memory** — token budgets, window trimming, rolling summary, pinned facts, SQLite persistence
5. **Sub-agents as tools** — stylist + critic agents wrapped as tools of the master agent
6. **Mini RAG** — mock wiki docs, proxy embeddings, `search_docs` tool
7. **Demo web UI** — FastAPI + SSE chat, approval cards, context panel, doc upload/paste ← *you are here*

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

## Stage 3 — Tool contract + guardrails

Files: `toolkit.py` (new), `tools.py` (rewritten to ToolSpecs), `graph.py`
(prepare/approval nodes, loop breaker), `main.py` (approval prompt).

What to internalize here:

- **One guarded execution path.** `toolkit.execute()` owns validation, timeout,
  error capture, and result caps; tool functions stay dumb (args in, string
  out). Guardrails live in the contract, not in each tool.
- **Errors are results, not exceptions.** INVALID_ARGUMENTS, timeouts, and
  crashes all come back as the ToolMessage text. The model reads the failure
  and self-corrects — "roll a million dice" now gets a polite "max is 100"
  instead of a hang, with zero prompt engineering.
- **Schemas are deliberate now.** Explicit pydantic args models with per-field
  `description=` and constraints (`ge`/`le`) replace docstring scraping.
- **`interrupt()` is a pause, not a prompt.** The approval node checkpoints the
  graph and stops the process; ANYTHING can resume it later with
  `Command(resume=...)` on the same thread. On decline, every pending tool
  call still gets a ToolMessage answer (the no-orphans invariant) so the model
  can respond gracefully.
- **The loop breaker** (`max_tool_loops`) exists because small local models can
  ping-pong tool calls forever. Try `MAX_TOOL_LOOPS=0 uv run playtime` to see
  it fire on the first call.

Try:

```bash
uv run playtime
# "roll a million dice"        → INVALID_ARGUMENTS → model self-corrects
# "email bob@x.com hi"         → ⚠ approval card; y runs it (see var/outbox/), n declines
# MAX_TOOL_LOOPS=0 uv run playtime → budget message on any tool use
```

## Stage 4 — Context & memory

Files: `context.py` (new), `graph.py` (prepare_context, layered prompt, SQLite),
`tools.py` (remember_fact), `toolkit.py` (LLM compaction), `main.py` (/debug).

The prompt the model sees each turn is now four layers:

    system prompt          — static rules
    pinned facts           — verbatim, model-chosen via remember_fact
    rolling summary        — utility-model digest of evicted turns
    trimmed window         — recent messages, ≤ ctx_window_budget tokens

What to internalize here:

- **Trimming has invariants.** An AIMessage with tool_calls and its
  ToolMessages evict together or not at all, and the window never starts with
  a ToolMessage — `trim_messages(start_on="human")` guarantees both. Violate
  them and tool-calling models break outright.
- **Eviction ≠ deletion.** Evicted turns are folded into the rolling summary
  by the utility model, then removed from checkpointed state via
  RemoveMessage. Lossy by design; that's the trade.
- **Two memory mechanisms, different guarantees.** The summary is cheap and
  lossy; pinned facts are verbatim and survive anything — the model decides
  what deserves pinning (`remember_fact` is special-cased in the tools node
  because tools can't touch graph state; nodes can).
- **Compaction beats truncation.** Oversized tool results now get compressed
  by the utility model ("preserve all numbers verbatim") instead of chopped.
- **Persistence is just a checkpointer swap.** MemorySaver → SqliteSaver
  (var/checkpoints.sqlite); nothing else changed, and threads survive restarts.

Try:

```bash
CTX_WINDOW_BUDGET=250 uv run playtime   # tiny budget so trimming fires fast
# "remember my budget is $50"  → ⚙ remember_fact → pinned
# ask for a few two-sentence stories → watch for the ✂ folded-into-summary line
# /debug  → the exact layered prompt + per-message token costs
# /quit, rerun, ask about old details → SQLite persistence
```

## Stage 5 — Sub-agents as tools

Files: `subagents.py` (new), `tools.py` (SHOUT extracted, off the master's
menu), `graph.py` (ALL_TOOLS = TOOLS + SUBAGENTS).

A sub-agent is a tool whose implementation is another LLM. The master can't
tell — same ToolSpec contract as roll_dice. What to internalize:

- **Context isolation.** The sub-agent sees ONLY its args, never the
  conversation. The master must pass everything relevant — watch it fail:
  establish a fact in chat, then ask for a rewrite that depends on it.
- **Private tools shrink the master's menu.** `shout` now belongs to the
  stylist; the master delegates. This is the scaling answer to "30 tools
  confuse a small model": group specialists behind sub-agents.
- **An agent is a loop, not a framework.** The stylist's ReAct loop is ~10
  lines of plain Python — no LangGraph. The framework buys checkpointing and
  interrupts; a fire-and-forget sub-agent needs neither.
- **Observability nests.** The master's stream shows only ⚙ stylist(...) →
  result; the stylist's internal shout call is invisible here (Langfuse traces
  would show every nested LLM call). Verified in testing: it does happen.
- Sub-agents don't have to be in-process: the same contract works over HTTP
  to a specialist running in another service.

Try:

```bash
uv run playtime
# 'shout "we are live"'          → master delegates: ⚙ stylist(...) (it has no shout of its own)
# "rewrite that as a haiku"      → stylist again, different style
# "write a tagline for X and have the critic review it"  → critic verdict
```

## Stage 6 — Mini RAG

Files: `rag.py` (new), `scripts/seed_docs.py` (new), `llm.py` (make_embeddings),
`graph.py` (+ SEARCH_DOCS).

RAG demystified — the whole thing is two paths in one file:

    ingest:  text → chunk → embed → sqlite blobs
    search:  query → embed → cosine vs all chunks → top-k → tool result

What to internalize:

- **Retrieval is just nearest-neighbor text.** No vector DB until the corpus
  outgrows RAM — numpy cosine over hundreds of chunks is instant. The
  embeddings come from the router's `embeddings` alias (local vllm-embed).
- **Chunk size is THE tuning knob** (chunk_text max_chars): too big → hits
  flood the window budget; too small → chunks lose their meaning.
- **The tool description does heavy lifting**: "always search before saying
  you don't know" is what makes the model reach for the KB instead of
  hallucinating or apologizing.
- **Retrieved text becomes context like anything else** — once a search result
  is in the window, follow-up questions can be answered from it without
  another search (watch: ask about wifi, then the mascot — the second answer
  rides on the first search's excerpts).
- Facts are deliberately unknowable (Dicecorp's mascot axolotl) so you can
  tell retrieval from pretraining.

Try:

```bash
uv run python scripts/seed_docs.py     # idempotent; 6 mock wiki docs
uv run playtime
# "what's the guest wifi password?"   → ⚙ search_docs → rollforinitiative
# "how many PTO days do I get?"       → 23, from the HR doc
```

## Stage 7 — Demo web UI

Files: `web/app.py` (FastAPI), `web/static/` (hand-written HTML/CSS/JS, no
build step, no CDN), `settings.py` (web_port 9020).

Nothing agent-side changed — that's the lesson. The graph, tools, context
machinery, and SQLite checkpoints are channel-agnostic; the web UI and the CLI
REPL are two skins over the same threads (open t1 in both and compare).

- **A turn is a stream.** POST /api/chat answers with SSE events — tool_call,
  tool_result, assistant, trim, approval, done — which is exactly main.py's
  run_turn() serialized. The frontend only renders; all agent concepts arrive
  as events.
- **Approvals become buttons.** The interrupt() card streams out as an
  `approval` event; Approve/Decline POSTs /api/resume, which streams the
  continuation. Same checkpoint mechanics as the REPL's y/N.
- **The Context tab is /debug, visualized**: window-budget gauge, pinned
  facts, rolling summary — refreshed after every turn.
- **The Docs tab exposes the RAG pipeline**: paste text or upload .txt/.md
  (same ingest path), list/delete docs, and a raw test-search box showing
  scored chunks — retrieval demoable without the agent in the way.

Demo flow: Docs tab → paste a memo with a made-up fact → test-search it →
back to chat → ask the agent → watch ⚙ search_docs cite your memo.

Try:

```bash
uv run playtime-web    # → http://127.0.0.1:9020 (host-only)
```
