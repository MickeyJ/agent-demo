"""Stage 4: context management.

The problem (you asked exactly this): the checkpointer re-sends the FULL history
every turn, forever. Cost and latency grow linearly; model quality degrades long
before the 64k window overflows. The fix is a layered budget:

  window   — recent turns, verbatim, trimmed to ctx_window_budget tokens
  summary  — everything evicted from the window, folded into a rolling
             LLM-written digest (utility role)
  pinned   — facts the model chose to keep verbatim via remember_fact

Invariants that BREAK tool-calling models when violated (trim_messages with
start_on="human" guarantees both):
- an AIMessage with tool_calls and its ToolMessages are one atomic unit —
  evict both or neither;
- the window never begins with a ToolMessage.

Token counting here is a chars/4 heuristic — deliberately. Budgets need to be
roughly right, not exact; a production system would use the model's real
tokenizer for exactness, at the cost of a tokenizers dependency.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import trim_messages

from playtime.llm import make_llm
from playtime.settings import get_settings

log = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        total += 4 + count_tokens(m.text)  # +4 ≈ per-message framing overhead
        for c in getattr(m, "tool_calls", None) or []:
            total += count_tokens(c["name"] + json.dumps(c["args"] or {}))
    return total


def trim_window(messages: list[BaseMessage]) -> list[BaseMessage]:
    return trim_messages(
        messages,
        max_tokens=get_settings().ctx_window_budget,
        token_counter=count_message_tokens,
        strategy="last",         # keep the most recent turns
        start_on="human",        # both invariants live in this line
        include_system=False,    # system prompt is budgeted separately
        allow_partial=False,
    )


def evicted_messages(
    original: list[BaseMessage], trimmed: list[BaseMessage]
) -> list[BaseMessage]:
    kept = {m.id for m in trimmed if m.id}
    return [m for m in original if m.id and m.id not in kept]


def render_for_summary(messages: list[BaseMessage]) -> str:
    """Terse rendering of evicted messages for the summary prompt — tool
    exchanges collapse to one line each."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.text}")
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
                lines.append(f"AGENT CALLED: {calls}")
            if m.text:
                lines.append(f"AGENT: {m.text}")
        elif isinstance(m, ToolMessage):
            lines.append(f"TOOL RESULT: {m.text[:200]}")
    return "\n".join(lines)


def update_summary(current_summary: str, evicted: list[BaseMessage]) -> str:
    """Fold evicted messages into the rolling summary. This is the utility
    role's job: cheap, mechanical, no tools needed."""
    s = get_settings()
    llm = make_llm("utility", max_tokens=s.summary_max_tokens, temperature=0.0)
    resp = llm.invoke(
        [
            HumanMessage(
                "Update the running summary of this conversation.\n"
                "Keep: facts the user stated, decisions, names, numbers, "
                "open questions, actions completed. Drop pleasantries. "
                "Output only the updated summary, no preamble.\n\n"
                f"CURRENT SUMMARY:\n{current_summary or '(none)'}\n\n"
                f"NEW MESSAGES TO FOLD IN:\n{render_for_summary(evicted)}"
            )
        ]
    )
    return resp.text.strip()


def compact_result(raw: str, cap_chars: int) -> str:
    """Task-aware compression of oversized tool results (upgrade from stage 3's
    hard truncation). Preserving numbers/names verbatim is the critical
    instruction — a summarized-away value means the model can't act on it."""
    if len(raw) <= cap_chars:
        return raw
    llm = make_llm("utility", max_tokens=max(100, cap_chars // 4), temperature=0.0)
    try:
        out = llm.invoke(
            [
                HumanMessage(
                    "Compress this tool output. HARD RULES: preserve ALL numbers, "
                    "ids, names, and totals verbatim; drop repetition and "
                    "boilerplate; output the compressed version only.\n\n" + raw[:20_000]
                )
            ]
        )
        return out.text.strip() or raw[:cap_chars]
    except Exception as e:
        log.warning("compaction failed (%s); hard-truncating", e)
        return raw[:cap_chars] + "\n...[truncated]"
