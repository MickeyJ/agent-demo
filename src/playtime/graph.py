"""Stage 4: context management joins the loop.

    START → prepare_context → agent ⇄ tools
                                ↓ ↖ approval

State grew two fields, and the assembled prompt now has four layers:

    [system prompt | pinned facts | rolling summary]  ← one SystemMessage
    [trimmed recent messages]                         ← the window

prepare_context runs once per user turn: trims the window to budget, folds
whatever fell out into the rolling summary (utility model), and removes the
evicted messages from checkpointed state (RemoveMessage). So state itself
stays small — the summary IS the memory of everything old.

The checkpointer is SQLite now: kill the process, restart, /thread back —
the conversation, summary, and pinned facts are all still there.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from playtime.context import evicted_messages, trim_window, update_summary
from playtime.llm import make_llm
from playtime.settings import get_settings
from playtime.toolkit import execute
from playtime.tools import REMEMBER_FACT, TOOLS

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant in a terminal chat. You have tools; "
    "use one when it fits the request, otherwise just answer. Never claim you "
    "did something a tool does without actually calling it."
)

TOOL_BUDGET_MSG = (
    "I hit my tool-call budget for this request — here's what I have so far. "
    "Ask me to continue if you want me to keep going."
)

DECLINED_MSG = (
    "USER DECLINED this action; it was NOT executed. Do not retry. "
    "Acknowledge the decline and ask what they'd like instead."
)


def merge_pinned(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_loops: int  # no reducer → last write wins
    summary: str  # rolling digest of evicted turns
    pinned: Annotated[dict[str, str], merge_pinned]  # remember_fact's target


def assemble_system(state: AgentState) -> SystemMessage:
    """Layered prompt, most-stable content first (good for prefix caching)."""
    parts = [SYSTEM_PROMPT]
    if state.get("pinned"):
        facts = "\n".join(f"- {k}: {v}" for k, v in state["pinned"].items())
        parts.append(f"PINNED FACTS (established earlier, still true):\n{facts}")
    if state.get("summary"):
        parts.append(f"CONVERSATION SUMMARY (older turns, verbatim window follows):\n{state['summary']}")
    return SystemMessage("\n\n".join(parts))


def build_graph(checkpointer=None):
    registry = {t.name: t for t in TOOLS}
    llm = make_llm("agent").bind_tools([t.openai_schema() for t in TOOLS])

    def prepare_context(state: AgentState) -> dict:
        out: dict = {"tool_loops": 0}
        messages = state["messages"]
        trimmed = trim_window(messages)
        evicted = evicted_messages(messages, trimmed)
        if evicted:
            try:
                out["summary"] = update_summary(state.get("summary", ""), evicted)
            except Exception:
                pass  # degrade, never crash the turn: evict without folding
            # RemoveMessage is add_messages' delete instruction: these ids
            # disappear from checkpointed state, not just from this prompt.
            out["messages"] = [RemoveMessage(id=m.id) for m in evicted if m.id]
        return out

    def agent(state: AgentState) -> dict:
        resp = llm.invoke([assemble_system(state), *state["messages"]])
        loops = state.get("tool_loops", 0) + (1 if resp.tool_calls else 0)
        if resp.tool_calls and loops > get_settings().max_tool_loops:
            partial = resp.text
            resp = AIMessage(content=(partial + "\n\n" if partial else "") + TOOL_BUDGET_MSG)
        return {"messages": [resp], "tool_loops": loops}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return END
        if any(registry[c["name"]].requires_approval for c in calls if c["name"] in registry):
            return "approval"
        return "tools"

    def approval(state: AgentState) -> Command:
        last = state["messages"][-1]
        gated = [c for c in last.tool_calls if registry.get(c["name"]) and registry[c["name"]].requires_approval]
        card = "; ".join(f"{c['name']}({c['args']})" for c in gated)
        decision = interrupt({"card": card})
        if decision == "approve":
            return Command(goto="tools")
        refusals = [
            ToolMessage(content=DECLINED_MSG, tool_call_id=c["id"], name=c["name"])
            for c in last.tool_calls
        ]
        return Command(goto="agent", update={"messages": refusals})

    def tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        out: list[ToolMessage] = []
        pinned: dict[str, str] = {}
        for call in last.tool_calls:
            spec = registry.get(call["name"])
            if spec is None:
                content = f"ERROR: no tool named {call['name']!r}"
            elif call["name"] == REMEMBER_FACT:
                # Special-cased: tools can't reach graph state, the node can.
                args = call["args"] or {}
                pinned[str(args.get("key", "fact"))] = str(args.get("value", ""))
                content = f"Pinned {args.get('key')}."
            else:
                content = execute(spec, call["args"])
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        update: dict = {"messages": out}
        if pinned:
            update["pinned"] = pinned  # merge_pinned folds it into state
        return update

    g = StateGraph(AgentState)
    g.add_node("prepare_context", prepare_context)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("approval", approval, destinations=("tools", "agent"))
    g.add_edge(START, "prepare_context")
    g.add_edge("prepare_context", "agent")
    g.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "approval": "approval", END: END}
    )
    g.add_edge("tools", "agent")
    if checkpointer is None:
        Path("var").mkdir(exist_ok=True)
        conn = sqlite3.connect("var/checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
