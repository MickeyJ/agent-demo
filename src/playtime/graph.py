"""Stage 3: guardrails join the loop.

    START → prepare → agent ⇄ tools
                        ↓ ↖______
                     approval    ↘
                        ↓ (decline)
                       END-ish (answers calls, back to agent)

New since stage 2:
- prepare node: resets the per-turn tool-loop counter (grows into full context
  management in stage 4).
- loop breaker in agent: local models can ping-pong tool calls forever; after
  max_tool_loops the turn ends with a partial-results message instead.
- approval node: if any requested call has requires_approval, interrupt() pauses
  the graph — state checkpointed, process free to do anything else — until
  someone resumes it with a decision. Approve → run the calls; decline → the
  calls are answered with a refusal ToolMessage (NEVER left unanswered: every
  tool_call id must get a ToolMessage or the model breaks) and the agent gets
  to respond gracefully.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from playtime.llm import make_llm
from playtime.settings import get_settings
from playtime.toolkit import execute
from playtime.tools import TOOLS

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


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_loops: int  # no reducer → last write wins


def build_graph():
    registry = {t.name: t for t in TOOLS}
    llm = make_llm("agent").bind_tools([t.openai_schema() for t in TOOLS])

    def prepare(state: AgentState) -> dict:
        return {"tool_loops": 0}

    def agent(state: AgentState) -> dict:
        resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        loops = state.get("tool_loops", 0) + (1 if resp.tool_calls else 0)
        if resp.tool_calls and loops > get_settings().max_tool_loops:
            # Circuit breaker. Replacing the message (not appending) keeps the
            # window clean: the over-budget tool_calls never enter state, so
            # there are no orphaned calls to answer.
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
        # interrupt() checkpoints the graph and raises; the value is surfaced to
        # whoever invoked us. On resume the NODE RE-RUNS from the top and
        # interrupt() returns the resume value instead of raising.
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
        out = []
        for call in last.tool_calls:
            spec = registry.get(call["name"])
            if spec is None:
                content = f"ERROR: no tool named {call['name']!r}"
            else:
                content = execute(spec, call["args"])  # the guarded path, always
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    g = StateGraph(AgentState)
    g.add_node("prepare", prepare)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("approval", approval, destinations=("tools", "agent"))
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "approval": "approval", END: END}
    )
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
