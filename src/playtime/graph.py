"""Stage 2: the ReAct loop as an explicit LangGraph graph.

    START → agent ⇄ tools
              ↓
             END

This is the standard ReAct shape before the prepare_context and approval
nodes arrive (stages 3–4 here). LangGraph ships create_react_agent() that
does all of this in one call — we hand-write it so the loop stays visible.

Two ideas carry this file:

1. STATE + REDUCER. AgentState is just a message list, but annotated with
   add_messages: whatever a node RETURNS is appended/merged into state, not
   assigned. Nodes never mutate; they emit updates.

2. CHECKPOINTER = MEMORY. compile(checkpointer=...) saves state after every
   node, keyed by thread_id. Invoking with the same thread_id resumes that
   conversation; a new thread_id is a blank one. The LLM is as stateless as
   in stage 1 — the graph re-sends the thread's history every turn. MemorySaver
   lives in-process, so a restart forgets everything (stage 4: SQLite).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from playtime.llm import make_llm
from playtime.tools import TOOLS

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant in a terminal chat. You have tools; "
    "use one when it fits the request, otherwise just answer. Never claim you "
    "did something a tool does without actually calling it."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    tools_by_name = {t.name: t for t in TOOLS}
    # bind_tools attaches the tools' JSON schemas to every request so the model
    # can emit native tool_calls. Binding ≠ executing — that's the tools node.
    llm = make_llm("agent").bind_tools(TOOLS)

    def agent(state: AgentState) -> dict:
        # System prompt is injected fresh each call, not stored in state:
        # editing it later shouldn't require migrating saved conversations.
        resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [resp]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def tools(state: AgentState) -> dict:
        """Execute every tool call in the last AI message. INVARIANT (models
        break if violated): each tool_call gets exactly one ToolMessage answering
        its id — even the failures, which is why errors are returned as results
        instead of raised. The model reads the error and self-corrects."""
        last = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            t = tools_by_name.get(call["name"])
            if t is None:
                content = f"ERROR: no tool named {call['name']!r}"
            else:
                try:
                    content = t.invoke(call["args"])  # raw: no timeout/caps yet
                except Exception as e:
                    content = f"ERROR: {e}"
            out.append(ToolMessage(content=str(content), tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")  # the ReAct loop: results go back to the brain
    return g.compile(checkpointer=MemorySaver())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
