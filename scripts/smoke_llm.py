"""End-to-end proof the proxy works for us: (1) plain chat, (2) a full native
tool-calling round trip. Run after any model/router change. If step 2 fails,
nothing agentic in this repo can work — fix the model before touching the graph.

Also prints which model actually answered: with qwen-chat disabled on the
router you'll see the cloud failover's name here.

Usage: uv run python scripts/smoke_llm.py
"""

from __future__ import annotations

import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from playtime.llm import make_llm


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def main() -> None:
    llm = make_llm("agent")

    # 1. Plain chat.
    resp = llm.invoke([HumanMessage("Reply with exactly: PONG")])
    served_by = resp.response_metadata.get("model_name", "?")
    print(f"[1] chat        ok  served_by={served_by!r}  reply={resp.text!r}")
    assert "PONG" in resp.text.upper(), "model did not follow a trivial instruction"

    # 2. Tool round trip: model must (a) emit a native tool_call, (b) produce a
    # final answer once we send the tool's result back. This is the exact
    # message shape the LangGraph agent loop will use from stage 2 on.
    with_tools = llm.bind_tools([get_current_time])
    msgs = [HumanMessage("What time is it right now? Use your tool.")]
    ai = with_tools.invoke(msgs)
    assert ai.tool_calls, f"model answered without calling the tool: {ai.text!r}"
    call = ai.tool_calls[0]
    print(f"[2] tool_call   ok  {call['name']}({call['args']})")

    result = get_current_time.invoke(call["args"])
    msgs += [ai, ToolMessage(content=result, tool_call_id=call["id"])]
    final = with_tools.invoke(msgs)
    assert final.text and not final.tool_calls, "no final answer after tool result"
    print(f"[3] tool answer ok  reply={final.text!r}")

    print("\nsmoke test PASSED — the model can drive an agent loop.")


if __name__ == "__main__":
    main()
