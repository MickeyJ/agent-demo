"""Stage 3 REPL: approval cards arrive.

When the graph hits interrupt(), the stream ends early and hands us the card.
The graph is now PAUSED — checkpointed, resumable anytime (in production the
resume could be a Slack button click hours later; here it's the y/N prompt).
We resume by streaming Command(resume=<decision>) into the same thread.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from playtime.graph import build_graph, thread_config

HELP = "commands: /new (fresh thread) | /thread <name> (switch) | /quit"


def run_turn(graph, payload, config) -> dict | None:
    """Stream one graph run; print progress. Returns the approval card if the
    run paused on an interrupt, None if it ran to completion."""
    card = None
    for step in graph.stream(payload, config, stream_mode="updates"):
        for node, update in step.items():
            if node == "__interrupt__":
                card = update[0].value
                continue
            if not update:  # nodes like prepare return no messages
                continue
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    for c in m.tool_calls:
                        print(f"  ⚙ {c['name']}({c['args']})")
                    if m.text:
                        print(f"\nbot> {m.text}")
                elif isinstance(m, ToolMessage):
                    print(f"    → {m.text}")
    return card


def main() -> None:
    graph = build_graph()
    thread = "t1"
    counter = 1
    print(f"agent-playtime stage 3 (guardrails). {HELP}")
    while True:
        try:
            user = input(f"\n[{thread}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/new":
            counter += 1
            thread = f"t{counter}"
            print(f"(new thread {thread})")
            continue
        if user.startswith("/thread "):
            thread = user.split(maxsplit=1)[1]
            print(f"(on thread {thread})")
            continue

        payload: dict | Command = {"messages": [HumanMessage(user)]}
        while True:
            card = run_turn(graph, payload, thread_config(thread))
            if card is None:
                break
            ans = input(f"  ⚠ approval needed: {card['card']}\n  approve? [y/N] ").strip()
            payload = Command(resume="approve" if ans.lower().startswith("y") else "decline")


if __name__ == "__main__":
    main()
