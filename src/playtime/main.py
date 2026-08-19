"""Stage 4 REPL: context made visible.

/debug prints exactly what the model would see next turn — the layered system
message (prompt + pinned + summary) and the trimmed window with token counts.
A ✂ line appears whenever prepare_context folds old turns into the summary.
Checkpoints live in var/checkpoints.sqlite now: restart and /thread back to
any conversation.

(Approval flow unchanged from stage 3: interrupt() pauses, y/N resumes.)
"""

from __future__ import annotations

import readline  # noqa: F401 — side-effect import: gives input() arrow keys + history

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from playtime.context import count_message_tokens
from playtime.graph import assemble_system, build_graph, stamp, thread_config
from playtime.settings import get_settings

HELP = "commands: /debug (context state) | /new (fresh thread) | /thread <name> (switch) | /quit"


def debug(graph, config) -> None:
    """The whole point of stage 4 made visible: what would the model see next
    turn, and how full is the budget?"""
    vals = graph.get_state(config).values
    if not vals:
        print("(empty thread)")
        return
    msgs = vals.get("messages", [])
    budget = get_settings().ctx_window_budget
    sys_msg = assemble_system(vals)
    print(f"-- system message (~{count_message_tokens([sys_msg])} tokens) --")
    print(sys_msg.text)
    print(f"-- window: {len(msgs)} messages, ~{count_message_tokens(msgs)}/{budget} tokens --")
    for m in msgs:
        kind = type(m).__name__.removesuffix("Message")
        print(f"  {kind:6} ~{count_message_tokens([m]):4}  {m.text[:70]!r}")


def run_turn(graph, payload, config) -> dict | None:
    """Stream one graph run; print progress. Returns the approval card if the
    run paused on an interrupt, None if it ran to completion."""
    card = None
    for step in graph.stream(payload, config, stream_mode="updates"):
        for node, update in step.items():
            if node == "__interrupt__":
                card = update[0].value
                continue
            if not update:  # nodes may return no visible output
                continue
            if node == "prepare_context":
                if "summary" in update:
                    n = len(update.get("messages", []))
                    print(f"  ✂ folded {n} old messages into the summary")
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
    print(f"agent-playtime stage 6 (mini RAG). {HELP}")
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
        if user == "/debug":
            debug(graph, thread_config(thread))
            continue
        if user == "/new":
            counter += 1
            thread = f"t{counter}"
            print(f"(new thread {thread})")
            continue
        if user.startswith("/thread "):
            thread = user.split(maxsplit=1)[1]
            print(f"(on thread {thread})")
            continue

        payload: dict | Command = {"messages": [stamp(HumanMessage(user))]}
        while True:
            card = run_turn(graph, payload, thread_config(thread))
            if card is None:
                break
            ans = input(f"  ⚠ approval needed: {card['card']}\n  approve? [y/N] ").strip()
            payload = Command(resume="approve" if ans.lower().startswith("y") else "decline")


if __name__ == "__main__":
    main()
