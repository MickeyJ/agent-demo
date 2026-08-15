"""Stage 2 REPL: the graph drives the conversation now.

Memory arrived, but notice WHERE: not in the model, not in this file — in the
checkpointer, keyed by thread id. `/new` proves isolation (fresh thread knows
nothing), `/thread <name>` proves resumption (the old thread still knows you).
Restart the process and everything is gone: MemorySaver is in-process only.

stream_mode="updates" yields each node's output as it lands, so you watch the
ReAct loop in real time:  ⚙ tool call  →  tool result  →  bot answer.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from playtime.graph import build_graph, thread_config

HELP = "commands: /new (fresh thread) | /thread <name> (switch) | /quit"


def main() -> None:
    graph = build_graph()
    thread = "t1"
    counter = 1
    print(f"agent-playtime stage 2 (ReAct loop). {HELP}")
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

        stream = graph.stream(
            {"messages": [HumanMessage(user)]}, thread_config(thread), stream_mode="updates"
        )
        for step in stream:
            for update in step.values():
                for m in update["messages"]:
                    if isinstance(m, AIMessage):
                        for c in m.tool_calls:
                            print(f"  ⚙ {c['name']}({c['args']})")
                        if m.text:
                            print(f"\nbot> {m.text}")
                    elif isinstance(m, ToolMessage):
                        print(f"    → {m.text}")


if __name__ == "__main__":
    main()
