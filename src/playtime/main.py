"""Stage 1: the dumbest possible chat REPL — deliberately STATELESS.

Each turn sends [system, this one user message] and nothing else. Ask it your
name two turns after telling it and watch it fail: an LLM API call has no
memory, ever. Everything that *looks* like memory (history, summaries, pinned
facts) is context we choose to assemble and resend each turn. Stage 2 starts
building that machinery.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from playtime.llm import make_llm

SYSTEM_PROMPT = (
    "You are a concise, friendly assistant in a terminal chat. Answer plainly."
)


def main() -> None:
    llm = make_llm("agent")
    print("agent-playtime stage 1 (stateless). Ctrl-D or /quit to exit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(user)])
        served_by = resp.response_metadata.get("model_name", "?")
        print(f"\nbot ({served_by})> {resp.text}")


if __name__ == "__main__":
    main()
