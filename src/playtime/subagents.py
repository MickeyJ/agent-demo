"""Stage 5: sub-agents as tools.

A sub-agent is just a tool whose implementation happens to be another LLM.
The master can't tell — it sees a name, a description, an args schema, and a
string result, exactly like roll_dice. That uniformity is the whole pattern
(in a production system the sub-agent can just as well live in another
PROCESS behind HTTP; same contract).

What makes it interesting:

- CONTEXT ISOLATION. The sub-agent sees ONLY its args — not the conversation.
  If the master doesn't pass something, the sub-agent cannot know it. (Watch
  it fail: establish a nickname in chat, then ask for "a haiku about me" —
  the stylist gets whatever text the master chose to send, nothing more.)
- PRIVATE TOOLS. The stylist owns `shout` now; the master doesn't have it
  anymore. Sub-agents shrink the master's tool menu — with 30 tools a small
  model picks badly, so you group them behind specialists.
- AN AGENT IS A LOOP, NOT A FRAMEWORK. The stylist's ReAct loop is ~10 lines
  of plain Python below — no LangGraph. The framework buys checkpointing and
  interrupts, which a fire-and-forget sub-agent doesn't need.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from playtime.llm import make_llm
from playtime.toolkit import ToolSpec, execute
from playtime.tools import SHOUT

STYLIST_SYSTEM = (
    "You are the Stylist, a text-rewriting specialist. Rewrite the given TEXT "
    "in the given STYLE. Preserve the meaning; change only the voice/form. "
    "You have a `shout` tool — use it when the style calls for shouting or "
    "all-caps emphasis. Output ONLY the rewritten text, no commentary."
)

CRITIC_SYSTEM = (
    "You are the Critic, a ruthless but constructive editor. Critique the given "
    "text in at most 3 short bullets (clarity, concision, impact), then a final "
    "line 'VERDICT: KEEP' or 'VERDICT: REVISE'. No rewriting, no pleasantries."
)


class StylistArgs(BaseModel):
    text: str = Field(description="The text to rewrite, verbatim and complete.")
    style: str = Field(description="Target style, e.g. 'pirate', 'formal', 'haiku', 'shouted'.")


def stylist(a: StylistArgs) -> str:
    # A complete ReAct agent in plain Python: its own system prompt, its own
    # message list (fresh context!), its own tool, its own little loop budget.
    llm = make_llm("agent", temperature=0.7).bind_tools([SHOUT.openai_schema()])
    messages: list = [
        SystemMessage(STYLIST_SYSTEM),
        HumanMessage(f"STYLE: {a.style}\nTEXT: {a.text}"),
    ]
    for _ in range(3):
        resp = llm.invoke(messages)
        if not resp.tool_calls:
            return resp.text.strip()
        messages.append(resp)
        for call in resp.tool_calls:
            messages.append(
                ToolMessage(
                    content=execute(SHOUT, call["args"]),  # same guarded path
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
    return resp.text.strip() or "(stylist hit its loop budget with no answer)"


class CriticArgs(BaseModel):
    text: str = Field(description="The text to critique, verbatim and complete.")


def critic(a: CriticArgs) -> str:
    resp = make_llm("agent", temperature=0.2).invoke(
        [SystemMessage(CRITIC_SYSTEM), HumanMessage(a.text)]
    )
    return resp.text.strip()


SUBAGENTS = [
    ToolSpec(
        name="stylist",
        description=(
            "Rewrite text in a target style (pirate, formal, haiku, shouted, ...). "
            "Use for ANY rewrite, restyle, or shout/emphasis request — include the "
            "full text to rewrite. Returns only the rewritten text."
        ),
        args_model=StylistArgs,
        fn=stylist,
        timeout_s=60.0,  # sub-agents make multiple LLM calls; 10s would misfire
        result_cap_chars=4000,
    ),
    ToolSpec(
        name="critic",
        description=(
            "Get a short editorial critique of a piece of text (3 bullets + "
            "KEEP/REVISE verdict). Use when the user asks for feedback, review, "
            "or judgment on text — include the full text. Returns the critique."
        ),
        args_model=CriticArgs,
        fn=critic,
        timeout_s=60.0,
    ),
]
