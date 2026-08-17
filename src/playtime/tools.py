"""Stage 3 tools: ToolSpecs with explicit pydantic arg models.

Compare to stage 2 (`git diff HEAD -- src/playtime/tools.py` after commit):
the @tool decorator scraped docstrings; now every model-facing string is
written on purpose — descriptions on the spec AND on each field. The Field
constraints (ge/le) are guardrails the model can't argue with: "roll a million
dice" now comes back as INVALID_ARGUMENTS instead of hanging the turn.
"""

from __future__ import annotations

import datetime
import random
from pathlib import Path

from pydantic import BaseModel, Field

from playtime.toolkit import ToolSpec


class ShoutArgs(BaseModel):
    text: str = Field(description="The text to shout.")


def shout(a: ShoutArgs) -> str:
    return f"🗣️ {a.text.upper().rstrip('.!')}!!!"


class RollDiceArgs(BaseModel):
    sides: int = Field(6, ge=2, le=10_000, description="Faces per die.")
    count: int = Field(1, ge=1, le=100, description="How many dice to roll (max 100).")


def roll_dice(a: RollDiceArgs) -> str:
    rolls = [random.randint(1, a.sides) for _ in range(a.count)]
    return f"rolled {rolls} (total {sum(rolls)})"


class PoopArgs(BaseModel):
    pass


def poop(a: PoopArgs) -> str:
    return "💩"


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Subject line.")
    body: str = Field(description="Plain-text body of the email.")


REMEMBER_FACT = "remember_fact"


class RememberFactArgs(BaseModel):
    key: str = Field(description="Short snake_case label, e.g. 'user_budget' or 'deadline'.")
    value: str = Field(description="The fact, verbatim, e.g. '$50 hard cap' or 'Friday 5pm'.")


def remember_fact(a: RememberFactArgs) -> str:  # pragma: no cover
    # Never actually runs: the tools node special-cases this name and writes
    # to graph state instead (a tool can't reach state; the node can).
    return f"Pinned {a.key}."


OUTBOX = Path("var/outbox")


def send_email(a: SendEmailArgs) -> str:
    # Mock "consequential write": drops a file instead of sending. The point is
    # the approval gate around it, not the sending.
    OUTBOX.mkdir(parents=True, exist_ok=True)
    path = OUTBOX / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{a.to}.txt"
    path.write_text(f"To: {a.to}\nSubject: {a.subject}\n\n{a.body}\n")
    return f"email to {a.to} saved to {path} (mock send)"


# Stage 5: shout left the master's menu — it's the stylist sub-agent's private
# tool now (subagents.py imports this spec). The master delegates instead.
SHOUT = ToolSpec(
    name="shout",
    description=(
        "Format text as a shout: uppercase, emphasized. Use when the requested "
        "style calls for shouting or all-caps. Returns the formatted text."
    ),
    args_model=ShoutArgs,
    fn=shout,
)

TOOLS = [
    ToolSpec(
        name="roll_dice",
        description=(
            "Roll `count` dice with `sides` sides each. Use for any dice roll or "
            "random-number request. Returns the individual rolls and their total."
        ),
        args_model=RollDiceArgs,
        fn=roll_dice,
    ),
    ToolSpec(
        name="poop",
        description="Return a poop emoji when something is unclear.",
        args_model=PoopArgs,
        fn=poop,
    ),
    ToolSpec(
        name=REMEMBER_FACT,
        description=(
            "Pin a durable fact (name, budget, date, decision) so it survives long "
            "conversations verbatim. Use immediately when the user establishes one. "
            "Returns confirmation."
        ),
        args_model=RememberFactArgs,
        fn=remember_fact,
    ),
    ToolSpec(
        name="send_email",
        description=(
            "Send an email on the user's behalf. Use only when the user explicitly "
            "asks to send/email something. Returns a delivery confirmation."
        ),
        args_model=SendEmailArgs,
        fn=send_email,
        requires_approval=True,  # consequential + outward-facing → human gate
    ),
]
