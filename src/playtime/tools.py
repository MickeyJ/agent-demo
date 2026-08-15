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


OUTBOX = Path("var/outbox")


def send_email(a: SendEmailArgs) -> str:
    # Mock "consequential write": drops a file instead of sending. The point is
    # the approval gate around it, not the sending.
    OUTBOX.mkdir(parents=True, exist_ok=True)
    path = OUTBOX / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{a.to}.txt"
    path.write_text(f"To: {a.to}\nSubject: {a.subject}\n\n{a.body}\n")
    return f"email to {a.to} saved to {path} (mock send)"


TOOLS = [
    ToolSpec(
        name="shout",
        description=(
            "Format text as a shout: uppercase, emphasized. Use when the user asks "
            "for text to be shouted, yelled, or emphasized. Returns the formatted text."
        ),
        args_model=ShoutArgs,
        fn=shout,
    ),
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
