"""Stage 2 tools, defined the RAW way: a plain function + @tool. The decorator
derives the JSON schema the model sees from the signature + docstring — print
`shout.args_schema.model_json_schema()` to see exactly what gets sent.

Deliberately missing here (stage 3 adds the contract that fixes all of it):
no timeout, no result-size cap, no argument sanity limits, no retry. Try
"roll a million dice" and watch the agent hang — that's the lesson.
"""

from __future__ import annotations

import random

from langchain_core.tools import tool


@tool
def shout(text: str) -> str:
    """Format text as a shout: uppercase, emphasized. Use when the user asks
    for text to be shouted, yelled, or emphasized. Returns the formatted text."""
    return f"🗣️ {text.upper().rstrip('.!')}!!!"


@tool
def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll `count` dice with `sides` sides each. Use for any dice roll or
    random-number request. Returns the individual rolls and their total."""
    rolls = [random.randint(1, sides) for _ in range(count)]
    return f"rolled {rolls} (total {sum(rolls)})"


TOOLS = [shout, roll_dice]
