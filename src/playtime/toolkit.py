"""Stage 3: the uniform tool contract.

Every tool is a ToolSpec, and execute() is the ONLY way one runs. The wrapper
owns the guardrails (validation, timeout, error capture, result cap) so the
tool functions themselves stay dumb: args in, string out.

Guardrail rule #1, used three times below: NEVER raise into the graph — return
the failure AS the tool result. The model reads it next loop and self-corrects
(retries with fixed args, apologizes, tries another tool). A raised exception
would kill the whole turn instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

# Tools run here so we can bound their runtime; a timed-out future's thread
# does keep running to completion in the background (Python can't kill it),
# but the AGENT stops waiting — which is the guarantee that matters.
_pool = ThreadPoolExecutor(max_workers=4)


@dataclass
class ToolSpec:
    name: str
    description: str  # ≤3 sentences: what it does, when to use it, what it returns
    args_model: type[BaseModel]  # explicit model — field descriptions ARE the prompt
    fn: Callable[[BaseModel], str]
    requires_approval: bool = False  # True → graph pauses on a human approval card
    timeout_s: float = 10.0
    result_cap_chars: int = 2000  # stage 4 upgrades hard truncation to LLM compaction

    def openai_schema(self) -> dict:
        """The exact JSON the model sees. Built here, on purpose, instead of
        scraped from a docstring by @tool — the model-facing text is deliberate."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def execute(spec: ToolSpec, args: dict) -> str:
    """Validate → run with timeout → cap. Every failure mode becomes a string."""
    try:
        parsed = spec.args_model.model_validate(args or {})
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return f"INVALID_ARGUMENTS: {errors}. Re-call {spec.name} with corrected arguments."
    try:
        raw = _pool.submit(spec.fn, parsed).result(timeout=spec.timeout_s)
    except FutureTimeout:
        log.warning("tool %s timed out after %.1fs", spec.name, spec.timeout_s)
        return f"ERROR: {spec.name} timed out after {spec.timeout_s}s. Tell the user it is unavailable right now."
    except Exception as e:
        log.warning("tool %s raised: %s", spec.name, e)
        return f"ERROR: {spec.name} failed: {e}"
    if len(raw) > spec.result_cap_chars:
        raw = raw[: spec.result_cap_chars] + f"\n...[truncated at {spec.result_cap_chars} chars]"
    return raw
