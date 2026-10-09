"""One forced-tool Claude call that returns a validated Pydantic model."""

from __future__ import annotations

from json import dumps
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from app.services.bot_trace import record_trace
from app.services.pii_redactor import redact_for_model
from app.services.rate_ceiling import charge_anthropic_call

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

# One extra attempt when the model's output fails validation; a second miss raises. Strict tool
# use was measured and rejected: it added about a second per call and a first-use grammar
# compile of ~18 s per schema, worse than tolerant optional fields plus one retry.
VALIDATION_RETRIES = 1


async def structured_call[T: BaseModel](
    client: AsyncAnthropic,
    *,
    name: str,
    schema: type[T],
    rules: str,
    payload: dict,
    model: str,
    max_tokens: int,
) -> T:
    """Force one tool call, validate its input, and raise on anything incomplete.

    The caller decides what a failure means; nothing here invents a result.
    """
    tool = {
        "name": name,
        "description": "Return the structured decision.",
        "input_schema": schema.model_json_schema(),
    }
    last_error: ValueError | None = None
    for attempt in range(1 + VALIDATION_RETRIES):
        if attempt:
            await charge_anthropic_call()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=rules,
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
            messages=[{"role": "user", "content": dumps(payload, ensure_ascii=False)}],
        )
        if response.stop_reason != "tool_use":
            record_trace(name, stop_reason=response.stop_reason, attempt=attempt)
            raise ValueError(f"incomplete_{name}_decision")
        block = next((b for b in response.content if b.type == "tool_use" and b.name == name), None)
        if block is None:
            raise ValueError(f"missing_{name}_decision")
        try:
            result = schema.model_validate(block.input)
        except ValidationError as exc:
            last_error = exc
            record_trace(
                name,
                error_class=type(exc).__name__,
                error=redact_for_model(str(exc))[:1200],
                attempt=attempt,
            )
            continue
        record_trace(name, result=result.model_dump(), attempts=attempt + 1)
        return result
    assert last_error is not None
    raise last_error
