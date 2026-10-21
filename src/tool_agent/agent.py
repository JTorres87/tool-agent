import sys
from typing import Any, cast

import anthropic
from anthropic.types import ToolResultBlockParam
from pydantic import ValidationError

from tool_agent.config import MAX_STEPS, MAX_TOKENS, MODEL
from tool_agent.schemas import FinalAnswer
from tool_agent.tools import dispatch, tool_definitions

SYSTEM = (
    "You help teams choose an LLM and plan what it will cost to run. "
    "Use the tools to search the model catalog, read model details and estimate costs. "
    "Never guess prices or capabilities, and never do the math yourself. "
    "Cite the tool calls your answer is based on."
)


def run_agent(question: str, client: anthropic.Anthropic | None = None) -> FinalAnswer:
    client = client or anthropic.Anthropic()
    tools = tool_definitions()
    messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": question}]
    tool_outputs: dict[str, str] = {}  # tool_use_id -> what the tool returned (for TODO 6)

    for step in range(MAX_STEPS):
        # TODO 1 (done): one model call with the whole conversation so far.
        # parse() is create() plus structured output: when Claude gives its final
        # answer, it must match FinalAnswer, and the SDK validates it for us.
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=tools,
            messages=messages,
            output_format=FinalAnswer,
        )
        # TODO 2 (done): keep every block (thinking, text, tool_use) in the history
        messages.append({"role": "assistant", "content": response.content})
        _trace(step, response)

        # TODO 3 (done): no tool call means this is the final answer
        if response.stop_reason != "tool_use":
            if response.parsed_output is None:
                raise RuntimeError(f"No final answer (stop_reason={response.stop_reason})")
            # TODO 6: check the citations here before returning: every
            #         source_tool_use_id must be in tool_outputs, and chosen (when
            #         set) must match that estimate_cost call's output. If not,
            #         tell the model what's wrong and `continue` so it tries again.
            return response.parsed_output

        # TODO 4 (done): run every tool call; errors go back to Claude, not up the stack
        results: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = dispatch(block.name, cast(dict[str, Any], block.input))
                is_error = False
            except (KeyError, ValidationError) as e:
                output = f"{type(e).__name__}: {e}"
                is_error = True
            tool_outputs[block.id] = output
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )

        # TODO 5 (done): ALL results in ONE user message, so parallel calls stay parallel
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")


def _trace(step: int, response: anthropic.types.Message) -> None:
    """TEMPORARY: print each step to stderr. Section 3 replaces this with real logging."""
    out = sys.stderr
    print(f"--- step {step}: stop_reason={response.stop_reason}", file=out)
    print(
        f"    usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out", file=out
    )
    for block in response.content:
        if block.type == "tool_use":
            print(f"    tool_use  id={block.id} name={block.name} input={block.input}", file=out)
        elif block.type == "text":
            print(f"    text      {block.text[:200]!r}", file=out)
        elif block.type == "thinking":
            print(f"    thinking  ({len(block.thinking)} chars)", file=out)
        else:
            print(f"    {block.type}", file=out)
