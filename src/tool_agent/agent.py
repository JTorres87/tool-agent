import anthropic

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

    for step in range(MAX_STEPS):
        # TODO 1 (done): one model call with the whole conversation so far
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        # TODO 2 (done): keep every block (thinking, text, tool_use) in the history
        messages.append({"role": "assistant", "content": response.content})

        # TEMPORARY: show what came back, then stop. Delete once you start TODO 3/4.
        print(f"--- step {step}: stop_reason={response.stop_reason}")
        print(f"    usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        for block in response.content:
            if block.type == "tool_use":
                print(f"    tool_use  id={block.id} name={block.name} input={block.input}")
            elif block.type == "text":
                print(f"    text      {block.text!r}")
            elif block.type == "thinking":
                print(f"    thinking  ({len(block.thinking)} chars)")
            else:
                print(f"    {block.type}")
        raise NotImplementedError("Stopped after TODO 2. Next: TODO 3 and 4.")

        # TODO 3: if stop_reason != "tool_use", parse the final answer into a
        #         FinalAnswer (structured outputs is the safe choice). Careful:
        #         FALLBACK_MODEL (Sonnet 5.5) rejects forced tool_choice "any"/"tool"
        #         with a 400, so a forced "answer" tool breaks the fallback.
        # TODO 4: for every tool_use block, call dispatch(block.name, block.input)
        #         and collect a tool_result block (is_error=True when it raises).
        #         Keep each result by block.id; TODO 6 needs them.
        # TODO 5: append ALL tool_result blocks in ONE user message
        # TODO 6: before returning, check the citations: every source_tool_use_id
        #         must be a call you ran, and chosen (when set) must match what
        #         that estimate_cost call returned. If not, tell the model what's
        #         wrong and let it try again.
        _ = dispatch  # used in TODO 4

    raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")
