import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, ToolResultBlockParam
from pydantic import ValidationError

from tool_agent.config import MAX_STEPS, MAX_TOKENS, MAX_VERIFIER_REJECTIONS, MODEL
from tool_agent.schemas import AgentResult, CostEstimate, FinalAnswer
from tool_agent.tools import dispatch, tool_definitions
from tool_agent.verify import ToolCall, verify

log = logging.getLogger(__name__)

SYSTEM = (
    "You help teams choose an LLM and plan what it will cost to run. "
    "Use the tools to search the model catalog, read model details and estimate costs. "
    "Never guess prices or capabilities, and never do the math yourself. "
    "Cite the tool calls your answer is based on."
)


@dataclass
class Conversation:
    """Everything that carries over between turns of a chat."""

    messages: list[MessageParam] = field(default_factory=list)
    # Every tool call so far, so an answer can cite results from earlier turns
    tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)


def run_agent(question: str, client: anthropic.Anthropic | None = None) -> AgentResult:
    """One question, one answer: a conversation that lasts a single turn."""
    return run_turn(Conversation(), question, client or anthropic.Anthropic())


def run_turn(conversation: Conversation, question: str, client: anthropic.Anthropic) -> AgentResult:
    """Add the question to the conversation and run the loop until there's an answer.

    The history keeps growing, so the model sees earlier questions, tool calls and
    answers. If the turn fails partway (API error, step guard, Ctrl-C), it's undone:
    a half-finished turn can end on a tool_use with no tool_result, and the API
    rejects every later request with that history. Rolling back keeps the session
    usable; the user just asks again.
    """
    saved_messages = len(conversation.messages)
    saved_calls = dict(conversation.tool_calls)
    conversation.messages.append({"role": "user", "content": question})
    try:
        result = _loop(conversation, question, client)
    except BaseException:
        del conversation.messages[saved_messages:]
        conversation.tool_calls = saved_calls
        raise
    conversation.questions.append(question)
    return result


def _loop(conversation: Conversation, question: str, client: anthropic.Anthropic) -> AgentResult:
    tools = tool_definitions()
    messages = conversation.messages
    tool_calls = conversation.tool_calls  # tool_use_id -> the call, for the citation check
    citation_failures = 0
    verifier_rejections = 0

    for step in range(MAX_STEPS):
        # One model call with the whole conversation so far.
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
        # Keep every block (thinking, text, tool_use) in the history
        messages.append({"role": "assistant", "content": response.content})
        _log_step(step, response)

        # No tool call means this is the final answer
        if response.stop_reason != "tool_use":
            answer = response.parsed_output
            if answer is None:
                raise RuntimeError(f"No final answer (stop_reason={response.stop_reason})")

            # Citation check (free): only return an answer whose citations check
            # out; otherwise say what's wrong and let Claude try again (the step
            # guard stops it from retrying forever)
            problems = citation_problems(answer, tool_calls)
            if problems:
                citation_failures += 1
                log.info("citation check failed (%d): %s", citation_failures, problems)
                messages.append(
                    {"role": "user", "content": _citation_feedback(problems, tool_calls)}
                )
                continue

            # Verifier: a stronger model checks the answer's claims against the cited
            # results. The citation check above is free; this costs one call per answer.
            verdict = verify(
                _with_earlier_questions(question, conversation.questions),
                answer,
                tool_calls,
                client,
            )
            if verdict.correct:
                log.info("verifier: correct")
                return AgentResult(
                    answer=answer,
                    verified=True,
                    concerns=[],
                    citation_failures=citation_failures,
                    verifier_rejections=verifier_rejections,
                )
            verifier_rejections += 1
            log.info("verifier rejected (%d): %s", verifier_rejections, verdict.problems)
            if verifier_rejections >= MAX_VERIFIER_REJECTIONS:
                # Don't let a wrong or picky verifier burn the step budget: hand back
                # the answer with its concerns and let the caller decide.
                return AgentResult(
                    answer=answer,
                    verified=False,
                    concerns=verdict.problems,
                    citation_failures=citation_failures,
                    verifier_rejections=verifier_rejections,
                )
            messages.append(
                {"role": "user", "content": _verifier_feedback(verdict.problems, tool_calls)}
            )
            continue

        # Run every tool call; errors go back to Claude, not up the stack
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
            tool_calls[block.id] = ToolCall(
                block.name, output, is_error, cast(dict[str, Any], block.input)
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    # The id goes in the text too: the model can't see tool_use ids
                    # otherwise, so it can't cite them on the first try.
                    "content": f"[tool_use_id: {block.id}]\n{output}",
                    "is_error": is_error,
                }
            )

        # ALL results in ONE user message, so parallel calls stay parallel
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")


def _with_earlier_questions(question: str, earlier: list[str]) -> str:
    """A follow-up like "the cheapest of those" needs the earlier questions to make sense."""
    if not earlier:
        return question
    history = "\n".join(f"- {q}" for q in earlier)
    return f"Earlier questions in this conversation:\n{history}\n\nCurrent question: {question}"


def citation_problems(answer: FinalAnswer, calls: dict[str, ToolCall]) -> list[str]:
    """Everything wrong with the answer's citations; an empty list means they check out.

    Failed calls may be cited: "Unknown model_id" is the evidence that a model doesn't
    exist. But chosen must match a successful estimate_cost result.
    """
    problems = [
        f"{tool_use_id!r} is not the id of any tool call you made"
        for tool_use_id in answer.source_tool_use_ids
        if tool_use_id not in calls
    ]

    if answer.chosen is not None:
        cited_estimates = [
            CostEstimate.model_validate_json(calls[i].output)
            for i in answer.source_tool_use_ids
            if i in calls and calls[i].name == "estimate_cost" and not calls[i].is_error
        ]
        if not any(_same_estimate(answer.chosen, e) for e in cited_estimates):
            problems.append("chosen doesn't match the result of any estimate_cost call you cited")
    return problems


def _same_estimate(a: CostEstimate, b: CostEstimate) -> bool:
    """Compare the numbers, not the explanation text, which the model may reword."""
    fields = ("model_id", "monthly_requests", "monthly_cost", "cost_per_1k_requests")
    return all(getattr(a, f) == getattr(b, f) for f in fields)


def _call_list(calls: dict[str, ToolCall]) -> str:
    """Every call so far, one per line, so the model can cite the right one."""
    if not calls:
        return "(no tool calls yet)"
    return "\n".join(
        f"- {i}: {c.name} {json.dumps(c.input)}{' (failed)' if c.is_error else ''}"
        for i, c in calls.items()
    )


def _citation_feedback(problems: list[str], calls: dict[str, ToolCall]) -> str:
    return (
        "Your answer failed the citation check:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\nCite tool calls by their tool_use id. Your tool calls so far:\n"
        + _call_list(calls)
        + "\nReply with a corrected final answer."
    )


def _verifier_feedback(problems: list[str], calls: dict[str, ToolCall]) -> str:
    return (
        "A reviewer checked your answer against the tool results you cited and found:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\nFix these, calling tools again if you need more evidence. Your tool calls "
        "so far are below; source_tool_use_ids must list every call the corrected answer "
        "relies on, including any new ones.\n"
        + _call_list(calls)
        + "\nReply with a corrected final answer."
    )


def _log_step(step: int, response: anthropic.types.Message) -> None:
    """Each model step at DEBUG level: run the CLI with -vv to see them."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    u = response.usage
    log.debug(
        "step %d: stop_reason=%s, %d in / %d out tokens",
        step,
        response.stop_reason,
        u.input_tokens,
        u.output_tokens,
    )
    for block in response.content:
        if block.type == "tool_use":
            log.debug("  tool_use %s %s %s", block.id, block.name, block.input)
        elif block.type == "text":
            log.debug("  text %r", block.text[:200])
        else:
            log.debug("  %s", block.type)
