"""Multi-turn chat against the fake client: history carries over, failed turns roll back."""

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest

from tests.test_agent import ESTIMATE, OK, WORKLOAD, FakeClient, answers, calls_tools, tool_use
from tool_agent import agent, chat
from tool_agent.agent import Conversation, run_turn
from tool_agent.schemas import FinalAnswer


class FlakyClient(FakeClient):
    """A FakeClient whose script can include exceptions, raised in order."""

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        response = super().parse(**kwargs)
        if isinstance(response, BaseException):
            raise response
        return response


def as_client(fake: FakeClient) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, fake)


def test_second_turn_sees_the_first_and_can_cite_its_calls() -> None:
    first = FinalAnswer(answer="acme models: ... Source: toolu_1.", source_tool_use_ids=["toolu_1"])
    # The follow-up cites toolu_1 from turn 1 as well as its own estimate
    second = FinalAnswer(
        answer="The cheapest of those costs $600 a month.",
        source_tool_use_ids=["toolu_1", "toolu_2"],
        chosen=ESTIMATE,
    )
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "list_models", provider="anthropic")),
        answers(first),
        OK,
        calls_tools(tool_use("toolu_2", "estimate_cost", **WORKLOAD)),
        answers(second),
        OK,
    )
    conversation = Conversation()

    assert run_turn(conversation, "Which Anthropic models are there?", as_client(fake)).verified
    result = run_turn(conversation, "What would the cheapest of those cost?", as_client(fake))

    assert result.answer == second
    assert result.citation_failures == 0  # turn 1's call is still a valid citation
    # Turn 2's first request carried turn 1's whole exchange
    turn2_first_request = fake.sent[3]
    assert turn2_first_request[0] == {
        "role": "user",
        "content": "Which Anthropic models are there?",
    }
    assert turn2_first_request[-1] == {
        "role": "user",
        "content": "What would the cheapest of those cost?",
    }
    # The verifier gets the earlier question, so "those" makes sense
    [evidence] = fake.sent[5]
    assert "- Which Anthropic models are there?" in evidence["content"]
    assert conversation.questions == [
        "Which Anthropic models are there?",
        "What would the cheapest of those cost?",
    ]


def test_failed_turn_rolls_back_and_the_next_one_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "MAX_STEPS", 1)
    good = FinalAnswer(answer="Found it.", source_tool_use_ids=["toolu_2"])
    fake = FakeClient(
        # Turn 1 hits the step guard after a tool call: history would end on a tool_result
        calls_tools(tool_use("toolu_1", "list_models")),
        # Turn 2 answers in one step
        calls_tools(tool_use("toolu_2", "get_model", model_id="anthropic/acme-mini")),
    )
    conversation = Conversation()

    with pytest.raises(RuntimeError, match="did not finish"):
        run_turn(conversation, "First question", as_client(fake))
    assert conversation.messages == []
    assert conversation.tool_calls == {}
    assert conversation.questions == []

    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    fake._responses.extend([answers(good), OK])
    assert run_turn(conversation, "Second question", as_client(fake)).answer == good
    assert conversation.messages[0] == {"role": "user", "content": "Second question"}


def test_api_error_mid_turn_rolls_back() -> None:
    error = anthropic.APIConnectionError(request=cast(Any, None))
    fake = FlakyClient(calls_tools(tool_use("toolu_1", "list_models")), error)  # type: ignore[arg-type]
    conversation = Conversation()
    with pytest.raises(anthropic.APIConnectionError):
        run_turn(conversation, "Question", as_client(fake))
    assert conversation.messages == []  # not left ending on a tool_use with no result


def test_chat_loop_answers_survives_a_failure_and_quits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["", "Break please", "What would acme-mini cost?", "/quit", "never read"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(lines))
    fake = FlakyClient(
        RuntimeError("model went away"),  # type: ignore[arg-type]
        calls_tools(tool_use("toolu_1", "estimate_cost", **WORKLOAD)),
        answers(
            FinalAnswer(answer="$600 a month.", source_tool_use_ids=["toolu_1"], chosen=ESTIMATE)
        ),
        OK,
    )

    chat.chat(as_client(fake))

    out = capsys.readouterr().out
    assert "That didn't work: model went away" in out
    assert "$600 a month." in out
    assert "Estimate: anthropic/acme-mini, $600.00 a month ($2.00 per 1,000 requests)" in out
    assert "Sources: toolu_1" in out
    assert "WARNING" not in out
    assert next(lines) == "never read"  # stopped at /quit


def test_chat_quits_on_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    chat.chat(as_client(FakeClient()))  # returns instead of hanging or crashing


def test_unverified_answer_shows_the_concerns() -> None:
    from tool_agent.schemas import AgentResult

    result = AgentResult(
        answer=FinalAnswer(answer="Maybe.", source_tool_use_ids=["toolu_1"]),
        verified=False,
        concerns=["The cited result doesn't show that."],
        citation_failures=0,
        verifier_rejections=2,
    )
    text = chat.format_result(result)
    assert "WARNING: the verifier did not accept this answer" in text
    assert "- The cited result doesn't show that." in text
