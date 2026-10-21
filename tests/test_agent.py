"""The agent loop against a fake client: scripted responses, no API calls."""

import json
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import BaseModel

from tool_agent import agent
from tool_agent.config import MODEL, VERIFIER_MODEL
from tool_agent.schemas import AgentResult, CostEstimate, FinalAnswer, Verdict

# What estimate_cost returns for this workload on the fixture model (see conftest.py)
WORKLOAD = {
    "model_id": "anthropic/acme-mini",
    "requests_per_day": 10_000,
    "input_tokens": 2000,
    "output_tokens": 500,
}
ESTIMATE = CostEstimate(
    model_id="anthropic/acme-mini",
    monthly_requests=300_000,
    monthly_cost=600.0,
    cost_per_1k_requests=2.0,
    explanation="whatever the model wrote",
)


def tool_use(tool_use_id: str, name: str, **tool_input: Any) -> ToolUseBlock:
    return ToolUseBlock(id=tool_use_id, name=name, input=tool_input, type="tool_use")


def calls_tools(*blocks: ToolUseBlock) -> SimpleNamespace:
    return _response(list(blocks), "tool_use", None)


def answers(answer: FinalAnswer) -> SimpleNamespace:
    return _response([TextBlock(text=answer.model_dump_json(), type="text")], "end_turn", answer)


def verdict(*problems: str) -> SimpleNamespace:
    """The verifier's reply: correct when no problems are given."""
    v = Verdict(correct=not problems, problems=list(problems))
    return _response([TextBlock(text=v.model_dump_json(), type="text")], "end_turn", v)


def _response(content: list[Any], stop_reason: str, parsed: BaseModel | None) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=0, output_tokens=0)
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, usage=usage, parsed_output=parsed
    )


OK = verdict()


class FakeClient:
    """Stands in for anthropic.Anthropic: returns scripted responses in order and
    records the model and messages of each call."""

    def __init__(self, *responses: SimpleNamespace) -> None:
        self._responses = list(responses)
        self.sent: list[list[Any]] = []
        self.models: list[str] = []
        self.messages = self

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.sent.append(list(kwargs["messages"]))  # copy: the loop keeps appending
        self.models.append(kwargs["model"])
        return self._responses.pop(0)


def run(fake: FakeClient, question: str = "What would acme-mini cost?") -> AgentResult:
    return agent.run_agent(question, client=cast(anthropic.Anthropic, fake))


def last_message(fake: FakeClient, call: int) -> Any:
    return fake.sent[call][-1]


def test_runs_tools_and_returns_cited_answer() -> None:
    final = FinalAnswer(
        answer="acme-mini costs $600 a month. Sources: toolu_1, toolu_2.",
        source_tool_use_ids=["toolu_1", "toolu_2"],
        chosen=ESTIMATE,
    )
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "list_models", provider="anthropic")),
        calls_tools(tool_use("toolu_2", "estimate_cost", **WORKLOAD)),
        answers(final),
        OK,
    )

    result = run(fake)
    assert result.answer == final
    assert result.verified
    assert fake.models == [MODEL, MODEL, MODEL, VERIFIER_MODEL]

    # The tool result went back as a user message, tied to the tool_use id
    [tool_result] = last_message(fake, 1)["content"]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["total_matching"] == 2

    # The whole assistant turn stays in the history, not just its text
    assert fake.sent[2][-2]["role"] == "assistant"
    assert fake.sent[2][-2]["content"][0].name == "estimate_cost"


def test_parallel_tool_calls_return_in_one_message() -> None:
    fake = FakeClient(
        calls_tools(
            tool_use("toolu_1", "get_model", model_id="anthropic/acme-mini"),
            tool_use("toolu_2", "get_model", model_id="anthropic/acme-pro"),
        ),
        answers(FinalAnswer(answer="Both found.", source_tool_use_ids=["toolu_1", "toolu_2"])),
        OK,
    )
    run(fake)
    results = last_message(fake, 1)["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]


def test_tool_errors_go_back_to_the_model() -> None:
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "get_model", model_id="anthropic/claude-ultra")),
        calls_tools(tool_use("toolu_2", "list_models", provider="anthropic")),
        answers(FinalAnswer(answer="Here they are.", source_tool_use_ids=["toolu_2"])),
        OK,
    )
    run(fake)
    [tool_result] = last_message(fake, 1)["content"]
    assert tool_result["is_error"] is True
    assert tool_result["content"].startswith("KeyError")


def test_bad_citation_gets_feedback_and_a_retry() -> None:
    # Cites tool names instead of ids: what a real run did before TODO 6
    wrong = FinalAnswer(answer="$600.", source_tool_use_ids=["estimate_cost"], chosen=ESTIMATE)
    fixed = FinalAnswer(answer="$600.", source_tool_use_ids=["toolu_1"], chosen=ESTIMATE)
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "estimate_cost", **WORKLOAD)),
        answers(wrong),  # no verifier call: the free check fails first
        answers(fixed),
        OK,
    )

    result = run(fake)
    assert result.answer == fixed
    assert result.citation_failures == 1
    feedback = last_message(fake, 2)
    assert feedback["role"] == "user"
    assert "'estimate_cost' is not the id" in feedback["content"]
    assert "- toolu_1: estimate_cost {" in feedback["content"]


def test_made_up_numbers_are_rejected() -> None:
    made_up = ESTIMATE.model_copy(update={"monthly_cost": 450.0})
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "estimate_cost", **WORKLOAD)),
        answers(FinalAnswer(answer="$450.", source_tool_use_ids=["toolu_1"], chosen=made_up)),
        answers(FinalAnswer(answer="$600.", source_tool_use_ids=["toolu_1"], chosen=ESTIMATE)),
        OK,
    )
    assert run(fake).answer.chosen == ESTIMATE
    assert "chosen doesn't match" in last_message(fake, 2)["content"]


def test_verifier_sees_only_the_cited_results() -> None:
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "list_models", provider="anthropic")),
        calls_tools(tool_use("toolu_2", "estimate_cost", **WORKLOAD)),
        answers(FinalAnswer(answer="$600.", source_tool_use_ids=["toolu_2"], chosen=ESTIMATE)),
        OK,
    )
    run(fake, question="What would acme-mini cost at 10k a day?")
    [evidence] = fake.sent[3]
    assert fake.models[3] == VERIFIER_MODEL
    assert "What would acme-mini cost at 10k a day?" in evidence["content"]
    assert 'id="toolu_2" tool="estimate_cost"' in evidence["content"]
    assert '"requests_per_day": 10000' in evidence["content"]  # the tool's input, too
    assert "toolu_1" not in evidence["content"]


def test_verifier_rejection_goes_back_and_the_fix_is_verified() -> None:
    wrong_claim = FinalAnswer(answer="acme-mini is the smartest.", source_tool_use_ids=["toolu_1"])
    fixed = FinalAnswer(answer="acme-pro is the smartest.", source_tool_use_ids=["toolu_1"])
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "list_models", sort_by="intelligence")),
        answers(wrong_claim),
        verdict("acme-pro has the higher intelligence index (60 vs 40)"),
        answers(fixed),
        OK,
    )

    result = run(fake)
    assert result.answer == fixed
    assert result.verified
    assert result.verifier_rejections == 1
    feedback = last_message(fake, 3)
    assert feedback["role"] == "user"
    assert "acme-pro has the higher intelligence index" in feedback["content"]
    # Lists the calls so new ones get cited (a real run cited a stale id here)
    assert '- toolu_1: list_models {"sort_by": "intelligence"}' in feedback["content"]


def test_a_missing_model_can_be_answered_from_the_failed_lookup() -> None:
    # The real "claude-ultra" run: the failed get_model call is the evidence
    final = FinalAnswer(
        answer="anthropic/claude-ultra isn't in the catalog. Source: toolu_1.",
        source_tool_use_ids=["toolu_1"],
    )
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "get_model", model_id="anthropic/claude-ultra")),
        answers(final),
        OK,
    )
    result = run(fake)
    assert result.answer == final
    assert result.citation_failures == 0
    [evidence] = fake.sent[2]  # what the verifier saw
    assert 'id="toolu_1" tool="get_model" status="error"' in evidence["content"]
    assert "Unknown model_id" in evidence["content"]


def test_two_rejections_return_the_answer_flagged() -> None:
    claim = FinalAnswer(answer="acme-mini is the smartest.", source_tool_use_ids=["toolu_1"])
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "list_models", sort_by="intelligence")),
        answers(claim),
        verdict("wrong: acme-pro scores higher"),
        answers(claim),
        verdict("still wrong: acme-pro scores higher"),
    )

    result = run(fake)
    assert result.answer == claim
    assert not result.verified
    assert result.concerns == ["still wrong: acme-pro scores higher"]
    assert result.verifier_rejections == 2
    assert len(fake.sent) == 5  # no third attempt


def test_no_verdict_counts_as_a_rejection() -> None:
    claim = FinalAnswer(answer="$600.", source_tool_use_ids=["toolu_1"], chosen=ESTIMATE)
    refused = _response([], "refusal", None)
    fake = FakeClient(
        calls_tools(tool_use("toolu_1", "estimate_cost", **WORKLOAD)),
        answers(claim),
        refused,
        answers(claim),
        refused,
    )
    result = run(fake)
    assert not result.verified
    assert "no verdict (stop_reason=refusal)" in result.concerns[0]


def test_step_guard_stops_a_loop_that_never_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "MAX_STEPS", 3)
    fake = FakeClient(*[calls_tools(tool_use(f"toolu_{i}", "list_models")) for i in range(3)])
    with pytest.raises(RuntimeError, match="did not finish within 3 steps"):
        run(fake)
    assert len(fake.sent) == 3


def test_missing_final_answer_raises() -> None:
    cut_off = _response([TextBlock(text='{"answer": "The cheap', type="text")], "max_tokens", None)
    with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
        run(FakeClient(cut_off))
