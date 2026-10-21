from tool_agent.agent import citation_problems
from tool_agent.schemas import CostEstimate, FinalAnswer
from tool_agent.verify import ToolCall

ESTIMATE = CostEstimate(
    model_id="anthropic/acme-mini",
    monthly_requests=300_000,
    monthly_cost=600.0,
    cost_per_1k_requests=2.0,
    explanation="2,000 input (0% cached) and 500 output tokens per request",
)

CALLS = {
    "toolu_list": ToolCall("list_models", '{"total_matching": 1, "models": []}', False),
    "toolu_cost": ToolCall("estimate_cost", ESTIMATE.model_dump_json(), False),
    "toolu_bad": ToolCall("get_model", "KeyError: 'Unknown model_id'", True),
}


def _answer(ids: list[str], chosen: CostEstimate | None = None) -> FinalAnswer:
    return FinalAnswer(answer="It costs $600 a month.", source_tool_use_ids=ids, chosen=chosen)


def test_real_ids_and_matching_estimate_pass() -> None:
    assert citation_problems(_answer(["toolu_list", "toolu_cost"], ESTIMATE), CALLS) == []


def test_reworded_explanation_still_matches() -> None:
    reworded = ESTIMATE.model_copy(update={"explanation": "300k requests a month"})
    assert citation_problems(_answer(["toolu_cost"], reworded), CALLS) == []


def test_tool_names_instead_of_ids_fail() -> None:
    # What a real run did before the check existed
    problems = citation_problems(_answer(["list_models", "estimate_cost"]), CALLS)
    assert len(problems) == 2
    assert "'list_models' is not the id" in problems[0]


def test_citing_a_failed_call_is_allowed() -> None:
    # "Unknown model_id" is the evidence that a model isn't in the catalog
    assert citation_problems(_answer(["toolu_bad"]), CALLS) == []


def test_chosen_cannot_come_from_a_failed_call() -> None:
    failed_estimate = {"toolu_x": ToolCall("estimate_cost", "KeyError: 'Unknown'", True)}
    assert citation_problems(_answer(["toolu_x"], ESTIMATE), failed_estimate) != []


def test_chosen_must_match_a_cited_estimate() -> None:
    made_up = ESTIMATE.model_copy(update={"monthly_cost": 450.0})
    assert citation_problems(_answer(["toolu_cost"], made_up), CALLS) == [
        "chosen doesn't match the result of any estimate_cost call you cited"
    ]
    # Right numbers, but the estimate_cost call isn't among the citations
    assert citation_problems(_answer(["toolu_list"], ESTIMATE), CALLS) != []
