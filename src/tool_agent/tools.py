"""Tools the agent can call, their API definitions, and a dispatcher. Data comes from db.py."""

from collections.abc import Callable
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel

from tool_agent import db
from tool_agent.schemas import (
    CostEstimate,
    EstimateCostInput,
    GetModelInput,
    ListModelsInput,
    ModelInfo,
    ModelList,
    ModelSummary,
)

DAYS_PER_MONTH = 30

# Fixed SQL for each sort option; the model only ever picks a key.
ORDER_BY = {
    "input_price": "input_per_mtok ASC, model_id",
    "intelligence": "intelligence_index IS NULL, intelligence_index DESC, model_id",
    "context": "context_length DESC, model_id",
}


def get_model(args: GetModelInput) -> ModelInfo:
    model = db.get_model(args.model_id)
    if model is None:
        raise KeyError(f"Unknown model_id {args.model_id!r}. Use list_models to find valid ids.")
    return model


def list_models(args: ListModelsInput) -> ModelList:
    matches = db.find_models(
        provider=args.provider,
        min_context=args.min_context,
        needs_tools=args.needs_tools,
        needs_reasoning=args.needs_reasoning,
        needs_image_input=args.needs_image_input,
        max_input_per_mtok=args.max_input_per_mtok,
        order_by=ORDER_BY[args.sort_by],
    )
    return ModelList(
        total_matching=len(matches),
        models=[ModelSummary.model_validate(m.model_dump()) for m in matches[: args.limit]],
    )


def estimate_cost(args: EstimateCostInput) -> CostEstimate:
    """Token cost per request, times requests per month.

    Cached input tokens bill at the cached rate (or the full rate when a model has no cache
    price). Simplified: ignores long-prompt price tiers and cache-write costs.
    """
    m = get_model(GetModelInput(model_id=args.model_id))
    cached_rate = (
        m.cached_input_per_mtok if m.cached_input_per_mtok is not None else m.input_per_mtok
    )
    cached_tokens = args.input_tokens * args.cached_fraction
    uncached_tokens = args.input_tokens - cached_tokens
    per_request = (
        uncached_tokens * m.input_per_mtok
        + cached_tokens * cached_rate
        + args.output_tokens * m.output_per_mtok
    ) / 1_000_000
    monthly_requests = args.requests_per_day * DAYS_PER_MONTH
    return CostEstimate(
        model_id=m.model_id,
        monthly_requests=monthly_requests,
        monthly_cost=round(per_request * monthly_requests, 2),
        cost_per_1k_requests=round(per_request * 1000, 2),
        explanation=(
            f"{args.input_tokens:,} input ({args.cached_fraction:.0%} cached) and "
            f"{args.output_tokens:,} output tokens per request, "
            f"{monthly_requests:,} requests a month"
        ),
    )


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        fn: Callable[[Any], BaseModel],
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.fn = fn

    def definition(self) -> ToolParam:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "list_models",
            "Search the model catalog (Anthropic, OpenAI and xAI models, prices from "
            "OpenRouter). Filter by provider, context window, tool use, reasoning, image "
            "input and input price; sort by price, intelligence or context.",
            ListModelsInput,
            list_models,
        ),
        Tool(
            "get_model",
            "Get one model's full details: description, limits, capabilities and prices.",
            GetModelInput,
            get_model,
        ),
        Tool(
            "estimate_cost",
            "Estimate the monthly cost and cost per 1,000 requests of running a workload "
            "on one model.",
            EstimateCostInput,
            estimate_cost,
        ),
    ]
}


def tool_definitions() -> list[ToolParam]:
    return [t.definition() for t in TOOLS.values()]


def dispatch(name: str, raw_input: dict[str, Any]) -> str:
    """Validate the model's input, run the tool, return JSON for a tool_result block.

    Raises KeyError for unknown tools or models and pydantic.ValidationError for bad input;
    the agent loop decides how to report those back to the model.
    """
    tool = TOOLS[name]
    args = tool.input_model.model_validate(raw_input)
    return tool.fn(args).model_dump_json()
