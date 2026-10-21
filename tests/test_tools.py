import json
from typing import Any

import pytest
from pydantic import ValidationError

from tool_agent.tools import dispatch, tool_definitions


def _estimate(**overrides: Any) -> dict[str, Any]:
    args = {
        "model_id": "anthropic/acme-mini",
        "requests_per_day": 10_000,
        "input_tokens": 2000,
        "output_tokens": 500,
    } | overrides
    result: dict[str, Any] = json.loads(dispatch("estimate_cost", args))
    return result


def test_definitions_have_schemas() -> None:
    defs = tool_definitions()
    assert {d["name"] for d in defs} == {"list_models", "get_model", "estimate_cost"}
    assert all("properties" in d["input_schema"] for d in defs)


def test_estimate_cost_without_caching() -> None:
    result = _estimate()
    # (2,000 x $0.50 + 500 x $2.00) / 1M = $0.002 a request, x 300,000 requests
    assert result["monthly_cost"] == 600
    assert result["cost_per_1k_requests"] == 2


def test_estimate_cost_with_caching() -> None:
    # Half the input at the cached rate: (1,000 x $0.50 + 1,000 x $0.05 + 500 x $2.00) / 1M
    assert _estimate(cached_fraction=0.5)["monthly_cost"] == 465


def test_cache_falls_back_to_full_rate_without_cache_price() -> None:
    no_cache = {"model_id": "openai/globex-lite"}
    assert (
        _estimate(**no_cache, cached_fraction=1)["monthly_cost"]
        == _estimate(**no_cache)["monthly_cost"]
    )


def test_list_models_filters_and_sorts() -> None:
    result = json.loads(dispatch("list_models", {"provider": "anthropic"}))
    assert [m["model_id"] for m in result["models"]] == [
        "anthropic/acme-mini",
        "anthropic/acme-pro",
    ]

    images = json.loads(dispatch("list_models", {"needs_image_input": True}))
    assert [m["model_id"] for m in images["models"]] == ["anthropic/acme-mini"]

    smart = json.loads(dispatch("list_models", {"sort_by": "intelligence", "limit": 1}))
    assert smart["total_matching"] == 3
    assert [m["model_id"] for m in smart["models"]] == ["anthropic/acme-pro"]

    tools = json.loads(dispatch("list_models", {"needs_tools": True, "min_context": 500_000}))
    assert [m["model_id"] for m in tools["models"]] == ["anthropic/acme-pro"]


def test_get_model_returns_description() -> None:
    result = json.loads(dispatch("get_model", {"model_id": "anthropic/acme-pro"}))
    assert result["description"] == "Fictional test model."


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        dispatch("get_model", {"model_id": "anthropic/acme-ultra"})


def test_bad_input_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        _estimate(requests_per_day=0)
    with pytest.raises(ValidationError):
        dispatch("list_models", {"provider": "globex-corp"})
