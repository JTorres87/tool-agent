from pathlib import Path
from typing import Any

import pytest

from tool_agent import db
from tool_agent.seed import parse_models


def _raw(model_id: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "description": "A model.",
        "context_length": 1_000_000,
        "architecture": {"input_modalities": ["text", "image"]},
        "top_provider": {"max_completion_tokens": 128_000},
        "supported_parameters": ["tools", "reasoning"],
        "benchmarks": {"artificial_analysis": {"intelligence_index": 43.4}},
        "pricing": {
            "prompt": "0.0000001",
            "completion": "0.0000005",
            "input_cache_read": "0.00000001",
        },
    }
    return item | overrides


def test_parse_converts_prices_and_capabilities() -> None:
    models, skipped = parse_models([_raw("anthropic/claude-haiku-5.5")])
    assert skipped == []
    [m] = models
    assert (m.input_per_mtok, m.output_per_mtok, m.cached_input_per_mtok) == (0.1, 0.5, 0.01)
    assert m.provider == "anthropic"
    assert m.supports_tools and m.supports_reasoning
    assert m.input_modalities == ["text", "image"]
    assert m.intelligence_index == 43.4


def test_parse_keeps_only_our_providers_and_skips_batch_variants() -> None:
    raw = [
        _raw("openai/gpt-6-luna"),
        _raw("openai/gpt-6-luna:batch"),
        _raw("x-ai/grok-4.7"),
        _raw("google/gemini-x"),
    ]
    models, _ = parse_models(raw)
    assert [m.model_id for m in models] == ["openai/gpt-6-luna", "x-ai/grok-4.7"]


def test_parse_handles_missing_cache_price_and_bad_rows() -> None:
    no_cache = _raw("openai/a", pricing={"prompt": "0.000001", "completion": "0.000002"})
    variable = _raw("openai/b", pricing={"prompt": "-1", "completion": "-1"})
    broken = _raw("openai/c", pricing={})
    models, skipped = parse_models([no_cache, variable, broken])
    assert [m.model_id for m in models] == ["openai/a"]
    assert models[0].cached_input_per_mtok is None
    assert skipped == ["openai/b: variable pricing", "openai/c: KeyError"]


def test_missing_database_says_how_to_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_AGENT_DB", str(tmp_path / "nope.db"))
    with pytest.raises(FileNotFoundError, match="tool-agent-setup"):
        db.get_model("anthropic/claude-haiku-5.5")
