from pathlib import Path

import pytest

from tool_agent import db
from tool_agent.schemas import ModelInfo


def _model(model_id: str, **overrides: object) -> ModelInfo:
    base: dict[str, object] = {
        "model_id": model_id,
        "provider": model_id.split("/")[0],
        "name": model_id,
        "description": "Fictional test model.",
        "context_length": 200_000,
        "max_output_tokens": 64_000,
        "input_modalities": ["text"],
        "supports_tools": True,
        "supports_reasoning": False,
        "intelligence_index": None,
        "input_per_mtok": 1.0,
        "output_per_mtok": 1.0,
        "cached_input_per_mtok": None,
    }
    return ModelInfo.model_validate(base | overrides)


# Fictional models and prices, so tests never depend on live data.
FIXTURE_MODELS = [
    _model(
        "anthropic/acme-mini",
        input_per_mtok=0.50,
        output_per_mtok=2.00,
        cached_input_per_mtok=0.05,
        input_modalities=["text", "image"],
        intelligence_index=40.0,
    ),
    _model(
        "anthropic/acme-pro",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cached_input_per_mtok=0.30,
        context_length=1_000_000,
        supports_reasoning=True,
        intelligence_index=60.0,
    ),
    _model("openai/globex-lite", input_per_mtok=0.30, output_per_mtok=2.50, supports_tools=False),
]


@pytest.fixture(autouse=True)
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "pricing.db"
    db.create_schema(path)
    db.replace_models(path, FIXTURE_MODELS)
    monkeypatch.setenv("TOOL_AGENT_DB", str(path))
    return path
