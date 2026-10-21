"""Fetch Anthropic, OpenAI and xAI models from OpenRouter and seed the SQLite catalog.

    uv run tool-agent-setup   # delete and rebuild the database, then seed it
    uv run tool-agent-seed    # refresh the models in an existing database

OpenRouter's model list needs no API key. Prices are OpenRouter's listed rates, which
normally match each provider's own.
"""

import json
import urllib.request
from typing import Any

from pydantic import ValidationError

from tool_agent import db
from tool_agent.schemas import ModelInfo

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PROVIDERS = ("anthropic", "openai", "x-ai")
PER_TOKEN_TO_PER_MTOK = 1_000_000


def fetch_openrouter(url: str = OPENROUTER_MODELS_URL) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "tool-agent-seeder"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data: list[dict[str, Any]] = json.load(response)["data"]
    return data


def parse_models(raw: list[dict[str, Any]]) -> tuple[list[ModelInfo], list[str]]:
    """Keep our providers' models, skip :batch variants, convert per-token prices.

    Returns the parsed models and a list of skip reasons for anything that didn't fit.
    """
    models: list[ModelInfo] = []
    skipped: list[str] = []
    for item in raw:
        model_id: str = item["id"]
        provider = model_id.split("/")[0]
        if provider not in PROVIDERS or ":" in model_id:
            continue
        pricing = item.get("pricing", {})
        params = item.get("supported_parameters") or []
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
            if prompt < 0 or completion < 0:  # OpenRouter uses -1 for variable pricing
                skipped.append(f"{model_id}: variable pricing")
                continue
            cache_read = pricing.get("input_cache_read")
            models.append(
                ModelInfo(
                    model_id=model_id,
                    provider=provider,
                    name=item["name"],
                    description=item.get("description") or "",
                    context_length=item["context_length"],
                    max_output_tokens=(item.get("top_provider") or {}).get("max_completion_tokens"),
                    input_modalities=(item.get("architecture") or {}).get(
                        "input_modalities", ["text"]
                    ),
                    supports_tools="tools" in params,
                    supports_reasoning="reasoning" in params,
                    intelligence_index=(
                        (item.get("benchmarks") or {}).get("artificial_analysis") or {}
                    ).get("intelligence_index"),
                    input_per_mtok=round(prompt * PER_TOKEN_TO_PER_MTOK, 6),
                    output_per_mtok=round(completion * PER_TOKEN_TO_PER_MTOK, 6),
                    cached_input_per_mtok=(
                        round(float(cache_read) * PER_TOKEN_TO_PER_MTOK, 6) if cache_read else None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            skipped.append(f"{model_id}: {type(e).__name__}")
    return models, skipped


def seed() -> None:
    path = db.db_path()
    db.create_schema(path)
    models, skipped = parse_models(fetch_openrouter())
    if not models:
        raise SystemExit("OpenRouter returned no usable models; the database was not changed.")
    count = db.replace_models(path, models)
    by_provider = {p: sum(m.provider == p for m in models) for p in PROVIDERS}
    print(f"Seeded {count} models into {path}: {by_provider}")
    for reason in skipped:
        print(f"  skipped {reason}")


def setup() -> None:
    path = db.db_path()
    path.unlink(missing_ok=True)
    print(f"Creating {path}")
    seed()
