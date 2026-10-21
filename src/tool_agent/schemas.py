"""Pydantic models for the model catalog, tool inputs and outputs, and the agent's final answer."""

from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["anthropic", "openai", "x-ai"]


class ModelInfo(BaseModel):
    """One row of the model catalog, as stored in SQLite."""

    model_id: str
    provider: str
    name: str
    description: str
    context_length: int
    max_output_tokens: int | None
    input_modalities: list[str]
    supports_tools: bool
    supports_reasoning: bool
    intelligence_index: float | None = Field(
        description="Artificial Analysis intelligence index; higher is more capable"
    )
    input_per_mtok: float = Field(description="USD per million input tokens")
    output_per_mtok: float = Field(description="USD per million output tokens")
    cached_input_per_mtok: float | None = Field(
        description="USD per million cached input tokens; null if the model has no cache price"
    )


class GetModelInput(BaseModel):
    model_id: str = Field(description="Model identifier, e.g. 'anthropic/claude-haiku-5.5'")


class ListModelsInput(BaseModel):
    provider: Provider | None = Field(default=None, description="Only this provider's models")
    min_context: int | None = Field(default=None, gt=0, description="Minimum context window")
    needs_tools: bool = Field(default=False, description="Only models that support tool use")
    needs_reasoning: bool = Field(default=False, description="Only models with reasoning")
    needs_image_input: bool = Field(default=False, description="Only models that accept images")
    max_input_per_mtok: float | None = Field(
        default=None, ge=0, description="Maximum USD per million input tokens"
    )
    sort_by: Literal["input_price", "intelligence", "context"] = "input_price"
    limit: int = Field(default=10, ge=1, le=50)


class ModelSummary(BaseModel):
    """A catalog row without the description, to keep list results short."""

    model_id: str
    name: str
    context_length: int
    input_modalities: list[str]
    supports_tools: bool
    supports_reasoning: bool
    intelligence_index: float | None
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None


class ModelList(BaseModel):
    total_matching: int
    models: list[ModelSummary]


class EstimateCostInput(BaseModel):
    model_id: str
    requests_per_day: int = Field(gt=0)
    input_tokens: int = Field(ge=0, description="Input tokens per request")
    output_tokens: int = Field(ge=0, description="Output tokens per request")
    cached_fraction: float = Field(
        default=0, ge=0, le=1, description="Share of input tokens served from the prompt cache"
    )


class CostEstimate(BaseModel):
    """What estimate_cost returns."""

    model_id: str
    monthly_requests: int
    monthly_cost: float
    cost_per_1k_requests: float
    explanation: str


class FinalAnswer(BaseModel):
    """Final answer the agent must return, citing the tool calls it relied on."""

    answer: str = Field(
        description="Plain-language reply to the user, ending with a sentence citing the sources"
    )
    source_tool_use_ids: list[str] = Field(
        min_length=1, description="The ids of the tool calls the answer is based on"
    )
    chosen: CostEstimate | None = Field(
        default=None,
        description="For cost questions only: the one estimate_cost result the answer picked",
    )
