# tool-agent

A CLI agent that helps you choose an LLM and plan what it will cost to run. It's written in plain Python with the Anthropic SDK and no agent framework. Ask it things like "Which Anthropic model that accepts images is cheapest?" or "What would 50,000 requests a day cost on each Claude model?" It answers by calling typed tools over a SQLite catalog of Anthropic, OpenAI and xAI models, with prices from OpenRouter.

## What's inside

- **Agent loop** ([agent.py](src/tool_agent/agent.py)): model call, tool dispatch, tool results fed back into context, stop condition, max-step guard
- **Chat session** ([chat.py](src/tool_agent/chat.py)): the same loop inside a conversation, so follow-up questions see earlier answers
- **Typed tools** ([tools.py](src/tool_agent/tools.py)): `list_models`, `get_model` and `estimate_cost`. Every tool input and output is a Pydantic model, and tool schemas are generated from those models.
- **Model catalog** ([db.py](src/tool_agent/db.py), [seed.py](src/tool_agent/seed.py)): SQLite, seeded from OpenRouter's public model list (no API key needed). Prices are OpenRouter's listed rates.
- **Cited final answer** ([schemas.py](src/tool_agent/schemas.py)): the agent returns a validated `FinalAnswer` that cites the tool calls it relied on, and the citations are checked against the calls actually made
- **Verifier** ([verify.py](src/tool_agent/verify.py)): Haiku does the work; Sonnet then checks each final answer's claims against the tool results it cites. A rejected answer goes back for a fix; after two rejections it's returned flagged, with the verifier's concerns.

## Run it

```bash
cp .env.example .env       # add your ANTHROPIC_API_KEY
uv sync
uv run tool-agent-setup    # build data/pricing.db from OpenRouter
uv run tool-agent "Which Anthropic model that accepts images is cheapest?"
uv run tool-agent-chat     # ask follow-up questions in one session
```

Run `uv run tool-agent-seed` to refresh prices in an existing database.

## Develop

```bash
uv run ruff check . && uv run ruff format .
uv run mypy src tests
uv run pytest
```

Tests use a temporary database of fictional models, so they never depend on live prices.

Architecture diagram: [docs/architecture.md](docs/architecture.md).
