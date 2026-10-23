# tool-agent

**A CLI agent that helps you choose an LLM and plan what it will cost to run.** It's a tool-calling agent written in plain Python on the Anthropic SDK, with no agent framework, so every part of the loop is visible.

> **Project 01** in my [AI engineering series](https://github.com/JTorres87/ai-engineering-lab): hands-on projects across agents, fine-tuning, evals and RAG, with a new one in the series every week or so. Each project ships as a working base and grows in tagged releases.

```text
$ uv run tool-agent-chat
Ask about models and costs. Type /quit to leave.
> Which OpenAI models support tools and reasoning?

42 OpenAI models match the filter for both tool use and reasoning: gpt-oss-20b,
gpt-5-nano, ... The lowest input price is gpt-oss-20b at $0.018 per million input
tokens. Source: list_models (toolu_01Mmq...).
  Sources: toolu_01MmqW9kaMQQDd9XA3WAGLr7

> What would the cheapest of those cost at 5,000 requests a day, with 1,000 input and 300 output tokens each?

The cheapest of the 42 matching models is openai/gpt-oss-20b ... $6.75 per month
for 150,000 requests, or $0.05 per 1,000 requests. Source: cost estimate
(toolu_015Uv...), with the choice based on the earlier filtered list (toolu_01Mmq...).
  Estimate: openai/gpt-oss-20b, $6.75 a month ($0.05 per 1,000 requests)
  Sources: toolu_015UvTMitBxvRgovE5ntmS1D, toolu_01MmqW9kaMQQDd9XA3WAGLr7
```

## What I built

- **A hand-written agent loop.** Model call, tool dispatch, tool results fed back into context, a stop condition and a max-step guard. Writing it without a framework was the point: I wanted to own every decision a framework would otherwise make for me.
- **Typed tools over a real catalog.** `list_models`, `get_model` and `estimate_cost` take and return Pydantic models, and their JSON schemas are generated from those models. They read a SQLite catalog of Anthropic, OpenAI and xAI models, seeded from OpenRouter's public model list. Prices, context windows, capabilities and an intelligence index all come from that list, so the agent never guesses a number.
- **Cited, structured answers.** The final answer is a validated `FinalAnswer`: a plain-language reply, the tool calls it relied on, and, for cost questions, the exact estimate it chose.
- **Two checks on every answer, cheap one first.**
  1. **Citation check** (free, in code): every cited tool call must have happened, and the chosen estimate must match what that call actually returned.
  2. **Verifier** (Claude Sonnet): a stronger model reads only the question, the answer and the cited results, and rejects any claim they don't support. The answer goes back to the agent for a fix. After two rejections it's returned flagged, with the verifier's concerns, so a wrong or picky verifier can't burn the budget.
- **A chat session on the same loop.** Follow-ups see earlier answers and can cite their tool calls. A turn that fails partway (API error, step limit, Ctrl-C) is rolled back, so the history never ends on an unanswered tool call and the session keeps going.

The agent runs on **Claude Haiku**; **Claude Sonnet** checks its work. A verified answer typically costs about **$0.01**: under a third of a cent for Haiku, the rest for the check.

Architecture diagram: [docs/architecture.md](docs/architecture.md).

## What I learned

**The verifier earns its cost.** The citation check can confirm that a number came from a tool, but not that a sentence about it is true. Two answers it would have let through:

- *"Is Claude Haiku 5.5 smarter than Claude Sonnet 5?"* The agent answered "No… Haiku 5.5 scores lower (43.4 vs 38.2)". The verifier rejected it: 43.4 is higher. The corrected answer passed.
- *"Which OpenAI models support tools and reasoning?"* The first draft said *all* OpenAI models support both. The verifier pointed out that a filtered search can't show that.

**A strict check can block the right answer.** Asking about a model that doesn't exist came back unverified after two rejections. The verifier was right each time: my citation rule banned citing failed tool calls, but a failed lookup ("Unknown model_id") is exactly the evidence that a model doesn't exist. Allowing those citations, and marking them as errors for the verifier, fixed it. Now I test the "not found" paths, not just the happy path.

**When a model keeps making the same mistake, check what it can see.** Haiku cited tool names instead of tool-call ids on its first answer every time, so the citation check forced a retry on every question. Rewording the instruction did nothing, because the model never saw the ids: the API attaches them to tool calls, and the model only reads the results. Writing each id into its tool result fixed it:

| Variant (5 runs each) | Citation check failed | Model calls | Agent tokens | Time |
|---|---|---|---|---|
| Before | 5 of 5 | 5 | 16,822 | 11.1 s |
| Reworded instruction | 5 of 5 | 5 | 16,641 | 9.4 s |
| **Id written into each tool result** | **0 of 5** | **4** | **11,423** | **8.0 s** |

**The verifier's cost is its input, not its effort.** At low, medium and high effort it gave the right verdict on all three test answers, with the same cost: about 2,500 tokens of evidence is nearly the whole bill, and a "correct" verdict is 16 output tokens. So the lever is sending less evidence, not lowering effort. That's on the roadmap.

## Run it

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
cp .env.example .env       # add your ANTHROPIC_API_KEY
uv sync
uv run tool-agent-setup    # build data/pricing.db from OpenRouter (no key needed)

uv run tool-agent "Which Anthropic model that accepts images is cheapest?"
uv run tool-agent-chat     # ask follow-up questions in one session
```

Add `-v` to see the citation check and the verifier's verdicts, or `-vv` to also see every model step and tool call. Run `uv run tool-agent-seed` to refresh prices in an existing database. Prices are OpenRouter's listed rates.

## How it's organized

| Path | What's there |
|---|---|
| [`src/tool_agent/agent.py`](src/tool_agent/agent.py) | The loop, the conversation, and the citation check |
| [`src/tool_agent/verify.py`](src/tool_agent/verify.py) | The verifier |
| [`src/tool_agent/tools.py`](src/tool_agent/tools.py) | The three tools, their schemas, and the dispatcher |
| [`src/tool_agent/schemas.py`](src/tool_agent/schemas.py) | Pydantic models for tool inputs and outputs, `FinalAnswer`, `Verdict` and `AgentResult` |
| [`src/tool_agent/db.py`](src/tool_agent/db.py), [`seed.py`](src/tool_agent/seed.py) | The SQLite catalog and the OpenRouter seeder |
| [`src/tool_agent/cli.py`](src/tool_agent/cli.py), [`chat.py`](src/tool_agent/chat.py) | The one-shot command and the chat session |
| [`src/tool_agent/config.py`](src/tool_agent/config.py) | Models, step limit, verifier limit |

## Tests

```bash
uv run pytest                                   # 36 tests, no API calls
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests                           # strict
```

The tests use a fake client that replays scripted model responses, so the loop, both checks, the retries and the chat rollback are all tested without calling the API. Tool tests run against a temporary database of fictional models, so they never depend on live prices.

## Roadmap

**v0.1.0 (this release):** the agent loop, typed tools over the model catalog, cited answers with a citation check and a verifier, and the chat session.

**Planned for v0.2.0:**

- **Logging and evals:** count tool-input errors and unparseable answers alongside citation failures and verifier rejections; grow the verifier's test answers into a labeled eval set and measure how often it's right.
- **Reliability:** my own retries with exponential backoff for rate limits and overloads, and switching to the fallback model when the main one keeps failing.
- **Cost:** prompt caching with before-and-after costs, smaller verifier evidence, and a price for every answer, calculated from the agent's own catalog.
- **Streaming,** so chat answers appear as they're written.
- **A reasoning model vs. a fast model** on the same questions: accuracy, time and cost.

## License

[Apache 2.0](LICENSE)
