"""Interactive chat: follow-up questions in one session, using the same agent loop."""

import anthropic
from dotenv import load_dotenv

from tool_agent.agent import Conversation, run_turn
from tool_agent.schemas import AgentResult

WELCOME = "Ask about models and costs. Type /quit to leave. (Step traces go to stderr.)"
QUIT = {"/quit", "/exit"}


def main() -> None:
    load_dotenv()
    chat(anthropic.Anthropic())


def chat(client: anthropic.Anthropic) -> None:
    # TODO 8 (done): read a line, run a turn, print the answer; repeat
    conversation = Conversation()
    print(WELCOME)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-Z, or Ctrl-C at the prompt
            print()
            break
        if not line:
            continue
        if line.lower() in QUIT:
            break

        # TODO 9 (done): a failed turn doesn't end the session. run_turn has already
        # rolled the history back, so the user can just ask again.
        try:
            result = run_turn(conversation, line, client)
        except KeyboardInterrupt:
            print("\n(Cancelled. That question was dropped; ask again or try another.)")
            continue
        except (anthropic.APIError, RuntimeError) as e:
            print(f"\n(That didn't work: {e}. The question was dropped; try again.)")
            continue
        print(format_result(result))


def format_result(result: AgentResult) -> str:
    lines = ["", result.answer.answer]
    if result.answer.chosen:
        c = result.answer.chosen
        lines.append(
            f"  Estimate: {c.model_id}, ${c.monthly_cost:,.2f} a month "
            f"(${c.cost_per_1k_requests:,.2f} per 1,000 requests)"
        )
    lines.append(f"  Sources: {', '.join(result.answer.source_tool_use_ids)}")
    if not result.verified:
        lines.append("  WARNING: the verifier did not accept this answer:")
        lines.extend(f"    - {concern}" for concern in result.concerns)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
