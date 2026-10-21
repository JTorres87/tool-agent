import sys

from dotenv import load_dotenv

from tool_agent.agent import run_agent

DEFAULT_QUESTION = (
    "Which Anthropic model that accepts images is cheapest, and what would it cost per month "
    "for 10,000 requests a day with 2,000 input and 500 output tokens each?"
)


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    result = run_agent(question)
    print(result.model_dump_json(indent=2))
    if not result.verified:
        print("\nWARNING: the verifier did not accept this answer.", file=sys.stderr)
