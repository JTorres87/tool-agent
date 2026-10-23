"""One-shot CLI: ask a question, get a verified answer as JSON."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from tool_agent.agent import run_agent

DEFAULT_QUESTION = (
    "Which Anthropic model that accepts images is cheapest, and what would it cost per month "
    "for 10,000 requests a day with 2,000 input and 500 output tokens each?"
)


def setup_logging(verbosity: int) -> None:
    """-v shows the checks and the verifier's verdicts; -vv also shows every model step."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    logging.getLogger("tool_agent").setLevel(level)


def verbosity_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v: show the checks; -vv: also show every model step (on stderr)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tool-agent", description="Ask which LLM to use and what it will cost."
    )
    parser.add_argument("question", nargs="*", help="your question (a default is used if empty)")
    verbosity_flag(parser)
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.verbose)
    result = run_agent(" ".join(args.question) or DEFAULT_QUESTION)
    print(result.model_dump_json(indent=2))
    if not result.verified:
        print("\nWARNING: the verifier did not accept this answer.", file=sys.stderr)
