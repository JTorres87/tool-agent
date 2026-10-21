"""Interactive chat: follow-up questions in one session. Build it after the agent loop works."""

from dotenv import load_dotenv

WELCOME = "Ask about models and costs. Type /quit to leave."


def main() -> None:
    load_dotenv()
    print(WELCOME)
    # TODO 7: refactor agent.py so the loop can continue a conversation:
    #         run_turn(messages, client) takes the history so far, runs the
    #         same inner loop (citation check and verifier included), and
    #         returns the AgentResult. The history keeps
    #         growing, so the model sees earlier questions, tool calls and answers.
    #         Keep run_agent(question) working by building on run_turn.
    # TODO 8: the outer loop: read a line with input("> "), stop on /quit or
    #         Ctrl-D (EOFError), append it as a user message, call run_turn,
    #         print the answer and its sources, and warn when it isn't verified.
    # TODO 9: keep the session alive when a turn fails (APIError, RuntimeError
    #         from the step guard): print the error and let the user try again.
    #         Then think about what to do with the messages from the failed turn.
    raise NotImplementedError("Write the chat loop")
