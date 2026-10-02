"""Chat with the Northwind Hotel agent in your terminal.

    uv run python demo.py                    # retrieval before the model, no handoff check
    uv run python demo.py --check enforce    # with the Jev handoff check (needs JEV_API_KEY)
    uv run python demo.py --no-presearch     # the model decides when to search

Each reply is followed by its trace: searches, the handoff check, model calls,
latency and tokens. Type "quit" to leave.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from harness import AgentConfig, HistoryMessage, TurnInput, run_turn
from harness.contracts import HandoffRequested, TextDelta, TurnFailed, TurnFinished
from harness.decisions import JevDecider
from harness.knowledge import KnowledgeIndex, OpenAIEmbedder
from harness.models import build_chat_model
from harness.sample import CACHE, knowledge_folder, load_agent

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=["off", "shadow", "enforce"], default="off")
    parser.add_argument("--no-presearch", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = AgentConfig(handoff_check=args.check, presearch=not args.no_presearch)
    decider = None
    if args.check != "off":
        decider = JevDecider(os.environ.get("JEV_API_KEY", ""))
    agent = load_agent()
    index = await KnowledgeIndex.build(knowledge_folder(), OpenAIEmbedder(), cache_dir=CACHE)
    model = build_chat_model(config)
    history: list[HistoryMessage] = []
    print(f"{BOLD}{agent.agent_name} at {agent.business_name}{RESET} ({len(index.chunks)} chunks)")

    while True:
        message = (await asyncio.to_thread(input, f"\n{BOLD}you>{RESET} ")).strip()
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        print(f"{BOLD}{agent.agent_name.lower()}>{RESET} ", end="", flush=True)
        turn = TurnInput(agent=agent, message=message, history=history)
        reply = ""
        async for event in run_turn(turn, index, model=model, config=config, decider=decider):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, HandoffRequested):
                print(f"\n{DIM}[handed to a person: {event.reason}. {event.summary}]{RESET}")
            elif isinstance(event, TurnFinished):
                reply = event.text
                print(f"\n{DIM}", end="")
                for step in event.steps:
                    print(f"  · {step.label}: {step.detail} ({step.duration_ms} ms)")
                usage = event.usage
                print(
                    f"  = {event.latency_ms} ms, first words at {event.first_token_ms} ms, "
                    f"{usage.model_calls} model call(s), "
                    f"{usage.tokens_in}+{usage.tokens_out} tokens{RESET}"
                )
            elif isinstance(event, TurnFailed):
                print(f"\n{DIM}[failed: {event.code}]{RESET}")
        history += [HistoryMessage("visitor", message), HistoryMessage("ai", reply)]

    if isinstance(decider, JevDecider):
        await decider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
