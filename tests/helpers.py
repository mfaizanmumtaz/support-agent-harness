"""Shared test pieces: a sample agent, a fake searcher, a run helper."""

from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage

from harness import AgentConfig, AgentSpec, TurnEvent, TurnFinished, TurnInput, run_turn
from harness.contracts import Hit, SearchResult
from harness.decisions import ScriptedDecider
from harness.models import ScriptedChatModel

AGENT = AgentSpec(
    agent_name="Ava",
    business_name="Northwind Hotel",
    instructions="Be warm and brief.",
    handoff_rules=(
        "- Refund requests.\n* Group bookings of more than 10 rooms\n\n2) Complaints about staff"
    ),
    offline_message="We're closed, leave your email and we'll reply tomorrow.",
)


def hit(content: str, score: float = 0.8, heading: str = "") -> Hit:
    return Hit(uuid4().hex, "policies", content, score, 20, heading)


class FakeSearcher:
    def __init__(self, *results: list[Hit], fail: bool = False) -> None:
        self.results = list(results)
        self.fail = fail
        self.queries: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("vector store down")
        hits = self.results.pop(0) if self.results else []
        return SearchResult(query=query, hits=hits, candidates=len(hits) + 2, below_threshold=2)


def call(name: str, **args: Any) -> AIMessage:
    """A model reply that asks for one tool."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:8]}"}]
    )


async def run(
    model: ScriptedChatModel,
    searcher: FakeSearcher | None = None,
    *,
    message: str = "Can I get a refund?",
    agent: AgentSpec = AGENT,
    config: AgentConfig | None = None,
    decider: ScriptedDecider | None = None,
    **turn: Any,
) -> list[TurnEvent]:
    return [
        event
        async for event in run_turn(
            TurnInput(agent=agent, message=message, **turn),
            searcher or FakeSearcher(),
            model=model,
            # Most tests are about the model searching for itself; presearch has its own.
            config=config or AgentConfig(presearch=False),
            decider=decider,
        )
    ]


def finished(events: list[TurnEvent]) -> TurnFinished:
    assert isinstance(events[-1], TurnFinished), events[-1]
    return events[-1]


def system_prompt(model: ScriptedChatModel, call_index: int = 0) -> str:
    first = model.calls[call_index][0]
    assert isinstance(first, SystemMessage)
    return str(first.content)
