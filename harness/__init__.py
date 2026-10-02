"""A customer-support AI agent harness.

    async for event in run_turn(TurnInput(agent, "Can I bring my dog?"), index, model=model):
        ...

Everything the agent needs is passed in (the business's agent spec, a searcher
locked to that business's knowledge, a chat model, optionally a decider for
the handoff check), and everything it does comes back as events. So it can be
tested without a network and tuned without touching the rest of an app.
"""

from harness.config import AgentConfig, RetrievalConfig
from harness.contracts import (
    AgentSpec,
    HandoffChecked,
    HandoffRequested,
    HistoryMessage,
    Hit,
    KnowledgeSearched,
    SearchResult,
    TextDelta,
    TurnEvent,
    TurnFailed,
    TurnFinished,
    TurnInput,
)
from harness.runner import run_turn

__all__ = [
    "AgentConfig",
    "AgentSpec",
    "HandoffChecked",
    "HandoffRequested",
    "HistoryMessage",
    "Hit",
    "KnowledgeSearched",
    "RetrievalConfig",
    "SearchResult",
    "TextDelta",
    "TurnEvent",
    "TurnFailed",
    "TurnFinished",
    "TurnInput",
    "run_turn",
]
