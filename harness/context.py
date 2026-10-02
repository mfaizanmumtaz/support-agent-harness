"""What the prompt and the tools can see during one turn.

The model never sees this object. Tools read the searcher from here, so the
model can only choose *what* to search for, never *whose* knowledge.
"""

from dataclasses import dataclass, field
from typing import Protocol

from harness.config import AgentConfig
from harness.contracts import AgentSpec, HandoffChecked, SearchResult, Step, TurnEvent


class KnowledgeSearcher(Protocol):
    async def search(self, query: str) -> SearchResult: ...


@dataclass
class TurnContext:
    agent: AgentSpec
    searcher: KnowledgeSearcher
    config: AgentConfig
    team_available: bool
    # Tools add events and trace lines here; the runner passes them on.
    events: list[TurnEvent] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    # Knowledge found before the first model call, formatted for the prompt.
    presearch: str | None = None
    # Set when an enforced handoff check says this message needs a person.
    handoff_verdict: HandoffChecked | None = None
