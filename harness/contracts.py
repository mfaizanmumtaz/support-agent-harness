"""What goes into a turn and what comes out of it. Plain data only."""

from dataclasses import dataclass, field
from typing import Any, Literal

HistoryRole = Literal["visitor", "ai", "human"]
HandoffReason = Literal[
    "visitor_asked", "rule_matched", "not_in_knowledge", "frustrated", "sensitive"
]


@dataclass(frozen=True)
class AgentSpec:
    """One business's agent: who it is, how it talks, and when it hands off."""

    agent_name: str
    business_name: str
    # The business's own instructions: role, tone, scope.
    instructions: str = ""
    handoff_enabled: bool = True
    # When to hand the chat to a person, one rule per line.
    handoff_rules: str = ""
    # Steps to take before handing off (e.g. "ask for the booking number").
    handoff_procedure: str = ""
    # What the agent says when nobody is there to take the chat.
    offline_message: str = "Our team is away right now. We'll reply as soon as we're back."


@dataclass(frozen=True)
class HistoryMessage:
    """An earlier message. `human` is a teammate who replied for the business."""

    role: HistoryRole
    text: str


@dataclass(frozen=True)
class TurnInput:
    agent: AgentSpec
    message: str
    history: list[HistoryMessage] = field(default_factory=list)
    # Someone is online to take a handoff. Decides what a handoff promises.
    team_available: bool = True


# ---------------------------------------------------------------- knowledge


@dataclass(frozen=True)
class Hit:
    """One chunk of knowledge a search found."""

    chunk_id: str
    source_name: str
    content: str
    score: float
    token_count: int
    heading: str = ""


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[Hit]
    # Chunks scored before filtering, and how many were too weak to use.
    candidates: int = 0
    below_threshold: int = 0


# ---------------------------------------------------------------- events


@dataclass(frozen=True)
class TextDelta:
    """The next piece of the reply, for the typing effect."""

    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class KnowledgeSearched:
    query: str
    hits: list[Hit]
    candidates: int
    below_threshold: int
    duration_ms: int
    # Set when the search itself failed; the model was told it couldn't check.
    error: str | None = None
    type: Literal["knowledge_searched"] = "knowledge_searched"


@dataclass(frozen=True)
class HandoffRequested:
    """The agent wants a person. The caller decides what actually happens."""

    reason: HandoffReason
    summary: str
    type: Literal["handoff_requested"] = "handoff_requested"


@dataclass(frozen=True)
class HandoffChecked:
    """The fast classifier's view of whether this message needs a person."""

    mode: Literal["shadow", "enforce"]
    # Probability of yes: visitor_asked, frustrated, sensitive.
    signals: dict[str, float]
    # Probability of yes per business rule, keyed by the rule's text.
    rules: dict[str, float]
    # visitor_asked or rule_matched when sure enough, else None. In shadow
    # mode it is what would have happened.
    verdict: HandoffReason | None
    rule: str | None
    duration_ms: int
    cost: float = 0.0
    # Set when the check failed or was too slow; the turn went on without it.
    error: str | None = None
    type: Literal["handoff_checked"] = "handoff_checked"


@dataclass(frozen=True)
class Usage:
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0


@dataclass(frozen=True)
class Step:
    """One line of the turn's trace."""

    label: str
    detail: str
    duration_ms: int


@dataclass(frozen=True)
class TurnFinished:
    text: str
    # Every chunk the model was shown, best first, each once.
    sources: list[Hit]
    usage: Usage
    latency_ms: int
    # Until the first words appeared: what the visitor feels as speed.
    first_token_ms: int | None
    steps: list[Step]
    type: Literal["turn_finished"] = "turn_finished"


@dataclass(frozen=True)
class TurnFailed:
    """No reply could be made. The caller shows a safe fallback message."""

    code: Literal["timeout", "model_error", "empty_reply", "engine_error"]
    message: str
    usage: Usage | None = None
    steps: list[Step] = field(default_factory=list)
    type: Literal["turn_failed"] = "turn_failed"


TurnEvent = (
    TextDelta | KnowledgeSearched | HandoffRequested | HandoffChecked | TurnFinished | TurnFailed
)


def as_dict(event: Any) -> dict[str, Any]:
    """An event as JSON-ready data, e.g. for a log or a WebSocket."""
    from dataclasses import asdict

    return asdict(event)
