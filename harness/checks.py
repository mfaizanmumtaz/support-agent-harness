"""The handoff check: a fast second opinion on whether this message needs a person.

A small classifier reads the latest message with a little context and answers
yes/no questions, all in one call, while the knowledge search runs:
- did the visitor ask for a person?
- does each of the business's handoff rules apply?
- is the visitor frustrated? is it a sensitive matter? (recorded, never acted on)

"shadow" mode only records the answers, next to what the LLM decided, so the
two can be compared on real chats before trusting the check. "enforce" mode
acts on a sure yes to the first two: the model's first call is made to call
`handoff_to_human`, so the normal handoff flow (reason, summary, wording)
does the rest. When the business has steps to follow before a handoff, the
model is told rather than forced, so those steps still happen.
"""

import re
import time
from collections.abc import Sequence
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage

from harness.config import AgentConfig
from harness.context import TurnContext
from harness.contracts import AgentSpec, HandoffChecked, HandoffReason, TurnInput
from harness.decisions import Decider, Question, YesNo

HANDOFF_TOOL = "handoff_to_human"
VISITOR_ASKED = "visitor_asked"
SIGNALS: dict[str, str] = {
    VISITOR_ASKED: (
        "In the latest message, does the visitor ask to talk to a human (a person, "
        "a teammate, an agent, staff) instead of the AI assistant?"
    ),
    "frustrated": "Is the visitor clearly frustrated or angry?",
    "sensitive": (
        "Is this a sensitive matter a person should handle, such as a legal threat, "
        "a safety issue, a complaint about staff, or a request about personal data?"
    ),
}
RULE_QUESTION = (
    'A business rule for handing a chat to a person says: "{rule}". Does this rule '
    "apply to the visitor's latest message?"
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
ROLE_NAMES = {"visitor": "Visitor", "ai": "Assistant", "human": "Teammate"}


def should_check(turn: TurnInput, config: AgentConfig) -> bool:
    return config.handoff_check != "off" and turn.agent.handoff_enabled


def rules_of(agent: AgentSpec, limit: int) -> list[str]:
    """The business's handoff rules, one per line, without their bullets."""
    rules = [_BULLET.sub("", line).strip() for line in agent.handoff_rules.splitlines()]
    return [rule for rule in rules if rule][:limit]


def check_state(turn: TurnInput, config: AgentConfig) -> dict[str, str]:
    count = config.handoff_check_history_messages
    earlier = turn.history[-count:] if count else []
    return {
        "business": turn.agent.business_name,
        "earlier_messages": "\n".join(
            f"{ROLE_NAMES[item.role]}: {item.text.strip()[:1000]}" for item in earlier
        )
        or "(none)",
        "latest_visitor_message": turn.message.strip()[: config.max_message_chars],
    }


def check_questions(rules: list[str]) -> dict[str, Question]:
    questions: dict[str, Question] = {name: YesNo(text) for name, text in SIGNALS.items()}
    for index, rule in enumerate(rules):
        questions[f"rule_{index + 1}"] = YesNo(RULE_QUESTION.format(rule=rule))
    return questions


async def run_check(turn: TurnInput, decider: Decider, config: AgentConfig) -> HandoffChecked:
    """Asks the decider. Raises what it raises: the caller treats that as no check."""
    rules = rules_of(turn.agent, config.handoff_check_max_rules)
    started = time.perf_counter()
    decisions = await decider.decide(check_state(turn, config), check_questions(rules))
    signals = {name: round(decisions.yes[name], 3) for name in SIGNALS}
    by_rule = {rule: round(decisions.yes[f"rule_{i + 1}"], 3) for i, rule in enumerate(rules)}
    verdict: HandoffReason | None = None
    matched: str | None = None
    threshold = config.handoff_check_threshold
    if signals[VISITOR_ASKED] >= threshold:
        verdict = "visitor_asked"
    elif by_rule:
        best = max(by_rule, key=lambda rule: by_rule[rule])
        if by_rule[best] >= threshold:
            verdict, matched = "rule_matched", best
    return HandoffChecked(
        mode=cast(Any, config.handoff_check),
        signals=signals,
        rules=by_rule,
        verdict=verdict,
        rule=matched,
        duration_ms=round((time.perf_counter() - started) * 1000),
        cost=decisions.cost,
    )


def failed_check(config: AgentConfig, error: str, duration_ms: int) -> HandoffChecked:
    return HandoffChecked(
        mode=cast(Any, config.handoff_check),
        signals={},
        rules={},
        verdict=None,
        rule=None,
        duration_ms=duration_ms,
        error=error,
    )


def describe(check: HandoffChecked) -> str:
    """One line for the turn's trace."""
    if check.error is not None:
        return f"skipped ({check.error})"
    scores = ", ".join(f"{name} {score:.2f}" for name, score in check.signals.items())
    if check.rules:
        scores += f", best rule {max(check.rules.values()):.2f}"
    if check.verdict is None:
        outcome = "no handoff"
    elif check.mode == "shadow":
        outcome = f"would hand off: {check.verdict}"
    else:
        outcome = f"hand off: {check.verdict}"
    return f"{outcome} ({scores})"


class ForceHandoff(AgentMiddleware[Any, TurnContext]):
    """Makes the model hand off first, when an enforced check was sure.

    Only the first call is forced: once the handoff tool has run, the model
    writes the reply as usual. Skipped when the business wants steps done first.
    """

    async def awrap_model_call(
        self, request: ModelRequest[TurnContext], handler: Any
    ) -> ModelResponse:
        return cast(ModelResponse, await handler(_forced(request)))

    def wrap_model_call(self, request: ModelRequest[TurnContext], handler: Any) -> ModelResponse:
        return cast(ModelResponse, handler(_forced(request)))


def _forced(request: ModelRequest[TurnContext]) -> ModelRequest[TurnContext]:
    context = request.runtime.context
    if (
        context.handoff_verdict is None
        or context.agent.handoff_procedure.strip()
        or _handed_off(request.messages)
    ):
        return request
    return request.override(tool_choice=HANDOFF_TOOL)


def _handed_off(messages: Sequence[BaseMessage]) -> bool:
    return any(
        isinstance(message, AIMessage)
        and any(call["name"] == HANDOFF_TOOL for call in message.tool_calls)
        for message in messages
    )
