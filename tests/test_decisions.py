"""The Jev adapter, against a recorded reply instead of OpenRouter."""

import json
from typing import Any

import httpx
import pytest

from harness.decisions import (
    Choice,
    DecisionError,
    JevDecider,
    Question,
    Scale,
    ScriptedDecider,
    YesNo,
    parse_decisions,
)

QUESTIONS: dict[str, Question] = {
    "wants_human": YesNo("Is the visitor asking for a person?"),
    "topic": Choice("What is it about?", {"billing": "Money", "other": "Anything else"}),
    "urgency": Scale("How urgent?", ("Can wait", "Today", "Right now")),
}
# A real reply's shape (trimmed), from the decisions endpoint.
REPLY: dict[str, Any] = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "wants_human": {"type": "noul", "noul": 0.99},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 1, "other": 0},
            "confidence": 1,
        },
        "urgency": {
            "type": "score",
            "score": 1.56,
            "legend": {"0": "Can wait", "1": "Today", "2": "Right now"},
            "probabilities": {"0": 0, "1": 0.43, "2": 0.57},
            "confidence": 0.35,
        },
    },
    "usage": {"input_tokens": 594, "output_tokens": 135, "cost": 0.000024948},
}


async def test_questions_are_sent_in_jevs_format_with_the_key() -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=REPLY)

    decider = JevDecider("sk-or-test", transport=httpx.MockTransport(answer))
    await decider.decide({"ticket": "I was charged twice"}, QUESTIONS)

    [request] = sent
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer sk-or-test"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == {"ticket": "I was charged twice"}
    assert body["questions"] == {
        "wants_human": {"type": "noul", "instructions": "Is the visitor asking for a person?"},
        "topic": {
            "type": "choice",
            "instructions": "What is it about?",
            "criteria": {"billing": "Money", "other": "Anything else"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Can wait", "Today", "Right now"],
        },
    }


def test_a_reply_becomes_typed_decisions() -> None:
    decisions = parse_decisions(REPLY, QUESTIONS)

    assert decisions.yes == {"wants_human": 0.99}
    assert decisions.choices == {"topic": "billing"}
    # 1.56 on a 0-2 scale rounds to the top level.
    assert decisions.levels == {"urgency": 2}
    assert decisions.confidence == {"topic": 1.0, "urgency": 0.35}
    assert decisions.cost == pytest.approx(0.000024948)


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"answers": {}},
        {"answers": {**REPLY["answers"], "topic": {"choice": "refunds"}}},
        {"answers": {**REPLY["answers"], "wants_human": {"noul": "maybe"}}},
    ],
)
def test_an_unreadable_reply_is_a_decision_error(reply: dict[str, Any]) -> None:
    with pytest.raises(DecisionError):
        parse_decisions(reply, QUESTIONS)


async def test_a_provider_error_is_a_decision_error() -> None:
    decider = JevDecider(
        "sk-or-test",
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="bad key")),
    )

    with pytest.raises(DecisionError, match="401"):
        await decider.decide({}, QUESTIONS)


def test_without_a_key_there_is_no_decider() -> None:
    with pytest.raises(DecisionError):
        JevDecider("")


async def test_the_fake_answers_no_the_first_option_and_the_lowest_level() -> None:
    decisions = await ScriptedDecider().decide({}, QUESTIONS)

    assert decisions.yes == {"wants_human": 0.0}
    assert decisions.choices == {"topic": "billing"}
    assert decisions.levels == {"urgency": 0}
