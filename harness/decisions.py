"""Fast typed decisions about a conversation, from a small classifier model.

A decider writes nothing. It reads a `state` (named pieces of text) and answers
every question in one call: a yes/no as a probability, one option from a list,
or a level on a scale. That is cheap and fast enough to run beside every turn,
where a second LLM call would double the cost and the wait.

`JevDecider` is TypeSafe's Jev through OpenRouter's decisions endpoint.
`ScriptedDecider` answers from a script, for tests.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

JEV_URL = "https://openrouter.ai/api/alpha/decisions"


@dataclass(frozen=True)
class YesNo:
    instructions: str


@dataclass(frozen=True)
class Choice:
    instructions: str
    # Option id -> what it means.
    options: Mapping[str, str]


@dataclass(frozen=True)
class Scale:
    instructions: str
    # Lowest first.
    levels: tuple[str, ...]


Question = YesNo | Choice | Scale


@dataclass(frozen=True)
class Decisions:
    # Per `YesNo`: the probability the answer is yes.
    yes: dict[str, float] = field(default_factory=dict)
    # Per `Choice`: the option picked.
    choices: dict[str, str] = field(default_factory=dict)
    # Per `Scale`: the index into its levels, rounded from the model's score.
    levels: dict[str, int] = field(default_factory=dict)
    # Per `Choice` and `Scale`: how sure the model is, 0 to 1.
    confidence: dict[str, float] = field(default_factory=dict)
    model: str = ""
    # In US dollars, as the provider reports it.
    cost: float = 0.0


class DecisionError(RuntimeError):
    """The decider couldn't answer: bad key, provider error, or an unreadable reply."""


class Decider(Protocol):
    async def decide(
        self, state: Mapping[str, str], questions: Mapping[str, Question]
    ) -> Decisions: ...


class JevDecider:
    """One HTTP client per process: a fresh TLS handshake costs more than the answer."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "typesafe/jev-1.13",
        url: str = JEV_URL,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise DecisionError("No Jev API key.")
        self._url = url
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    async def decide(
        self, state: Mapping[str, str], questions: Mapping[str, Question]
    ) -> Decisions:
        response = await self._client.post(
            self._url,
            json={
                "model": self._model,
                "state": dict(state),
                "questions": {name: question_json(q) for name, q in questions.items()},
            },
        )
        if response.status_code != 200:
            raise DecisionError(f"Jev answered {response.status_code}: {response.text[:300]}")
        return parse_decisions(response.json(), questions)

    async def aclose(self) -> None:
        await self._client.aclose()


def question_json(question: Question) -> dict[str, Any]:
    if isinstance(question, YesNo):
        return {"type": "noul", "instructions": question.instructions}
    if isinstance(question, Choice):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.options),
        }
    return {
        "type": "score",
        "instructions": question.instructions,
        "criteria": list(question.levels),
    }


def parse_decisions(body: Any, questions: Mapping[str, Question]) -> Decisions:
    """Reads Jev's reply. A question missing from it, or of the wrong shape, is an error."""
    try:
        answers = body["answers"]
        decisions = Decisions(
            model=str(body.get("model", "")),
            cost=float((body.get("usage") or {}).get("cost") or 0.0),
        )
        for name, question in questions.items():
            answer = answers[name]
            if isinstance(question, YesNo):
                decisions.yes[name] = float(answer["noul"])
            elif isinstance(question, Choice):
                choice = str(answer["choice"])
                if choice not in question.options:
                    raise DecisionError(f"Jev picked an unknown option for {name}: {choice}")
                decisions.choices[name] = choice
                decisions.confidence[name] = float(answer.get("confidence", 0.0))
            else:
                top = len(question.levels) - 1
                decisions.levels[name] = min(max(round(float(answer["score"])), 0), top)
                decisions.confidence[name] = float(answer.get("confidence", 0.0))
        return decisions
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionError(f"Jev's reply couldn't be read: {exc!r}") from exc


@dataclass
class ScriptedDecider:
    """For tests. Unscripted questions get "no", the first option, the lowest level."""

    answers: dict[str, float | str | int] = field(default_factory=dict)
    fail: bool = False
    delay_seconds: float = 0.0
    calls: list[tuple[dict[str, str], dict[str, Question]]] = field(default_factory=list)

    async def decide(
        self, state: Mapping[str, str], questions: Mapping[str, Question]
    ) -> Decisions:
        self.calls.append((dict(state), dict(questions)))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.fail:
            raise DecisionError("scripted failure")
        decisions = Decisions(model="scripted-decider")
        for name, question in questions.items():
            scripted = self.answers.get(name)
            if isinstance(question, YesNo):
                decisions.yes[name] = float(scripted or 0.0)
            elif isinstance(question, Choice):
                decisions.choices[name] = str(scripted or next(iter(question.options)))
                decisions.confidence[name] = 1.0
            else:
                decisions.levels[name] = int(scripted or 0)
                decisions.confidence[name] = 1.0
        return decisions
