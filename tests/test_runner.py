"""One agent turn, with a scripted model in place of OpenAI."""

import asyncio

import httpx2
import openai
import pytest

from harness import (
    AgentConfig,
    HandoffRequested,
    HistoryMessage,
    KnowledgeSearched,
    TextDelta,
    TurnFailed,
)
from harness.contracts import AgentSpec, SearchResult
from harness.models import ScriptedChatModel
from harness.runner import is_follow_up, trim_history
from harness.tokens import count_tokens
from harness.tools import NOTHING_FOUND, SEARCH_FAILED, TEAM_AVAILABLE
from tests.helpers import AGENT, FakeSearcher, call, finished, hit, run, system_prompt

# ---------------------------------------------------------------- answering


async def test_a_direct_answer_streams_then_finishes() -> None:
    model = ScriptedChatModel().script("Hello! How can I help?")

    events = await run(model, message="hi")

    deltas = [event.text for event in events if isinstance(event, TextDelta)]
    assert "".join(deltas) == "Hello! How can I help?"
    result = finished(events)
    assert result.text == "Hello! How can I help?"
    assert result.usage.model_calls == 1
    assert result.first_token_ms is not None


async def test_the_model_searches_then_answers_from_what_it_found() -> None:
    refund = hit("Flexible rates can be cancelled until 48 hours before arrival.")
    searcher = FakeSearcher([refund])
    model = ScriptedChatModel().script(
        call("search_knowledge", query="cancellation policy"), "Until 48 hours before arrival."
    )

    events = await run(model, searcher, message="When can I cancel?")

    assert searcher.queries == ["cancellation policy"]
    searched = next(e for e in events if isinstance(e, KnowledgeSearched))
    assert searched.hits == [refund]
    tool_result = model.calls[1][-1].content
    assert "<knowledge>" in tool_result and "48 hours" in tool_result
    result = finished(events)
    assert result.sources == [refund]
    assert result.usage.model_calls == 2


async def test_nothing_found_is_said_plainly_to_the_model() -> None:
    model = ScriptedChatModel().script(call("search_knowledge", query="spa"), "We have no spa.")

    await run(model, FakeSearcher([]))

    assert model.calls[1][-1].content == NOTHING_FOUND


async def test_a_search_outage_is_survived() -> None:
    model = ScriptedChatModel().script(call("search_knowledge", query="x"), "I can't check now.")

    events = await run(model, FakeSearcher(fail=True))

    assert model.calls[1][-1].content == SEARCH_FAILED
    assert next(e for e in events if isinstance(e, KnowledgeSearched)).error == "ConnectionError"
    assert finished(events).text == "I can't check now."


# ---------------------------------------------------------------- handoff


async def test_a_handoff_is_requested_and_the_model_says_someone_is_coming() -> None:
    model = ScriptedChatModel().script(
        call("handoff_to_human", reason="visitor_asked", summary=" Wants a person. "),
        "Connecting you with the team now.",
    )

    events = await run(model, message="let me talk to a person")

    assert HandoffRequested(reason="visitor_asked", summary="Wants a person.") in events
    assert model.calls[1][-1].content == TEAM_AVAILABLE
    assert finished(events).text == "Connecting you with the team now."


async def test_when_the_team_is_away_the_model_is_given_the_offline_message() -> None:
    model = ScriptedChatModel().script(
        call("handoff_to_human", reason="visitor_asked", summary="x"), "ok"
    )

    await run(model, team_available=False)

    assert AGENT.offline_message in model.calls[1][-1].content


async def test_without_handoff_the_tool_is_not_offered() -> None:
    model = ScriptedChatModel().script("I'm an automated assistant.")
    no_handoff = AgentSpec(**{**AGENT.__dict__, "handoff_enabled": False})

    await run(model, agent=no_handoff)

    assert model.bound_tools == ["search_knowledge"]
    assert "No human handoff is available" in system_prompt(model)


async def test_only_one_handoff_per_turn() -> None:
    model = ScriptedChatModel().script(
        call("handoff_to_human", reason="visitor_asked", summary="a"),
        call("handoff_to_human", reason="visitor_asked", summary="b"),
        "Connecting you.",
    )

    events = await run(model)

    assert sum(isinstance(e, HandoffRequested) for e in events) == 1


# ---------------------------------------------------------------- limits


async def test_a_model_that_keeps_searching_is_stopped_at_the_search_limit() -> None:
    model = ScriptedChatModel().script(
        *[call("search_knowledge", query=f"try {n}") for n in range(5)], "I couldn't find it."
    )
    searcher = FakeSearcher()

    await run(model, searcher, config=AgentConfig(presearch=False, search_limit=2))

    assert len(searcher.queries) == 2


async def test_model_calls_are_capped() -> None:
    model = ScriptedChatModel().script(*[call("search_knowledge", query="x") for _ in range(10)])

    events = await run(
        model, config=AgentConfig(presearch=False, model_call_limit=2, search_limit=10)
    )

    assert len(model.calls) == 2
    assert isinstance(events[-1], TurnFailed) or finished(events).usage.model_calls == 2


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (openai.APITimeoutError(request=httpx2.Request("POST", "https://x")), "timeout"),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), "model_error"),
        (RuntimeError("bug"), "engine_error"),
    ],
)
async def test_failures_end_the_turn_cleanly(error: BaseException, code: str) -> None:
    events = await run(ScriptedChatModel().script(error))

    assert isinstance(events[-1], TurnFailed) and events[-1].code == code


async def test_a_turn_that_takes_too_long_is_stopped() -> None:
    class SlowSearcher(FakeSearcher):
        async def search(self, query: str) -> SearchResult:
            await asyncio.sleep(5)
            return SearchResult(query=query, hits=[])

    config = AgentConfig(turn_timeout_seconds=0.1, timeout_seconds=0.1)
    events = await run(ScriptedChatModel(), SlowSearcher(), config=config)

    assert isinstance(events[-1], TurnFailed) and events[-1].code == "timeout"


# ---------------------------------------------------------------- the prompt


async def test_platform_rules_come_before_the_business_text() -> None:
    model = ScriptedChatModel().script("ok")

    await run(model)

    prompt = system_prompt(model)
    assert prompt.index("Rules that always apply") < prompt.index("Be warm and brief.")
    assert "- Refund requests." in prompt
    assert "You are Ava, working for Northwind Hotel." in prompt


async def test_teammates_speak_for_the_business_with_a_marker() -> None:
    model = ScriptedChatModel().script("ok")
    history = [HistoryMessage("visitor", "hi"), HistoryMessage("human", "Jo here, one sec")]

    await run(model, history=history)

    sent = model.calls[0]
    assert sent[-2].content == "[Human teammate]: Jo here, one sec"
    assert sent[-1].content == "Can I get a refund?"


def test_history_is_trimmed_to_the_budget_oldest_first() -> None:
    history = [HistoryMessage("visitor", f"message number {n}") for n in range(10)]
    budget = sum(count_tokens(item.text) for item in history[-3:])

    assert trim_history(history, budget) == history[-3:]


# ---------------------------------------------------------------- search before the model


async def test_knowledge_found_first_lets_the_model_answer_in_one_call() -> None:
    refund = hit("Refunds take 5 to 10 working days.", heading="Refunds")
    searcher = FakeSearcher([refund])
    model = ScriptedChatModel().script("Refunds take 5 to 10 working days.")

    events = await run(model, searcher, config=AgentConfig())

    result = finished(events)
    assert searcher.queries == ["Can I get a refund?"]
    assert result.usage.model_calls == 1
    assert result.sources == [refund]
    assert [step.label for step in result.steps] == ["search before the model", "model call 1"]
    prompt = system_prompt(model)
    assert "Knowledge already found" in prompt and "5 to 10 working days" in prompt


async def test_when_nothing_is_found_first_the_model_is_told_it_can_search() -> None:
    model = ScriptedChatModel().script("I don't know that.")

    await run(model, FakeSearcher([]), config=AgentConfig())

    assert "found nothing relevant" in system_prompt(model)


async def test_a_short_follow_up_is_searched_with_the_previous_question() -> None:
    searcher = FakeSearcher([])
    history = [HistoryMessage("visitor", "Do you allow dogs?"), HistoryMessage("ai", "Yes.")]

    await run(
        ScriptedChatModel(),
        searcher,
        message="how much is it?",
        history=history,
        config=AgentConfig(),
    )

    assert searcher.queries == ["Do you allow dogs? how much is it?"]


@pytest.mark.parametrize(
    ("message", "follow_up"),
    [
        ("how much is it?", True),
        ("and for kids?", True),
        ("What time is breakfast?", False),
        ("Is there parking at the hotel for my electric car overnight?", False),
    ],
)
def test_only_messages_that_lean_on_the_last_one_are_follow_ups(
    message: str, follow_up: bool
) -> None:
    assert is_follow_up(message, AgentConfig()) is follow_up


async def test_a_failed_search_before_the_model_does_not_stop_the_answer() -> None:
    model = ScriptedChatModel().script("Happy to help.")

    events = await run(model, FakeSearcher(fail=True), config=AgentConfig())

    assert finished(events).text == "Happy to help."
