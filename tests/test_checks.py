"""The handoff check beside each turn, with a scripted decider in place of Jev."""

from dataclasses import replace
from typing import Any

from harness import (
    AgentConfig,
    AgentSpec,
    HandoffChecked,
    HandoffRequested,
    HistoryMessage,
    TurnEvent,
    TurnInput,
    run_turn,
)
from harness.checks import check_state, rules_of
from harness.decisions import ScriptedDecider, YesNo
from harness.models import ScriptedChatModel
from tests.helpers import AGENT, FakeSearcher, call, finished
from tests.helpers import system_prompt as _system_prompt

RULES_AGENT = replace(
    AGENT, handoff_rules="- Refund disputes over $500.\n* Enterprise pricing\n\n2) Legal threats"
)


async def run(
    model: ScriptedChatModel,
    decider: ScriptedDecider | None,
    mode: str,
    *,
    agent: AgentSpec = RULES_AGENT,
    message: str = "I want my $900 back, this is a dispute",
    **settings: Any,
) -> list[TurnEvent]:
    turn = TurnInput(
        agent=agent,
        message=message,
        history=[HistoryMessage("visitor", "hi"), HistoryMessage("ai", "Hello!")],
    )
    config = AgentConfig(presearch=False, handoff_check=mode, **settings)
    return [
        event
        async for event in run_turn(
            turn, FakeSearcher(), model=model, config=config, decider=decider
        )
    ]


def checked(events: list[TurnEvent]) -> HandoffChecked:
    [check] = [event for event in events if isinstance(event, HandoffChecked)]
    return check


# ---------------------------------------------------------------- what is asked


def test_each_line_of_the_rules_is_its_own_rule_without_its_bullet() -> None:
    assert rules_of(RULES_AGENT, limit=10) == [
        "Refund disputes over $500.",
        "Enterprise pricing",
        "Legal threats",
    ]
    assert rules_of(RULES_AGENT, limit=2) == ["Refund disputes over $500.", "Enterprise pricing"]


def test_the_check_reads_the_latest_message_with_a_little_history() -> None:
    turn = TurnInput(
        agent=AGENT,
        message="and now?",
        history=[HistoryMessage("visitor", f"message {i}") for i in range(10)]
        + [HistoryMessage("human", "Jo here, looking into it")],
    )

    state = check_state(turn, AgentConfig(handoff_check_history_messages=2))

    assert state["latest_visitor_message"] == "and now?"
    assert state["earlier_messages"] == "Visitor: message 9\nTeammate: Jo here, looking into it"


async def test_every_rule_and_signal_is_asked_in_one_call() -> None:
    decider = ScriptedDecider()

    await run(ScriptedChatModel(), decider, "shadow")

    [(state, questions)] = decider.calls
    assert state["business"] == "Northwind Hotel"
    assert list(questions) == [
        "visitor_asked",
        "frustrated",
        "sensitive",
        "rule_1",
        "rule_2",
        "rule_3",
    ]
    assert all(isinstance(question, YesNo) for question in questions.values())
    rule = questions["rule_1"]
    assert isinstance(rule, YesNo) and "Refund disputes over $500." in rule.instructions


# ---------------------------------------------------------------- when it runs


async def test_off_never_asks() -> None:
    decider = ScriptedDecider()

    events = await run(ScriptedChatModel(), decider, "off")

    assert decider.calls == []
    assert not any(isinstance(event, HandoffChecked) for event in events)


async def test_no_check_when_the_business_has_handoff_off() -> None:
    decider = ScriptedDecider()
    no_handoff = replace(RULES_AGENT, handoff_enabled=False)

    await run(ScriptedChatModel(), decider, "enforce", agent=no_handoff)

    assert decider.calls == []


# ---------------------------------------------------------------- shadow


async def test_shadow_records_what_would_happen_and_changes_nothing() -> None:
    decider = ScriptedDecider(answers={"rule_1": 0.97, "frustrated": 0.6})
    model = ScriptedChatModel().script("Let me look into that refund for you.")

    events = await run(model, decider, "shadow")

    check = checked(events)
    assert check.mode == "shadow"
    assert check.verdict == "rule_matched"
    assert check.rule == "Refund disputes over $500."
    assert check.signals == {"visitor_asked": 0.0, "frustrated": 0.6, "sensitive": 0.0}
    assert check.rules["Refund disputes over $500."] == 0.97
    # The AI answered as it would have without the check.
    assert model.tool_choices == [None]
    assert "A check of this message" not in _system_prompt(model)
    assert not any(isinstance(event, HandoffRequested) for event in events)
    result = finished(events)
    assert result.text == "Let me look into that refund for you."
    [step] = [step for step in result.steps if step.label == "handoff check"]
    assert step.detail.startswith("would hand off: rule_matched")


async def test_below_the_threshold_is_no_verdict() -> None:
    decider = ScriptedDecider(answers={"visitor_asked": 0.8, "rule_2": 0.84})

    events = await run(ScriptedChatModel(), decider, "shadow", handoff_check_threshold=0.85)

    assert checked(events).verdict is None


# ---------------------------------------------------------------- enforce


async def test_enforce_makes_the_first_call_hand_off_then_the_ai_replies() -> None:
    decider = ScriptedDecider(answers={"visitor_asked": 0.99})
    model = ScriptedChatModel().script(
        call("handoff_to_human", reason="visitor_asked", summary="Wants a person."),
        "Connecting you with the team now.",
    )

    events = await run(model, decider, "enforce", message="let me talk to a human")

    assert checked(events).verdict == "visitor_asked"
    assert model.tool_choices == ["handoff_to_human", None]
    assert "the visitor asked for a person" in _system_prompt(model)
    handoff = next(e for e in events if isinstance(e, HandoffRequested))
    assert handoff.reason == "visitor_asked"
    assert finished(events).text == "Connecting you with the team now."


async def test_enforce_with_steps_before_a_handoff_tells_rather_than_forces() -> None:
    with_steps = replace(
        RULES_AGENT,
        handoff_procedure="Ask for the booking number first.",
    )
    decider = ScriptedDecider(answers={"rule_1": 0.99})
    model = ScriptedChatModel().script("Sure, what's your booking number?")

    await run(model, decider, "enforce", agent=with_steps)

    assert model.tool_choices == [None]
    prompt = _system_prompt(model)
    assert 'this business handoff rule applies: "Refund disputes over $500."' in prompt
    assert "Follow the business's steps before handing off" in prompt


async def test_enforce_without_a_verdict_changes_nothing() -> None:
    model = ScriptedChatModel().script("Happy to help.")

    await run(model, ScriptedDecider(), "enforce")

    assert model.tool_choices == [None]
    assert "A check of this message" not in _system_prompt(model)


# ---------------------------------------------------------------- failures


async def test_a_failed_check_is_recorded_and_the_turn_goes_on() -> None:
    model = ScriptedChatModel().script("Happy to help.")

    events = await run(model, ScriptedDecider(fail=True), "enforce")

    check = checked(events)
    assert (check.error, check.verdict) == ("DecisionError", None)
    assert model.tool_choices == [None]
    assert finished(events).text == "Happy to help."


async def test_a_slow_check_is_dropped_rather_than_keep_the_visitor_waiting() -> None:
    decider = ScriptedDecider(answers={"visitor_asked": 0.99}, delay_seconds=5)
    model = ScriptedChatModel().script("Happy to help.")

    events = await run(model, decider, "enforce", handoff_check_timeout_seconds=0.05)

    assert checked(events).error == "timeout"
    assert model.tool_choices == [None]
    assert finished(events).text == "Happy to help."
