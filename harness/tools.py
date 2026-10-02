"""The two actions the model can take: search the knowledge, or hand off to a person."""

import logging
import time
from typing import Any, cast

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from harness.context import TurnContext
from harness.contracts import (
    AgentSpec,
    HandoffReason,
    HandoffRequested,
    Hit,
    KnowledgeSearched,
    Step,
)
from harness.prompts import KNOWLEDGE_END, KNOWLEDGE_START

logger = logging.getLogger(__name__)

NOTHING_FOUND = (
    "No relevant knowledge found for this search. If rephrasing won't help, tell the "
    "visitor you don't have that information."
)
SEARCH_FAILED = (
    "Knowledge search is unavailable right now. Tell the visitor you can't check that at "
    "the moment."
)
TEAM_AVAILABLE = (
    "The team has been notified. Tell the visitor you're connecting them with a teammate "
    "who will reply here shortly. Don't promise a specific time."
)
TEAM_AWAY = (
    "The team is away, so nobody can take this chat. Tell the visitor, in their language "
    'and your own words, this message from the business: "{offline_message}" Then carry '
    "on helping them yourself."
)


@tool
async def search_knowledge(query: str, runtime: ToolRuntime[Any]) -> str:
    """Search the business's knowledge base: policies, products, prices, FAQs.

    Use it before answering any question about the business. Write `query` as a short,
    self-contained search, e.g. "refund policy for damaged items", not "what about that?".
    """
    context = cast(TurnContext, runtime.context)
    started = time.perf_counter()
    try:
        result = await context.searcher.search(query)
    except Exception as exc:
        # A search outage must not end the conversation: the model says it can't check.
        logger.exception("knowledge search failed")
        elapsed = _ms_since(started)
        context.events.append(KnowledgeSearched(query, [], 0, 0, elapsed, error=type(exc).__name__))
        context.steps.append(Step("search_knowledge", f"{query!r}: search failed", elapsed))
        return SEARCH_FAILED
    elapsed = _ms_since(started)
    context.events.append(
        KnowledgeSearched(
            result.query, result.hits, result.candidates, result.below_threshold, elapsed
        )
    )
    context.steps.append(Step("search_knowledge", f"{query!r}: {len(result.hits)} found", elapsed))
    return format_hits(result.hits) if result.hits else NOTHING_FOUND


@tool
async def handoff_to_human(reason: HandoffReason, summary: str, runtime: ToolRuntime[Any]) -> str:
    """Hand the conversation to a person on the business's team.

    Use it when the visitor asks for a human, or when the business's handoff rules say
    so. `reason` is why. `summary` is one or two sentences for the teammate: what the
    visitor needs and anything already tried.
    """
    context = cast(TurnContext, runtime.context)
    context.events.append(HandoffRequested(reason=reason, summary=summary.strip()))
    context.steps.append(Step("handoff_to_human", f"{reason}: {summary.strip()}", 0))
    if context.team_available:
        return TEAM_AVAILABLE
    return TEAM_AWAY.format(offline_message=context.agent.offline_message.strip())


def tools_for(agent: AgentSpec) -> list[BaseTool]:
    """Handoff is only offered when the business switched it on."""
    return [search_knowledge, handoff_to_human] if agent.handoff_enabled else [search_knowledge]


def format_hits(hits: list[Hit]) -> str:
    """Numbered snippets inside the knowledge markers, each saying where it came from."""
    blocks = [
        f"[{number}] ({hit.source_name}{f' › {hit.heading}' if hit.heading else ''})\n"
        f"{hit.content.strip()}"
        for number, hit in enumerate(hits, start=1)
    ]
    return f"{KNOWLEDGE_START}\n" + "\n\n".join(blocks) + f"\n{KNOWLEDGE_END}"


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
