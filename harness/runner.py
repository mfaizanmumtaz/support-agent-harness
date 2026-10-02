"""`run_turn`: one visitor message in, a stream of `TurnEvent`s out.

The LangChain agent runs in its own task and hands events over through a
queue. That keeps the whole-turn time limit safe to enforce (the agent is
cancelled, never the caller) and lets tool events arrive as they happen.

One turn, in order:
1. Start the handoff check (if on) and search the knowledge, side by side.
2. In enforce mode, wait for the check (bounded), and act on a sure verdict.
3. Run the agent: the model answers from the knowledge already in its prompt,
   or calls `search_knowledge` / `handoff_to_human` itself.
4. Stream the reply, then finish with usage, sources and a step-by-step trace.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage

from harness.checks import ForceHandoff, describe, failed_check, run_check, should_check
from harness.config import AgentConfig
from harness.context import KnowledgeSearcher, TurnContext
from harness.contracts import (
    HandoffChecked,
    HistoryMessage,
    Hit,
    KnowledgeSearched,
    Step,
    TextDelta,
    TurnEvent,
    TurnFailed,
    TurnFinished,
    TurnInput,
    Usage,
)
from harness.decisions import Decider
from harness.prompts import PRESEARCH_EMPTY, PRESEARCH_FOUND, TEAMMATE_MARKER, turn_prompt
from harness.tokens import count_tokens
from harness.tools import format_hits, tools_for

logger = logging.getLogger(__name__)

# The node LangChain's agent runs the model in.
MODEL_NODE = "model"
TOOLS_NODE = "tools"
FAILURE_MESSAGES = {
    "timeout": "The assistant took too long to answer.",
    "model_error": "The AI service couldn't be reached.",
    "empty_reply": "The assistant didn't produce a reply.",
    "engine_error": "Something went wrong while answering.",
}


async def run_turn(
    turn: TurnInput,
    searcher: KnowledgeSearcher,
    *,
    model: BaseChatModel,
    config: AgentConfig | None = None,
    decider: Decider | None = None,
) -> AsyncIterator[TurnEvent]:
    """Answers one visitor message. Always ends with `TurnFinished` or `TurnFailed`.

    Never raises for model, network or search trouble: those become
    `TurnFailed`, and the caller decides what the visitor sees. `decider` runs
    the handoff check when `config.handoff_check` asks for it.
    """
    config = config or AgentConfig()
    run = _Run(turn, searcher, model, config, decider)
    async for event in run.events():
        yield event


class _Run:
    def __init__(
        self,
        turn: TurnInput,
        searcher: KnowledgeSearcher,
        model: BaseChatModel,
        config: AgentConfig,
        decider: Decider | None = None,
    ) -> None:
        self.turn = turn
        self.model = model
        self.config = config
        self.decider = decider
        self.context = TurnContext(
            agent=turn.agent,
            searcher=searcher,
            config=config,
            team_available=turn.team_available,
        )
        self.queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        self.started = time.perf_counter()
        self.first_token_ms: int | None = None
        self.streamed: list[str] = []
        self.reply_texts: list[str] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.model_calls = 0
        self.last_mark = self.started
        self.last_message_id: str | None = None
        self.forwarded = 0
        self.check_started = self.started

    async def events(self) -> AsyncIterator[TurnEvent]:
        task = asyncio.create_task(self._drive())
        deadline = self.started + self.config.turn_timeout_seconds
        try:
            while True:
                remaining = deadline - time.perf_counter()
                try:
                    event = await asyncio.wait_for(self.queue.get(), timeout=max(remaining, 0))
                except TimeoutError:
                    task.cancel()
                    yield self._failed("timeout")
                    return
                if event is None:
                    return
                yield event
        finally:
            if not task.done():
                task.cancel()

    async def _drive(self) -> None:
        """Runs the agent and puts every event on the queue, then `None`."""
        try:
            await self._stream()
            await self.queue.put(self._finish())
        except asyncio.CancelledError:
            raise
        except openai.APITimeoutError:
            await self.queue.put(self._failed("timeout"))
        except (openai.APIError, httpx.HTTPError):
            logger.warning("chat model call failed", exc_info=True)
            await self.queue.put(self._failed("model_error"))
        except Exception:
            logger.exception("AI turn failed")
            await self.queue.put(self._failed("engine_error"))
        await self.queue.put(None)

    async def _stream(self) -> None:
        # The check runs while the knowledge is searched, so it rarely adds a wait.
        check = self._start_check()
        try:
            await self._answer(check)
        finally:
            if check is not None and not check.done():
                check.cancel()

    async def _answer(self, check: asyncio.Task[HandoffChecked] | None) -> None:
        if self.config.presearch:
            await self._presearch()
            await self._forward_tool_events()
        if check is not None and self.config.handoff_check == "enforce":
            await self._finish_check(check)
            self.last_mark = time.perf_counter()
        agent: Any = create_agent(
            self.model,
            tools=tools_for(self.turn.agent),
            middleware=self._middleware(),
            context_schema=TurnContext,
        )
        stream = agent.astream(
            {"messages": self._messages()},
            context=self.context,
            stream_mode=["messages", "updates"],
        )
        async for mode, data in stream:
            if mode == "messages":
                await self._on_token(*data)
            elif mode == "updates":
                self._on_update(data)
            await self._forward_tool_events()
        if check is not None and self.config.handoff_check == "shadow":
            await self._finish_check(check)
            await self._forward_tool_events()

    def _start_check(self) -> asyncio.Task[HandoffChecked] | None:
        if self.decider is None or not should_check(self.turn, self.config):
            return None
        self.check_started = time.perf_counter()
        return asyncio.create_task(run_check(self.turn, self.decider, self.config))

    async def _finish_check(self, task: asyncio.Task[HandoffChecked]) -> None:
        """Waits what is left of the check's time. A failed or late check changes nothing."""
        remaining = self.config.handoff_check_timeout_seconds - (
            time.perf_counter() - self.check_started
        )
        try:
            check = await asyncio.wait_for(task, timeout=max(remaining, 0))
        except TimeoutError:
            check = failed_check(self.config, "timeout", _ms(self.check_started))
        except Exception as exc:
            logger.warning("handoff check failed", exc_info=True)
            check = failed_check(self.config, type(exc).__name__, _ms(self.check_started))
        self.context.events.append(check)
        self.context.steps.append(Step("handoff check", describe(check), check.duration_ms))
        if check.mode == "enforce" and check.verdict is not None:
            self.context.handoff_verdict = check

    async def _presearch(self) -> None:
        """Searches before the first model call, so it can usually answer straight away.

        Without this the model spends a whole call deciding to search. A failed
        search here is harmless: the model can still search for itself.
        """
        query = presearch_query(self.turn, self.config)
        started = time.perf_counter()
        try:
            result = await self.context.searcher.search(query)
        except Exception as exc:
            logger.warning("search before the model failed", exc_info=True)
            elapsed = _ms(started)
            self.context.events.append(
                KnowledgeSearched(query, [], 0, 0, elapsed, error=type(exc).__name__)
            )
            self.context.steps.append(Step("search before the model", "failed", elapsed))
            return
        elapsed = _ms(started)
        self.context.events.append(
            KnowledgeSearched(
                query=result.query,
                hits=result.hits,
                candidates=result.candidates,
                below_threshold=result.below_threshold,
                duration_ms=elapsed,
            )
        )
        self.context.steps.append(
            Step("search before the model", f"{query!r}: {len(result.hits)} found", elapsed)
        )
        self.context.presearch = (
            PRESEARCH_FOUND.format(knowledge=format_hits(result.hits))
            if result.hits
            else PRESEARCH_EMPTY
        )
        self.last_mark = time.perf_counter()

    def _middleware(self) -> list[Any]:
        middleware: list[Any] = [
            turn_prompt,
            ModelCallLimitMiddleware(run_limit=self.config.model_call_limit, exit_behavior="end"),
            ToolCallLimitMiddleware(
                tool_name="search_knowledge", run_limit=self.config.search_limit
            ),
        ]
        if self.turn.agent.handoff_enabled:
            middleware.append(ToolCallLimitMiddleware(tool_name="handoff_to_human", run_limit=1))
            if self.config.handoff_check == "enforce":
                middleware.append(ForceHandoff())
        return middleware

    def _messages(self) -> list[AnyMessage]:
        message = self.turn.message.strip()[: self.config.max_message_chars]
        history = trim_history(self.turn.history, self.config.history_token_budget)
        return [*(_as_message(item) for item in history), HumanMessage(content=message)]

    async def _on_token(self, chunk: Any, metadata: dict[str, Any]) -> None:
        if metadata.get("langgraph_node") != MODEL_NODE or not isinstance(chunk, AIMessageChunk):
            return
        text = _text_of(chunk.content)
        if not text:
            return
        # A second model call in the same turn starts a new paragraph.
        if self.streamed and chunk.id and self.last_message_id and chunk.id != self.last_message_id:
            await self._emit_text("\n\n")
        self.last_message_id = chunk.id or self.last_message_id
        await self._emit_text(text)

    async def _emit_text(self, text: str) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = _ms(self.started)
        self.streamed.append(text)
        await self.queue.put(TextDelta(text))

    def _on_update(self, update: dict[str, Any]) -> None:
        for node, state in update.items():
            if node == TOOLS_NODE:
                # Tool steps time themselves; the next model call starts now.
                self.last_mark = time.perf_counter()
            if node != MODEL_NODE or not isinstance(state, dict):
                continue
            for message in state.get("messages", []):
                if isinstance(message, AIMessage):
                    self._count_model_call(message)

    def _count_model_call(self, message: AIMessage) -> None:
        self.model_calls += 1
        usage: dict[str, Any] = dict(message.usage_metadata or {})
        self.tokens_in += int(usage.get("input_tokens", 0))
        self.tokens_out += int(usage.get("output_tokens", 0))
        if text := _text_of(message.content).strip():
            self.reply_texts.append(text)
        calls = [call["name"] for call in message.tool_calls]
        detail = f"asked for {', '.join(calls)}" if calls else "wrote the reply"
        now = time.perf_counter()
        self.context.steps.append(
            Step(f"model call {self.model_calls}", detail, _ms(self.last_mark, now))
        )
        self.last_mark = now

    async def _forward_tool_events(self) -> None:
        pending = self.context.events[self.forwarded :]
        self.forwarded = len(self.context.events)
        for event in pending:
            await self.queue.put(event)

    def _finish(self) -> TurnFinished | TurnFailed:
        text = "".join(self.streamed).strip()
        if not text and self.reply_texts:
            # The model answered without streaming (some providers, and the test fake).
            text = "\n\n".join(self.reply_texts)
            if self.first_token_ms is None:
                self.first_token_ms = _ms(self.started)
        if not text:
            return self._failed("empty_reply")
        return TurnFinished(
            text=text,
            sources=self._sources(),
            usage=self._usage(),
            latency_ms=_ms(self.started),
            first_token_ms=self.first_token_ms,
            steps=list(self.context.steps),
        )

    def _failed(self, code: str) -> TurnFailed:
        return TurnFailed(
            code=code,  # type: ignore[arg-type]
            message=FAILURE_MESSAGES[code],
            usage=self._usage(),
            steps=list(self.context.steps),
        )

    def _usage(self) -> Usage:
        name = getattr(self.model, "model_name", None) or getattr(self.model, "model", "unknown")
        return Usage(
            model=str(name),
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model_calls=self.model_calls,
        )

    def _sources(self) -> list[Hit]:
        """Every chunk the AI was shown, each once with its best score."""
        best: dict[Any, Hit] = {}
        for event in self.context.events:
            if isinstance(event, KnowledgeSearched):
                for hit in event.hits:
                    if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                        best[hit.chunk_id] = hit
        return sorted(best.values(), key=lambda hit: hit.score, reverse=True)


def presearch_query(turn: TurnInput, config: AgentConfig) -> str:
    """The visitor's message, plus their previous one when this is a follow-up.

    "how much is it?" means nothing on its own, so it is searched together with
    the question before it. "What time is check-in?" is short but complete, and
    adding the previous question would only muddy the search.
    """
    message = turn.message.strip()[: config.max_message_chars]
    if not config.presearch_include_previous or not is_follow_up(message, config):
        return message
    previous = next((item.text for item in reversed(turn.history) if item.role == "visitor"), None)
    return f"{previous.strip()} {message}" if previous else message


_FOLLOW_UP_STARTS = ("and ", "also ", "but ", "what about", "how about", "same for", "plus ")
_REFERRING_WORDS = {
    "it", "its", "it's", "that", "this", "those", "these", "they", "them", "there", "one", "ones",
}  # fmt: skip


def is_follow_up(message: str, config: AgentConfig) -> bool:
    """A short message that leans on the one before: "and for kids?", "how much is it?"."""
    words = [word.strip("?!.,;:\"'").lower() for word in message.split()]
    if not words or len(words) > config.presearch_short_message_words:
        return False
    return message.lower().startswith(_FOLLOW_UP_STARTS) or any(
        word in _REFERRING_WORDS for word in words
    )


def trim_history(history: list[HistoryMessage], budget: int) -> list[HistoryMessage]:
    """The most recent messages that fit in `budget` tokens, in their original order."""
    kept: list[HistoryMessage] = []
    used = 0
    for item in reversed(history):
        cost = count_tokens(_text_for_model(item))
        if used + cost > budget:
            break
        kept.append(item)
        used += cost
    return list(reversed(kept))


def _text_for_model(item: HistoryMessage) -> str:
    """What the model actually reads for this message, marker and all."""
    return f"{TEAMMATE_MARKER} {item.text}" if item.role == "human" else item.text


def _as_message(item: HistoryMessage) -> AnyMessage:
    if item.role == "visitor":
        return HumanMessage(content=item.text)
    if item.role == "human":
        # A teammate spoke for the business, so this is our side of the
        # conversation, but it isn't the model's own words. The marker has to
        # live in the text: `name="teammate"` would be dropped before sending.
        return AIMessage(content=_text_for_model(item))
    return AIMessage(content=item.text)


def _text_of(content: Any) -> str:
    """Message content is either text or a list of blocks; only text blocks count."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") in (None, "text")
        )
    return ""


def _ms(start: float, end: float | None = None) -> int:
    return round(((end if end is not None else time.perf_counter()) - start) * 1000)
