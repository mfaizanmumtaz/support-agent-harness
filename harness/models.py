"""The chat model: OpenAI in production, a scripted fake in tests.

Build one model per process and reuse it, so its connection stays open
between replies instead of being set up again every time.
"""

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from harness.config import AgentConfig

DEFAULT_CHAT_MODEL = "gpt-6-luna"


def build_chat_model(config: AgentConfig | None = None, model: str | None = None) -> BaseChatModel:
    """OpenAI through LangChain, streaming, with usage reported on the last chunk."""
    from langchain_openai import ChatOpenAI

    config = config or AgentConfig()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return ChatOpenAI(
        model=model or os.environ.get("CHAT_MODEL") or DEFAULT_CHAT_MODEL,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        max_tokens=config.max_output_tokens,
        streaming=True,
        # Token counts arrive with the last streamed piece, for usage and cost.
        stream_usage=True,
        # The Responses API: current models refuse tool calls over Chat Completions.
        use_responses_api=True,
    )


class ScriptedChatModel(BaseChatModel):
    """A chat model for tests that replays scripted replies, in order.

    A reply is an `AIMessage` (with or without tool calls) or an exception to
    raise, which is how a test simulates the provider failing. When the script
    runs out it answers with `default`. Replies without tool calls are streamed
    word by word, like the real model.
    """

    replies: list[AIMessage | BaseException] = Field(default_factory=list)
    default: str = "This is a scripted answer."
    model_name: str = "scripted-model"
    # What each call was sent, and which tools were offered, for assertions.
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)
    # The tool each call was made to use (None: the model's own choice).
    tool_choices: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools = [getattr(tool, "name", str(tool)) for tool in tools]
        self.tool_choices.append(kwargs.get("tool_choice"))
        return self

    def script(self, *replies: AIMessage | BaseException | str) -> "ScriptedChatModel":
        """Queues replies; plain strings become text answers."""
        self.replies.extend(AIMessage(content=r) if isinstance(r, str) else r for r in replies)
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else AIMessage(content=self.default)
        if isinstance(reply, BaseException):
            raise reply
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        return reply.model_copy(
            update={"usage_metadata": usage, "id": reply.id or f"scripted-{len(self.calls)}"}
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from _chunks(self._next(messages))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in _chunks(self._next(messages)):
            if run_manager is not None and isinstance(chunk.message.content, str):
                await run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
            yield chunk


def _chunks(message: AIMessage) -> Iterator[ChatGenerationChunk]:
    """A reply as streamed pieces: word by word, tool calls and usage on the last piece."""
    text = message.content if isinstance(message.content, str) else ""
    words = text.split(" ") if text else []
    pieces = [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)] or [""]
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=piece,
                id=message.id,
                tool_call_chunks=(
                    [
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                        }
                        for index, call in enumerate(message.tool_calls)
                    ]
                    if last
                    else []
                ),
                usage_metadata=message.usage_metadata if last else None,
            )
        )
