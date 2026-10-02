"""Shared eval plumbing: load cases, set up the real model and index, run a turn, write tables."""

import asyncio
import json
import os
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from harness import AgentConfig, HistoryMessage, TurnInput, run_turn
from harness.contracts import (
    AgentSpec,
    HandoffChecked,
    HandoffRequested,
    TurnFailed,
    TurnFinished,
)
from harness.decisions import Decider
from harness.knowledge import KnowledgeIndex, OpenAIEmbedder
from harness.models import build_chat_model
from harness.sample import CACHE, knowledge_folder, load_agent

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"


def load_cases(name: str) -> list[dict[str, Any]]:
    lines = (EVALS / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def history_of(case: dict[str, Any]) -> list[HistoryMessage]:
    return [HistoryMessage(role, text) for role, text in case.get("history", [])]


@dataclass
class Setup:
    agent: AgentSpec
    index: KnowledgeIndex
    model: BaseChatModel
    model_name: str


async def setup() -> Setup:
    load_dotenv()
    index = await KnowledgeIndex.build(knowledge_folder(), OpenAIEmbedder(), cache_dir=CACHE)
    model = build_chat_model()
    return Setup(load_agent(), index, model, str(getattr(model, "model_name", "unknown")))


@dataclass
class TurnRecord:
    text: str = ""
    failed: str | None = None
    handoff: HandoffRequested | None = None
    check: HandoffChecked | None = None
    latency_ms: int = 0
    first_token_ms: int | None = None
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    searches: int = 0
    steps: list[str] = field(default_factory=list)


async def run_case(
    setup: Setup,
    case: dict[str, Any],
    config: AgentConfig,
    decider: Decider | None = None,
) -> TurnRecord:
    record = TurnRecord()
    turn = TurnInput(agent=setup.agent, message=case["message"], history=history_of(case))
    started = time.perf_counter()
    async for event in run_turn(
        turn, setup.index, model=setup.model, config=config, decider=decider
    ):
        if isinstance(event, HandoffRequested):
            record.handoff = event
        elif isinstance(event, HandoffChecked):
            record.check = event
        elif isinstance(event, TurnFinished | TurnFailed):
            steps = event.steps
            record.steps = [f"{step.label}: {step.detail}" for step in steps]
            record.searches = sum(
                step.label in {"search before the model", "search_knowledge"} for step in steps
            )
            if event.usage is not None:
                record.model_calls = event.usage.model_calls
                record.tokens_in = event.usage.tokens_in
                record.tokens_out = event.usage.tokens_out
            if isinstance(event, TurnFinished):
                record.text = event.text
                record.latency_ms = event.latency_ms
                record.first_token_ms = event.first_token_ms
            else:
                record.failed = event.code
                record.latency_ms = round((time.perf_counter() - started) * 1000)
    return record


async def gather_limited[T](
    jobs: Sequence[Callable[[], Awaitable[T]]], limit: int, label: str = ""
) -> list[T]:
    """Runs jobs a few at a time, printing progress."""
    gate = asyncio.Semaphore(limit)
    done = 0

    async def one(job: Callable[[], Awaitable[T]]) -> T:
        nonlocal done
        async with gate:
            result = await job()
        done += 1
        print(f"\r{label} {done}/{len(jobs)}", end="", flush=True)
        return result

    results = await asyncio.gather(*(one(job) for job in jobs))
    print()
    return list(results)


def percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(int(round(share * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[position]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_results(name: str, markdown: str, data: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
    (RESULTS / f"{name}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    # Windows consoles default to a code page that can't print every character.
    print(markdown.encode("ascii", "replace").decode() if os.name == "nt" else markdown)
    print(f"\nSaved evals/results/{name}.md and .json")


def jev_key() -> str:
    key = os.environ.get("JEV_API_KEY", "")
    if not key:
        raise SystemExit("JEV_API_KEY is not set: the handoff eval needs the Jev classifier.")
    return key
