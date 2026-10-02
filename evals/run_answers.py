"""Searching before the model vs letting the model decide when to search.

Each question runs twice: once with retrieval before the model (the
knowledge is already in the prompt), once tool-only (the model must call
`search_knowledge` itself). Cases run one at a time so latencies are fair.

    uv run python -m evals.run_answers
"""

import asyncio
import json
import sys
from typing import Any

from evals.common import (
    RESULTS,
    load_cases,
    mean,
    percentile,
    run_case,
    setup,
    table,
    write_results,
)
from harness import AgentConfig

SETUPS = {
    "Search before the model": AgentConfig(presearch=True),
    "Tool-only (model decides)": AgentConfig(presearch=False),
}


def correct(case: dict[str, Any], reply: str, failed: str | None = None) -> bool:
    """Every expected group has at least one of its phrasings in the reply."""
    text = normalized(reply)
    return not failed and all(
        any(normalized(option) in text for option in group) for group in case["expect"]
    )


def normalized(text: str) -> str:
    # Models write typographic apostrophes and dashes ("can’t"); the phrases don't.
    return text.lower().replace("’", "'").replace("‘", "'").replace("–", "-")


async def main() -> None:
    env = await setup()
    cases = load_cases("answer_cases.jsonl")
    # One warm-up call, so the first measured case doesn't pay for the connection.
    await run_case(env, {"message": "hi"}, AgentConfig())
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in SETUPS}
    for number, case in enumerate(cases, start=1):
        print(f"\rcases {number}/{len(cases)}", end="", flush=True)
        for name, config in SETUPS.items():
            record = await run_case(env, case, config)
            results[name].append(
                {
                    "id": case["id"],
                    "message": case["message"],
                    "correct": correct(case, record.text, record.failed),
                    "reply": record.text,
                    "failed": record.failed,
                    "latency_ms": record.latency_ms,
                    "first_token_ms": record.first_token_ms,
                    "model_calls": record.model_calls,
                    "searches": record.searches,
                    "tokens_in": record.tokens_in,
                    "tokens_out": record.tokens_out,
                }
            )
    print()
    write_results(
        "answers",
        report(results, len(cases), env.model_name),
        {"model": env.model_name, "results": results},
    )


def report(results: dict[str, list[dict[str, Any]]], count: int, model: str) -> str:
    rows = []
    for name, items in results.items():
        first = [float(r["first_token_ms"]) for r in items if r["first_token_ms"] is not None]
        rows.append(
            [
                f"**{name}**",
                f"{sum(r['correct'] for r in items)}/{len(items)}",
                f"{mean([r['model_calls'] for r in items]):.2f}",
                f"{percentile(first, 0.5):.0f} ms",
                f"{percentile([float(r['latency_ms']) for r in items], 0.5):.0f} ms",
                f"{percentile([float(r['latency_ms']) for r in items], 0.9):.0f} ms",
                f"{mean([r['tokens_in'] for r in items]):.0f}",
            ]
        )
    misses = [
        f'- {name}: `{r["id"]}` "{r["message"]}" → {r["reply"][:160]!r}'
        for name, items in results.items()
        for r in items
        if not r["correct"]
    ]
    return "\n\n".join(
        [
            f"## Answer eval ({count} questions, chat model `{model}`)",
            table(
                [
                    "Setup",
                    "Correct",
                    "Model calls / answer",
                    "Median first words",
                    "Median full reply",
                    "p90 full reply",
                    "Input tokens / answer",
                ],
                rows,
            ),
            "Correct = the reply contains the fact from the knowledge base (or, for the 3 "
            "questions the knowledge doesn't cover, says it doesn't know). Checked by phrase "
            "match, so it is a floor, not a judge.",
            "### Misses",
            "\n".join(misses) or "None.",
        ]
    )


def rescore() -> None:
    """Grades the saved replies again (after a grader fix) without calling the model."""
    cases = {case["id"]: case for case in load_cases("answer_cases.jsonl")}
    saved = json.loads((RESULTS / "answers.json").read_text(encoding="utf-8"))
    results = saved["results"]
    for items in results.values():
        for item in items:
            item["correct"] = correct(cases[item["id"]], item["reply"], item["failed"])
    write_results("answers", report(results, len(cases), saved["model"]), saved)


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        rescore()
    else:
        asyncio.run(main())
