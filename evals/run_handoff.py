"""Does the agent hand off when it should, and only then?

Each labelled message runs through four setups:
- LLM only: the model decides by itself whether to call `handoff_to_human`.
- Jev only: the fast classifier's verdict on its own (threshold 0.85), no LLM.
- LLM + Jev (enforce, 0.85 / 0.95): the classifier runs beside the search, and
  a verdict at or over the threshold makes the model hand off.

The report also sweeps the threshold over Jev's saved scores, to show the
trade-off between missed and wrong handoffs without more API calls.

    uv run python -m evals.run_handoff
"""

import asyncio
import functools
from collections import defaultdict
from typing import Any

from evals.common import (
    Setup,
    gather_limited,
    history_of,
    jev_key,
    load_cases,
    mean,
    percentile,
    run_case,
    setup,
    table,
    write_results,
)
from harness import AgentConfig, TurnInput
from harness.checks import run_check
from harness.decisions import JevDecider

ENFORCE = {
    "LLM + Jev (enforce, 0.85)": AgentConfig(handoff_check="enforce"),
    "LLM + Jev (enforce, 0.95)": AgentConfig(handoff_check="enforce", handoff_check_threshold=0.95),
}
SETUPS = ["LLM only", "Jev only", *ENFORCE]
SWEEP = (0.5, 0.7, 0.85, 0.9, 0.95)


async def main() -> None:
    env = await setup()
    decider = JevDecider(jev_key())
    cases = load_cases("handoff_cases.jsonl")
    llm_only = AgentConfig()

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        plain = await run_case(env, case, llm_only)
        turn = TurnInput(agent=env.agent, message=case["message"], history=history_of(case))
        check = await run_check(turn, decider, ENFORCE["LLM + Jev (enforce, 0.85)"])
        combined = {
            name: await run_case(env, case, config, decider) for name, config in ENFORCE.items()
        }
        return {
            "id": case["id"],
            "category": case["category"],
            "message": case["message"],
            "expected": case["expected"] == "handoff",
            "LLM only": plain.handoff is not None,
            "Jev only": check.verdict is not None,
            **{name: record.handoff is not None for name, record in combined.items()},
            "jev_verdict": check.verdict,
            "jev_signals": check.signals,
            "jev_rules": check.rules,
            "jev_ms": check.duration_ms,
            "jev_cost": check.cost,
            "llm_only_reply": plain.text,
            "llm_only_failed": plain.failed,
            "llm_only_latency_ms": plain.latency_ms,
            "combined": {
                name: {"reply": r.text, "failed": r.failed, "latency_ms": r.latency_ms}
                for name, r in combined.items()
            },
        }

    results = await gather_limited(
        [functools.partial(evaluate, case) for case in cases], limit=6, label="cases"
    )
    await decider.aclose()
    write_results("handoff", report(results, env), {"model": env.model_name, "cases": results})


def report(results: list[dict[str, Any]], env: Setup) -> str:
    rows = []
    for name in SETUPS:
        tp = sum(r[name] and r["expected"] for r in results)
        fp = sum(r[name] and not r["expected"] for r in results)
        fn = sum(not r[name] and r["expected"] for r in results)
        tn = sum(not r[name] and not r["expected"] for r in results)
        rows.append(
            [
                f"**{name}**",
                f"{(tp + tn) / len(results):.0%}",
                f"{tp / (tp + fn):.0%}" if tp + fn else "-",
                f"{tp / (tp + fp):.0%}" if tp + fp else "-",
                str(fn),
                str(fp),
            ]
        )
    summary = table(
        ["Setup", "Accuracy", "Recall", "Precision", "Missed handoffs", "Wrong handoffs"], rows
    )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    category_rows = [
        [category, str(len(items))]
        + [f"{sum(r[name] == r['expected'] for r in items)}/{len(items)}" for name in SETUPS]
        for category, items in by_category.items()
    ]
    categories = table(["Category", "Cases", *SETUPS], category_rows)

    jev_ms = [r["jev_ms"] for r in results]
    cost = sum(r["jev_cost"] for r in results)
    misses = [
        f"- `{r['id']}` ({'should' if r['expected'] else 'should not'} hand off): "
        f'"{r["message"]}" → LLM only: {"handoff" if r["LLM only"] else "no handoff"}, '
        f"Jev: {r['jev_verdict'] or 'no handoff'} (best score {best_score(r):.2f})"
        for r in results
        if any(r[name] != r["expected"] for name in SETUPS)
    ]
    enforced_latency = [r["combined"]["LLM + Jev (enforce, 0.85)"]["latency_ms"] for r in results]
    failed = sum(
        bool(r["llm_only_failed"]) or any(c["failed"] for c in r["combined"].values())
        for r in results
    )
    return "\n\n".join(
        [
            f"## Handoff eval ({len(results)} labelled messages, chat model `{env.model_name}`)",
            summary,
            "### By category",
            categories,
            "### The Jev check itself",
            table(
                ["Calls", "Median latency", "p90 latency", "Total cost", "Cost per 1,000 messages"],
                [
                    [
                        str(len(results)),
                        f"{percentile(jev_ms, 0.5):.0f} ms",
                        f"{percentile(jev_ms, 0.9):.0f} ms",
                        f"${cost:.4f}",
                        f"${cost / len(results) * 1000:.3f}",
                    ]
                ],
            ),
            f"Mean turn latency: LLM only {mean([r['llm_only_latency_ms'] for r in results]):.0f}"
            f" ms, LLM + Jev (enforce, 0.85) {mean(enforced_latency):.0f} ms. The check runs"
            " while the knowledge is searched, so it adds no wait."
            + (f" Failed turns: {failed}." if failed else ""),
            "### Jev threshold sweep (from the saved scores, no extra calls)",
            sweep(results),
            "### Where a setup got it wrong",
            "\n".join(misses) or "None.",
        ]
    )


def best_score(result: dict[str, Any]) -> float:
    """Jev's strongest reason to hand off: the person signal or the best rule."""
    rules = max(result["jev_rules"].values(), default=0.0)
    return max(float(result["jev_signals"].get("visitor_asked", 0.0)), float(rules))


def sweep(results: list[dict[str, Any]]) -> str:
    rows = []
    for threshold in SWEEP:
        predicted = [best_score(r) >= threshold for r in results]
        expected = [r["expected"] for r in results]
        missed = sum(e and not p for p, e in zip(predicted, expected, strict=True))
        wrong = sum(p and not e for p, e in zip(predicted, expected, strict=True))
        accuracy = (len(results) - missed - wrong) / len(results)
        rows.append([f"{threshold:.2f}", f"{accuracy:.0%}", str(missed), str(wrong)])
    return table(["Threshold", "Jev-only accuracy", "Missed", "Wrong"], rows)


if __name__ == "__main__":
    asyncio.run(main())
