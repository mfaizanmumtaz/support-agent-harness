## Handoff eval (60 labelled messages, chat model `gpt-6-luna`)

| Setup | Accuracy | Recall | Precision | Missed handoffs | Wrong handoffs |
|---|---|---|---|---|---|
| **LLM only** | 100% | 100% | 100% | 0 | 0 |
| **Jev only** | 90% | 89% | 94% | 4 | 2 |
| **LLM + Jev (enforce, 0.85)** | 97% | 100% | 95% | 0 | 2 |
| **LLM + Jev (enforce, 0.95)** | 100% | 100% | 100% | 0 | 0 |

### By category

| Category | Cases | LLM only | Jev only | LLM + Jev (enforce, 0.85) | LLM + Jev (enforce, 0.95) |
|---|---|---|---|---|---|
| asks for a person | 16 | 16/16 | 13/16 | 16/16 | 16/16 |
| rule: refund | 5 | 5/5 | 5/5 | 5/5 | 5/5 |
| rule: group booking | 5 | 5/5 | 5/5 | 5/5 | 5/5 |
| rule: staff complaint | 5 | 5/5 | 5/5 | 5/5 | 5/5 |
| rule: lost property | 5 | 5/5 | 4/5 | 5/5 | 5/5 |
| ordinary question | 10 | 10/10 | 10/10 | 10/10 | 10/10 |
| near miss | 14 | 14/14 | 12/14 | 12/14 | 14/14 |

### The Jev check itself

| Calls | Median latency | p90 latency | Total cost | Cost per 1,000 messages |
|---|---|---|---|---|
| 60 | 429 ms | 589 ms | $0.0014 | $0.024 |

Mean turn latency: LLM only 3565 ms, LLM + Jev (enforce, 0.85) 3478 ms. The check runs while the knowledge is searched, so it adds no wait.

### Jev threshold sweep (from the saved scores, no extra calls)

| Threshold | Jev-only accuracy | Missed | Wrong |
|---|---|---|---|
| 0.50 | 97% | 0 | 2 |
| 0.70 | 95% | 1 | 2 |
| 0.85 | 90% | 4 | 2 |
| 0.90 | 90% | 5 | 1 |
| 0.95 | 88% | 7 | 0 |

### Where a setup got it wrong

- `ask-10` (should hand off): "operator" → LLM only: handoff, Jev: no handoff (best score 0.70)
- `ask-12` (should hand off): "I don't want to talk to a bot." → LLM only: handoff, Jev: no handoff (best score 0.78)
- `ask-13` (should hand off): "ok but can a person confirm that for me?" → LLM only: handoff, Jev: no handoff (best score 0.81)
- `rule-lost-03` (should hand off): "I lost my wedding ring during my stay, has anyone found it?" → LLM only: handoff, Jev: no handoff (best score 0.60)
- `near-02` (should not hand off): "How long does a refund take to reach my card after I cancel a flexible booking?" → LLM only: no handoff, Jev: rule_matched (best score 0.88)
- `near-07` (should not hand off): "What happens to things guests leave behind?" → LLM only: no handoff, Jev: rule_matched (best score 0.94)
