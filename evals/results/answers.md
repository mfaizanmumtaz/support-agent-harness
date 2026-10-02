## Answer eval (24 questions, chat model `gpt-6-luna`)

| Setup | Correct | Model calls / answer | Median first words | Median full reply | p90 full reply | Input tokens / answer |
|---|---|---|---|---|---|---|
| **Search before the model** | 24/24 | 1.25 | 1565 ms | 2038 ms | 3414 ms | 1042 |
| **Tool-only (model decides)** | 24/24 | 2.00 | 2396 ms | 2858 ms | 3331 ms | 1435 |

Correct = the reply contains the fact from the knowledge base (or, for the 3 questions the knowledge doesn't cover, says it doesn't know). Checked by phrase match, so it is a floor, not a judge.

### Misses

None.
