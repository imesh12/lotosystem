# Stage 03 Notes

Stage 03 hardens historical data and evaluation before any machine learning or LLM work.

## Historical CSV Schema

Recommended explicit schema:

```text
lottery,draw_number,draw_date,n1,n2,n3,n4,n5,n6,bonus,source,source_url,retrieved_at,content_hash
```

Mini Loto omits `n6`.

Packed fields are also supported:

```text
draw_number,draw_date,main_numbers,bonus_numbers
```

`lottery` is optional, but when present it must match the requested lottery.

## Synthetic Fixtures

Small fixture CSV files exist for both lotteries under `tests/fixtures`. They are synthetic structural fixtures, not authoritative historical draw data.

Authoritative historical import remains a future task.

## Official Rule Context

Prize classifications are represented without fixed payouts. The official pages state that winning amounts vary by sales and winner counts, so payout calculation is intentionally deferred.

## Out Of Scope

Stage 03 does not add machine learning, databases, official website scraping, LLM agents, scheduling, dashboards, or automatic purchasing.
