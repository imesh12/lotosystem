# Data Model

Stage 03 keeps the domain objects as the lottery-rule source of truth and hardens the deterministic research draw representation. These models define lottery rules and validate tickets and draws. They do not implement machine learning, historical scraping, database persistence, or LLM behavior.

## Lottery Definitions

The source of truth for supported lottery rules is `backend.app.domain.rules`.

Current definitions:

- LOTO6: numbers 1-43, 6 main numbers, 1 bonus number, 200 yen per ticket, drawings on Monday and Thursday.
- Mini Loto: numbers 1-31, 5 main numbers, 1 bonus number, 200 yen per ticket, drawings on Tuesday.

Prize tiers are stored as metadata so matching and evaluation logic can use a consistent rule source. Prize classification is implemented separately from monetary payout calculation because payout amounts vary by draw.

## Canonical Tickets

Tickets are normalized by sorting unique numbers and rendering them as two-digit values separated by hyphens.

Example:

```text
42 31 15 08 03 24 -> 03-08-15-24-31-42
```

Number order is never treated as meaningful for tickets.

## Validation

The domain layer validates:

- required number count
- number range
- duplicate main numbers
- duplicate bonus numbers
- bonus numbers overlapping main numbers
- prediction tickets matching their lottery code

Validation errors are raised before later data, statistics, or prediction stages can consume invalid lottery records.

## Historical Draws

The research engine uses `HistoricalDraw` for loaded historical records. It clearly separates main numbers, bonus numbers, and optional source metadata:

- `source`
- `source_url`
- `retrieved_at`
- `content_hash`
- `source_row`

Historical draw sequences reject duplicate draw identities, duplicate draw numbers, invalid source order, malformed dates, invalid number counts, invalid ranges, duplicate main numbers, and bonus/main conflicts. Mixed-lottery datasets are rejected before statistics or backtesting run.

## Dataset Hashing

Dataset hashes are calculated from canonical lottery code, draw number, draw date, sorted main numbers, and sorted bonus numbers. File paths, import timestamps, persistence timestamps, and operating-system details are excluded. Canonical ordering is by lottery code, draw date, and draw number.
