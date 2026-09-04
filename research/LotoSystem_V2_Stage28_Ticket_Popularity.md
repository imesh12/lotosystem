# LotoSystem V2 Stage 28: Mini Loto Ticket Popularity Research

## Purpose

Stage 28 studies a different question from prediction:

> Can Mini Loto tickets be scored for human-pattern popularity risk, so that if
> a ticket wins a shared prize it may be less likely to split the prize with many
> other winning tickets?

This stage does not change draw probability, hit probability, production model
selection, Stage 27 prospective tracking, or ticket generation.

## Data Reality

The current repository contains canonical Mini Loto winning-number history and
recent settlement records with official payout and winner-count fields where
settlements exist.

The repository does not contain actual player ticket-selection distributions,
winner ticket combinations, Quick Pick/manual ratios, sales amount history, or
public number-selection popularity rankings. Therefore Stage 28 cannot directly
estimate actual combination popularity.

## Proxy Score

Stage 28 defines `HEURISTIC_POPULARITY_PROXY`.

Higher score means higher human-pattern popularity risk. Lower score means a
ticket appears less conventionally patterned under the proxy. This is a heuristic
only, not an observed player-choice probability.

Components:

- consecutive-pair count
- longest consecutive run
- arithmetic progression indicator
- low range concentration
- low-number fraction
- same-decade concentration
- repeated last digit count
- even/odd extremeness
- sum extremeness
- spacing regularness
- adjacent gap repetition
- simple sequence indicator
- round-number fraction

Each component is normalized to `0..1` and the final proxy score is the average
of all components.

## Universe Enumeration

Mini Loto has:

```text
C(31, 5) = 169,911
```

Stage 28 deterministically enumerates every valid sorted 5-number combination
from 1 through 31 and scores the full universe. No random sampling is used for
the universe.

## Historical Winner Calibration

Every historical Mini Loto winning combination is scored against the full
combination universe. The resulting percentile distribution is a calibration
check, not a prediction test. Randomly drawn winning combinations should not be
expected to systematically favor low- or high-popularity proxy scores.

## Winner-Count Association

The preregistered primary endpoint is:

```text
popularity score vs first-prize winner count after sales normalization
```

Preferred normalization:

```text
first_prize_winners / (sales_amount_yen / ticket_price_yen)
```

If sales amount is unavailable, the primary endpoint is classified as
`INCONCLUSIVE`. Raw winner counts alone are not sufficient because more tickets
sold naturally increases expected winner counts.

## Stage 28B Economic Dataset Acquisition

Stage 28B adds a separate Mini Loto economic-result acquisition path under:

```text
data/exports/stage28/mini_loto_economic_history.csv
```

The dataset is separate from canonical draw history and production ledgers. It
does not modify `data/processed/`, prediction records, settlements, Stage 27
prospective records, scheduler state, email, or frontend state.

Current source audit:

- Official Mizuho Chrome-saved backnumber HTML available locally provides draw
  numbers, draw dates, winning numbers, and bonus numbers, but not historical
  sales amount or prize-tier winner/payout fields in the saved pages inspected.
- Existing SMBC XML payout ingestion provides prize-tier winner counts and
  payout amounts for recent processed settlements.
- Existing local settlements preserve those SMBC winner-count and payout fields
  where a paper-trading settlement exists.
- No current local source provides `sales_amount_yen`.
- No current local source provides actual purchased-number distribution,
  winning ticket combinations, Quick Pick/manual selection ratio, or official
  player-choice popularity rankings.

Because `sales_amount_yen` is still unavailable, Stage 28's preregistered
sales-normalized primary endpoint remains `INCONCLUSIVE`. Raw first-prize
winner counts are not substituted for the primary analysis.

## Conditional Payout Illustration

Conditional payout examples use a fixed illustrative prize pool:

```text
conditional payout per winning ticket = prize pool / winner count
```

This explains why popularity could matter conditional on winning. It does not
model the official payout formula and does not claim that the proxy predicts
winner count.

## Recommendation Gate

An anti-popularity selector is built only if the sales-normalized association
shows at least `WEAK_SIGNAL` after correction and the effect direction is
positive. Otherwise:

```text
recommendation = NONE
```

Stage 28 cannot alter production tickets or Stage 27 signals.

## Production Isolation

Stage 28 writes only research exports under:

```text
data/exports/stage28/
```

It does not write:

- `data/predictions/`
- `data/processed/`
- `data/settlements/`
- `data/prospective/stage27/`
- `config/`
- scheduler, email, notification, or frontend state

Historical statistical patterns do not guarantee future lottery outcomes.
