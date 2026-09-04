# LotoSystem V2 Stage 27: Mini Loto Prospective Signal Tracking

## Purpose

Stage 27 creates a prospective-only tracking ledger for Mini Loto ranking signals.
It does not search for a new historical model, tune parameters, change production
strategy, or generate extra tickets.

Stages 24 through 26 found no corrected statistical evidence that recent temporal
features, pair-network features, ranking calibration, or single feature ranking
audits justified a new challenger. Stage 27 therefore stops historical feature
search and begins collecting genuinely unseen evidence.

## Tracked Signals

The initial frozen signals are:

- `production_pair_lr`: the current production-compatible Mini Loto Logistic
  Regression ranking using the `pair_only` feature group.
- `pair_strength_direct`: direct ranking by `pair_strength_rate`, tracked to
  test whether it remains equivalent to the production ranking.
- `frequency_20`: direct ranking by prior 20-draw frequency, retained as a
  research-only signal because Stage 26 found it descriptively strongest but
  statistically unsupported after correction.
- `paired_random`: a deterministic random full ranking control derived from the
  global seed, lottery, target draw number, and control id.

No other exploratory signal is part of Stage 27.

## Freeze Lifecycle

At initialization, Stage 27 reads the current canonical Mini Loto history and
sets:

```text
prospective_start_draw = latest_known_canonical_draw + 1
```

Draws before this boundary are historical metadata only. They are not counted as
Stage 27 prospective evidence.

For each future target draw, Stage 27 freezes:

- target draw number and expected draw date
- creation timestamp
- history cutoff draw/date
- dataset hash of the history available before the target
- full 31-number ranking for each tracked signal
- top-5, top-10, top-15, and top-20 snapshots
- deterministic signal configuration
- random control seed and ranking hash
- freeze SHA256 over the immutable pre-draw payload

If the target result is already present in canonical history and no frozen record
exists, Stage 27 records a missed prospective opportunity rather than rebuilding
the ranking after the fact.

## Evaluation Lifecycle

After the official result is present, Stage 27 evaluates the existing frozen
record without rebuilding the signal rankings. It records:

- actual main numbers and bonus number
- winner ranks for each signal
- mean, median, best, and worst winner rank
- top-5, top-10, top-15, and top-20 capture counts and rates
- paired differences versus the single frozen random control
- production-vs-direct pair-strength equality diagnostics
- evaluation hash

The bonus number is stored separately and is not counted as a main-number
winner.

## Random Baseline

The paired random control is a full permutation of numbers 1 through 31. Its seed
is deterministic and target-specific. The same frozen random ranking is reused
for every signal comparison for that draw.

## Immutability

Individual draw records are the source of truth. `summary.json` is derived from
those records and can be rebuilt.

The freeze hash covers only the immutable pre-result payload. Evaluation fields
are appended later and have a separate evaluation hash. Duplicate freeze and
duplicate evaluation are idempotent when the payloads match; conflicts fail.

## Evidence Gates

Prospective evidence is gated conservatively:

- fewer than 10 evaluated draws: `INSUFFICIENT_DATA`
- 10 through 25 evaluated draws: `EARLY_TRACKING`
- 26 through 49 evaluated draws: preliminary state only
- 50 or more evaluated draws: eligible for formal review

Stage 27 cannot automatically promote any signal to production.

## Automation Integration

Mini Loto operational cycles may run Stage 27 after the normal V1 lifecycle:

```text
history update -> prediction evaluation -> settlement -> next prediction -> Stage 27 tracking
```

Stage 27 failures are reported as research warnings and must not roll back
canonical history, settlements, production predictions, notifications, or
automation outcomes.

LOTO6 does not run Stage 27.

## Production Isolation

Stage 27 writes only under:

```text
data/prospective/stage27/MINI_LOTO/
```

It does not write production predictions, settlements, notification records,
shadow challengers, scheduler state, configuration, or canonical history.

Historical statistical patterns do not guarantee future lottery outcomes.

## Current Initialization

On the first Stage 27 initialization in this repository, canonical Mini Loto
history ended at draw `#1402` on `2026-09-01`. Stage 27 therefore set
`prospective_start_draw = 1403` and froze the first research-only target record
for draw `#1403`, expected on `2026-09-08`.

The frozen record uses canonical history through `#1402` only. It contains no
official result for `#1403`, and the current evidence state is
`INSUFFICIENT_DATA` because there are zero evaluated Stage 27 draws.
