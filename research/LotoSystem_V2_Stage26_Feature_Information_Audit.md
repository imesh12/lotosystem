# LotoSystem V2 Stage 26 - Mini Loto Feature Information Audit

Stage 26 is a research-only audit of the existing leakage-safe Mini Loto
number-level features. It asks whether any current feature contains stable
forward information about the next draw before changing models or adding more
feature families.

## Frozen Boundary

Discovery is frozen at Mini Loto draw `#1401`.

Draw `#1402` may be used only after the frozen Stage 26 decision is written.
Draws `#1403` and later are excluded from discovery, feature ranking,
statistical testing, time-segment analysis, rolling analysis, correlation
analysis, and Stage 27 recommendations.

## Feature Inventory

The audited features are the existing `number-features-v2` fields already
defined by the deterministic research engine. Stage 26 does not introduce a new
feature family.

For every feature the audit records:

- definition
- existing feature groups
- expected ranking direction
- historical lookback
- confirmation that the target draw is excluded

## Confirmatory Tests

Each feature is tested independently as:

- winner-vs-loser value separation
- a direct standalone ranker
- a time-segment stability profile
- a rolling 100-target-draw diagnostic
- a redundancy/correlation participant

Primary endpoints are:

- mean winning-number rank
- top-5 winning-number capture
- top-15 winning-number capture

All confirmatory feature by endpoint tests are included in Holm multiplicity
correction. Benjamini-Hochberg values are reported as exploratory diagnostics.
Inverse feature directions are reported as diagnostic only and do not validate a
feature in Stage 26.

## Stage 27 Rule

A feature can become a Stage 27 candidate only if corrected evidence is
favorable, stable across periods, practically non-trivial, leakage-safe, and not
fully redundant with a stronger feature.

If fewer than two distinct feature families survive, the Stage 27 ensemble
recommendation is `NONE`.

Historical feature behavior is descriptive research. It is not a winning
probability and does not guarantee future lottery outcomes.
