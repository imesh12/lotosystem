# LotoSystem V2 Stage 24 Temporal Discovery

Stage 24 is a research-only Mini Loto temporal-signal experiment.

## Boundary

Discovery is frozen at Mini Loto draw `#1401`. Draw `#1402` is reserved as a
single holdout observation and must not affect signal discovery, feature design,
parameter choice, ranking, or statistical correction. Draws after `#1402` are
excluded from the Stage 24 decision.

The live-system stale-ingestion hypothesis for Mini Loto `#1402` is obsolete:
the Windows production diagnosis confirmed that `#1401` was ingested before the
`#1402` prediction, dataset hashes changed, and scores moved slightly. The
repeated tickets came from stable model rankings, not stale history.

## Signals

The experiment evaluates these pre-registered recent temporal signals:

- recent-frequency momentum
- recent-frequency mean reversion
- consecutive-draw carryover
- multi-lag recurrence, emphasizing `t-2`
- transition from previous draw numbers
- number persistence
- short-window regime concentration
- hot/cold state change
- bonus-to-main follow behavior, separated from main-number co-occurrence

Each signal ranks Mini Loto numbers using only draws strictly before the target
draw. The primary metric is main-number capture among the top five ranked
numbers, compared against paired seeded random controls.

## Governance

Stage 24 reports paired differences, bootstrap confidence intervals,
permutation p-values, Holm adjusted p-values, Benjamini-Hochberg exploratory
p-values, effect sizes, and conservative labels:

- `EVIDENCE`
- `WEAK_SIGNAL`
- `NO_EVIDENCE`
- `NEGATIVE`
- `INCONCLUSIVE`

No Stage 25 challenger is justified unless the strongest signal has a positive
effect, a confidence interval excluding zero, and corrected statistical support.

## Holdout

After the frozen decision record is written, Mini Loto `#1402` may be evaluated
once as holdout-only evidence. This one draw cannot change the frozen Stage 24
classification or Stage 25 recommendation.

Historical statistical patterns do not guarantee future lottery outcomes.
