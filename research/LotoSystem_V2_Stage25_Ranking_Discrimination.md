# LotoSystem V2 Stage 25 - Mini Loto Ranking Discrimination

Stage 25 is a research-only audit of the current Mini Loto champion:

- model: Logistic Regression
- feature group: `pair_only`
- feature columns: `pair_strength_rate`
- portfolio method: `top_ranked`

The experiment asks whether the champion meaningfully separates future main
winning numbers from non-winning numbers, whether its probabilities are overly
compressed, and whether conservative calibration or Logistic Regression
regularization improves ranking without adding new features.

## Frozen Boundary

Discovery is frozen at Mini Loto draw `#1401`.

Draw `#1402` may be evaluated only after the frozen Stage 25 decision is
written. Draws `#1403` and later are excluded from discovery, tuning, signal
selection, statistical correction, and the frozen decision.

## Primary Endpoints

The pre-registered primary ranking endpoints are:

- mean winning-number rank
- top-15 winning-number capture rate
- top-5 winning-number capture rate

Secondary diagnostics include score compression, winner-vs-loser score
separation, rank stability, Brier score, log loss, expected calibration error,
and the fact that monotonic score transforms preserve rankings.

## Candidate Changes

No new features or model families are introduced.

The tested changes are:

- uncalibrated champion probabilities
- sigmoid calibration
- isotonic calibration
- Logistic Regression `C` grid: `0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0`

Calibration is fit only from walk-forward training rows available before each
target draw. Regularization uses the same `pair_only` feature and the same
walk-forward policy as the champion.

## Decision Rule

A challenger is recommended only if it improves a primary ranking endpoint,
does not materially worsen the other primary endpoints, survives multiplicity
correction at least as a weak signal, and passes leakage checks.

If no configuration satisfies that rule, the Stage 25 challenger recommendation
is `NONE`.

Historical ranking differences are research observations, not winning
probabilities and not evidence of future lottery predictability by themselves.
