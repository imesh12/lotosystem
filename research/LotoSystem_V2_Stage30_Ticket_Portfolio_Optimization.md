# LotoSystem V2 Stage 30: Mini Loto Exact Ticket Portfolio Optimization

## Purpose

Stage 30 does not search for predictive signal. Stage 29 already established,
via an authoritative global multiple-testing correction across 180 verified
historical hypotheses, that none survive correction (global Holm and global
BH surviving count: 0). Stage 30 takes that null result as given and asks a
different, purely combinatorial question:

```text
Given a fixed number of Mini Loto tickets per draw, and a FAIR
(uniform-random) draw, which ticket-construction structure maximizes the
EXACT probability of winning at least one prize?
```

Stage 30 creates no model, tunes no feature, changes no production
behavior, and does not regenerate or modify any frozen prediction. It adds
only new, isolated files:

- `backend/app/research/stage30_ticket_portfolio_optimization.py`
- `tests/test_stage30_ticket_portfolio_optimization.py`
- `research/LotoSystem_V2_Stage30_Ticket_Portfolio_Optimization.md` (this file)
- generated outputs under `data/exports/stage30/` (gitignored, matching
  existing repository policy for `data/exports/*`)

## A. Authoritative Mini Loto prize rules (verified, not assumed)

Read directly from `backend/app/domain/rules.py` and cross-validated against
`backend/app/research/prize.py::match_ticket` in
`tests/test_stage30_ticket_portfolio_optimization.py::test_qualify_table_matches_authoritative_match_ticket`:

| Rank | Name | Required main matches | Requires bonus |
|---|---|---|---|
| 1 | 1st | 5 | No |
| 2 | 2nd | 4 | Yes |
| 3 | 3rd | 4 | No |
| 4 | 4th (minimum prize) | 3 | No |

Verified structural fact (checked in code by
`assert_monotonic_bonus_eligibility`, not assumed): the bonus number can
only ever upgrade a ticket's tier (4 matches without bonus is still 3rd
prize, never "no prize"), never create or remove eligibility. This is what
makes most of Stage 30's headline probabilities exact without needing to
enumerate the bonus draw explicitly.

## B. Exact outcome space

`C(31, 5) = 169,911`, verified by direct computation and asserted at import
time. No Monte Carlo was used for any primary objective.

## C/D. Exact evaluation method

Rather than enumerating all 169,911 raw draws for every candidate portfolio,
Stage 30 partitions the 31-number universe into regions by exactly which
tickets own each number (at most 7 non-empty regions for 3 tickets), then
enumerates every valid "weak composition" of the 5 drawn numbers across
those regions (at most ~800 allocations, each with an exact multivariate
hypergeometric weight). This is mathematically exact — verified by an
internal invariant check that the summed weights equal `C(31,5)` exactly for
every portfolio evaluated — and several orders of magnitude faster than raw
enumeration, which is what makes the full structure search in part E
possible.

Primary objective and the full required breakdown, computed exactly as
fractions:

- `P(at least one prize)`
- `P(any 4th prize)` (exact tier, not cumulative)
- `P(3rd-or-better)`
- `P(2nd-or-better)` (the only tier genuinely sensitive to the bonus draw;
  resolved exactly via a region-identity argument, not simulation)
- `P(1st prize)`
- expected winning tickets per draw

### Baseline portfolios (exact)

| Portfolio | Distinct numbers | P(at least one prize) |
|---|---|---|
| 1 ticket | 5 | 0.019899 |
| 2 disjoint tickets | 10 | 0.039797 |
| **3 disjoint tickets (current production method)** | **15** | **0.059696** |
| 3 identical tickets | 5 | 0.019899 |
| 3 tickets, 7-number pool | 7 | 0.048926 |
| 3 tickets, 8-number pool | 8 | 0.052987 |
| 3 tickets, 9-number pool | 9 | 0.055511 |
| 3 tickets, 10-number pool | 10 | 0.056677 |
| 3 tickets, 11-number pool | 11 | 0.057860 |
| 3 tickets, 12-number pool | 12 | 0.059060 |
| 3 tickets, 13-number pool | 13 | 0.059272 |
| 3 tickets, 14-number pool | 14 | 0.059484 |
| 3 tickets, 15-number pool (= disjoint) | 15 | 0.059696 |

`P(at least one prize)` is strictly monotonically increasing in distinct
numbers covered, for every pool size tested.

A deterministic sample of 4,000 seeded random 3-ticket portfolios was also
evaluated exactly: observed range 0.048908–0.059696, with the sampled
maximum exactly equal to the proven global optimum (expected, since many
random triples of 5-number sets from 31 numbers are fully disjoint by
chance).

## E. Exact global optimum (N = 3)

A 3-ticket portfolio's structure is fully characterized by four integers:
`|A∩B|`, `|A∩C|`, `|B∩C|`, `|A∩B∩C|` (each 0–5, subject to validity
constraints). Stage 30 exhaustively enumerated all **150** valid
structures — not a handful of examples — and evaluated each exactly.

**Result: the fully disjoint structure (all four intersection sizes = 0,
15 distinct numbers) is the unique global maximum**, at
`P(at least one prize) = 161/2697 ≈ 0.0596960`. Every structure with any
overlap scored strictly lower, in a clean monotonic relationship with
distinct-number coverage (confirmed by the full 150-row table saved under
`data/exports/stage30/`).

**Why, mathematically:** expected winning-ticket count per draw is constant
regardless of structure (3 × a fixed single-ticket probability, by
linearity of expectation — overlap cannot change it). But for disjoint
tickets specifically, two different tickets can *never* simultaneously
reach the minimum prize threshold, because that would require at least 6 of
the 5 drawn numbers to be spread across two 3+-match tickets with no shared
numbers — impossible. So for disjoint tickets, `P(at least one wins)` and
`E[number of winning tickets]` are exactly equal. Any overlap introduces the
possibility of two tickets winning *simultaneously* off shared numbers,
which burns part of that fixed expected-win budget on redundant double-wins
instead of spreading it across mutually exclusive chances — strictly
lowering `P(at least one)` for the same fixed expectation. This is the
rigorous version of "don't play a lottery wheel with no real information" —
confirmed by exhaustive exact computation, not assumed from folklore.

## F. Comparison against actual current production (#1404)

Read-only; `#1404` (`data/predictions/MINI_LOTO/1404.json`, status
`PENDING`) and the corresponding Stage 27 record were never opened for
writing by this module, and a before/after file-hash test confirms no
mutation occurred.

Current production tickets: `[3,11,14,19,31]`, `[2,21,22,27,30]`,
`[4,5,16,23,29]` — 15 distinct numbers, fully disjoint.

**`P(at least one prize)` = 161/2697 ≈ 0.059696 — exactly equal to the
proven global optimum.** Absolute difference: 0.0. The real production
ticket-construction method (`_generate_ranked_tickets`'s disjoint-block
slicing, confirmed in the earlier independent audit) is already the
mathematically optimal structure for this objective. No change to
production is recommended or needed.

## G. Prediction ranking vs. portfolio construction — two separate scenarios

**Scenario 1 — fair lottery null.** All 31 numbers equally likely. Result:
as above, fully disjoint is the unique global optimum at
`P ≈ 0.059696`.

**Scenario 2 — model-weighted diagnostic (explicitly not authoritative).**
*Stage 29 found no evidence that any historical predictive hypothesis
survives correction. This scenario is a hypothetical: IF the production
model's scores were genuine probabilities (which is not established),
would disjoint construction still be optimal?* Using the same
recent-20-draw frequency feature family as production's `pair_only` input,
and a seeded weighted-without-replacement Monte Carlo (200,000 draws,
explicitly non-exact since this is a secondary diagnostic, not the primary
objective):

| Candidate (built from the model's own top-ranked numbers) | Simulated P(at least one prize) |
|---|---|
| Disjoint, top-15 ranked numbers | 0.1877 |
| Overlapping, 9-number pool of top-ranked numbers | 0.2268 |
| Actual #1404 production tickets (built from a different, whole-history feature, shown for reference only) | 0.0622 |

Under this specific hypothetical weighting, overlap on a small
high-confidence pool modestly outperforms full disjoint coverage — the
opposite of the fair-null result. This is intuitive: wheeling-style overlap
only pays off when you have a real reason to trust a smaller pool is more
likely to contain the winners, which is exactly the condition that does not
hold today (Stage 29: no verified signal). **This result is reported for
completeness and transparency only. It is not evidence of a real
advantage, it does not justify changing production's disjoint construction,
and the gap between "top-15 ranked" and "actual #1404 tickets" above is
mostly an artifact of two different feature lookback windows, not a flaw in
production.**

## H. Cost and long-run interpretation (optimal structure)

Cost: 3 tickets × ¥200 = ¥600 per draw. `P(at least one prize) ≈ 0.059696`.

- Expected draws between prize events: **≈ 16.8 draws** (≈ 16–17 weeks at
  Mini Loto's weekly cadence).
- Probability of **zero** prizes over 10 draws: 54.0% / over 20 draws: 29.2%
  / over 50 draws: 4.6%.
- Probability of **at least one** prize over 10 draws: 46.0% / over 20
  draws: 70.8% / over 50 draws: 95.4%.

These are long-run averages of a fair random process. They are not a
guarantee of periodic wins — a long streak of non-winning draws remains
fully possible even at the mathematically optimal structure.

## I. Ticket-count sensitivity (1–6 tickets, representative, not exhaustive)

Unlike part E (full exhaustive proof for N=3), this is a representative
sensitivity check: disjoint vs. one mild-overlap alternative at each ticket
count.

| Tickets | Disjoint P(at least one prize) | Mild-overlap alternative | Disjoint still best? |
|---|---|---|---|
| 1 | 0.019899 | n/a | — |
| 2 | 0.039797 | 0.039585 | Yes |
| 3 | 0.059696 | 0.059272 | Yes |
| 4 | 0.079595 | 0.077953 | Yes |
| 5 | 0.099493 | 0.098646 | Yes |
| 6 | 0.119392 | 0.116320 | Yes |

`P(at least one prize)` scales almost exactly linearly with ticket count
under disjoint construction (each additional disjoint ticket adds a fixed
amount of distinct-number coverage), and disjoint remained the best of the
tested alternatives at every ticket count. Six is the maximum number of
fully disjoint 5-number tickets obtainable from a 31-number universe.

This section is explicitly mathematical sensitivity analysis only. It is
not a recommendation to increase spending.

## J/K. Implementation and tests

All logic lives in the two new isolated files listed under Purpose. Test
coverage (29 tests, all passing) verifies: `C(31,5) = 169,911`; exhaustive
(not sampled) weak-composition enumeration sums to the full outcome space
for every structure evaluated; exact prize classification cross-validated
against the repository's own authoritative `match_ticket`; bonus handling
(including the region-identity argument for simultaneous bonus-sensitive
tickets); known closed-form baselines (single-ticket, identical-tickets,
disjoint-tickets) matching hand-derived hypergeometric values exactly;
intersection-structure equivalence under relabeling/shifting; deterministic,
reproducible global search; confirmation that the fair-null search takes no
history or dataset input (no lookahead possible by construction); read-only
access to `#1404` and Stage 27 records with before/after file-hash equality
checks; and that disjoint remains best across the 1–6 ticket sensitivity
sweep.

`ruff check` and `ruff format --check` pass clean on both new files and on
the full `backend`/`tests` tree. `git diff --check` reports no whitespace
errors. The full repository pytest suite was run
(364 passed, 1 failed, 131s): the one failure
(`test_stage29_global_experiment_ledger.py::test_repository_ledger_handles_stage27_and_stage28_without_runtime_mutation`)
is a **pre-existing, unrelated** failure — confirmed by re-running the exact
same test against a `git stash`-clean checkout of the original commit
(`c9c4454`) before any Stage 30 file existed, where it fails identically.
It stems from a missing `data/exports/stage24/` generated-output directory
in this checkout, not from anything Stage 30 touches.

## L. Safety confirmation

No production selector, operational cycle, prediction, settlement,
canonical history file, or Stage 27 record was modified. No new prediction
was generated. No auto-cycle was run. No scheduler, notification, or
frontend code was touched. Nothing was staged, committed, or pushed by
this work. `#1404` (and Stage 27's `#1404` record) remain exactly as they
were before Stage 30 began, verified by before/after file hashes in the
test suite.

## Final answer to the governing question

For Mini Loto, 3 tickets per draw, under a fair random draw: **the
mathematically optimal ticket structure is full disjoint coverage (15
distinct numbers, no overlap) — and this is exactly what current
production already does.** No portfolio-construction change is
recommended. The realistic long-run rate for the minimum prize under this
optimal structure is about once every 16–17 draws (roughly every four
months at Mini Loto's weekly cadence); there is no ticket-selection trick
that improves on this absent a real predictive signal, which Stage 29 did
not find.
