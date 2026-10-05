# LotoSystem V2 — Stage 31: Operational Integrity + Prospective Evidence Monitor

## Purpose

Stage 31 is not a new prediction experiment. It introduces no new predictive
features, ML models, number-selection strategies, or retrospective
hypothesis searches. It is a read-only auditor and machine-readable summary
builder that continuously proves -- or flags doubt about -- the operational
integrity of Stage 27's prospective tracking, its relationship to Stage 29's
global hypothesis governance, and the repository's handling of disposable
test artifacts versus intentional evidence.

It does not modify the mathematical conclusions of Stage 27, Stage 28/28B,
Stage 29, or Stage 30, and it does not change production ticket generation.
Those conclusions stand exactly as published:

- **Stage 29:** 0 globally corrected predictive hypotheses currently survive.
- **Stage 30:** for three Mini Loto tickets under the fair-null objective,
  the fully disjoint portfolio is globally optimal for P(at least one
  prize).

## What already existed (reused, not duplicated)

Stage 27 (`backend/app/research/stage27_prospective_signals.py`) already
implements a correctly-ordered, already-idempotent prospective pipeline:
`initialize_stage27` -> `record_missed_draws` -> `evaluate_stage27_records`
-> `freeze_next_stage27_record` -> `rebuild_stage27_summary` /
`save_stage27_summary`, wired into the live operational cycle
(`backend/app/research/operational_cycle.py::run_post_draw_cycle`) in
exactly this order, right after canonical history update, previous
prediction settlement, and next production-prediction generation -- and
always against the same just-updated history, so there is no future-data
leakage by construction. Re-running the cycle on an already-frozen/evaluated
target is a no-op: `freeze_stage27_record` returns the existing record
unchanged (after re-verifying its hash) rather than refreezing, and
`evaluate_stage27_records` skips any record already `EVALUATED` after
re-checking its evaluation hash.

Stage 29 (`backend/app/research/stage29_global_experiment_ledger.py`)
already marks every Stage 27 signal's hypothesis record
`provenance_status=PROSPECTIVE_ACTIVE`,
`eligible_for_global_correction=False`,
`exclusion_reason="PROSPECTIVE_NOT_YET_EVALUABLE"` unconditionally -- so
interim Stage 27 looks can never enter the global Holm/Benjamini-Hochberg
correction family while active, by construction, regardless of how the
sample size grows between runs.

Stage 31 adds a new, independent audit layer on top of both -- it never
edits either file.

## B. Stage 27 health monitor

`backend/app/research/stage31_operational_integrity_monitor.py::audit_stage27_health`
independently re-derives, directly from the per-draw JSON files and
`metadata.json` (never from the cached `summary.json`):

- latest canonical Mini Loto draw, prospective start draw
- every frozen target, evaluated target, pending target
- missing targets (gaps in the expected `prospective_start_draw..latest`
  range) cross-checked against `metadata.json`'s own `missed_draws` list
- duplicate targets (defensive; the one-file-per-draw-number layout should
  make this structurally impossible, but it is still checked)
- freeze timestamp, history cutoff draw/date, and dataset hash for every
  target
- **freeze-hash validity**: recomputes `stage27_freeze_hash(record)` and
  compares to the stored hash -- tamper-evidence for the frozen ranking
- **evaluation-hash validity**: recomputes the evaluation payload via
  `evaluated_stage27_record(record, actual, lottery)` and compares the
  resulting hash to the stored one -- tamper-evidence for the evaluated
  result
- **freeze-before-result check**: flags any record where
  `history_cutoff_draw >= draw_number` (the frozen ranking's own claimed
  information cutoff reaches or passes its own target draw)
- **independent wall-clock check**: flags any record whose `created_at`
  date is not strictly before its own target `draw_date`, a second,
  independent signal of accidental retrospective generation that does not
  rely on the record's own self-reported cutoff field

If any of these checks fail for a record, Stage 31 lists that draw number in
`integrity_failure_targets` and reports it. **It never rewrites, restores,
or silently repairs a record** -- the audit is read-only end to end.

## C. Automatic prospective evaluation (verified, not re-built)

Items C.1-C.7 (history update -> previous prediction settlement -> previous
Stage 27 evaluation -> immutable frozen ranking -> next Stage 27 freeze ->
cutoff-only-through-latest-known-draw -> no future information) are already
implemented, in exactly this order, by the existing
`run_post_draw_cycle` / `run_stage27_cycle` pipeline (see above). Stage 31
does not re-implement this orchestration -- re-implementing it would risk
exactly the kind of drift this stage exists to prevent. Instead, Stage 31's
test suite exercises `run_stage27_cycle` twice against identical synthetic
inputs in isolated `tmp_path` fixtures and asserts the resulting audit is
byte-for-byte identical both times (idempotency), and separately verifies
every one of the freeze/evaluate/hash/leak checks above against both
healthy and deliberately-tampered synthetic records.

## D. Evidence dashboard / summary

`build_stage31_summary` returns a `Stage31Summary` containing:
`prospective_draws_frozen`, `prospective_draws_evaluated`,
`prospective_draws_pending`, `prospective_draws_missed`,
`integrity_failures`, `earliest_target`, `latest_target`,
`latest_evaluated_target`, and a `counts_consistent_with_stage27_summary`
flag that cross-checks Stage 31's independently-derived counts against a
freshly rebuilt `rebuild_stage27_summary()` call.

For each Stage 27 signal (`production_pair_lr`, `pair_strength_direct`,
`frequency_20`, `paired_random`) it reports: top5/top10/top15 capture
average (top10 is newly aggregated by Stage 31 directly from the per-draw
records, since Stage 27's own summary only aggregates top5/top15), mean rank
of the winning numbers, the paired comparison against `paired_random`, and
Stage 27/29's own `classification` string -- reused verbatim via
`rebuild_stage27_summary`, never re-derived, so Stage 31's evidence-gate
decision can never drift from Stage 27/29's own tested permutation-test
machinery.

`declares_predictive_evidence` is hardcoded `False` on every signal
snapshot, and `predictive_edge_currently_supported` is hardcoded `False` on
the summary as a whole. Stage 31 never itself declares predictive evidence,
regardless of sample size or classification -- that remains a deliberate,
one-time human decision routed through Stage 29 (section E).

## E. Stage 29 integration: deterministic registration preview

`preview_stage27_global_registration` is a pure, side-effect-free preview.
It never writes to any Stage 29 export, never mutates Stage 27 data, and
never changes Stage 29's current conclusion. For each signal (excluding the
`paired_random` control) whose classification currently equals
`ELIGIBLE_FOR_REVIEW` -- Stage 27/29's own preregistered evidence gate --
it builds the exact, deterministic `GlobalHypothesisRecord` draft a human
maintainer would register with Stage 29, using Stage 29's own
`make_hypothesis_record` builder, tagged `provenance_status=
VERIFIED_OUTPUT_ONLY`, `preregistered=True`, and an explicit note that this
represents the **one-time** final registration the hypothesis is entitled
to, and that registering it twice or registering an earlier interim look
would be optional stopping / cherry-picking. While no signal's gate is
reached (the current, live state -- see Final Report), this always and only
returns `gate_reached: False` with an empty draft list and
`action_required: "none"`.

## F. Repository safety: .gitignore / disposable-artifact audit

`audit_repository_hygiene` is read-only: it reads `.gitignore` and runs only
`git ls-files` (a listing command -- never `add`/`commit`/`push`/`reset`/
`restore`/`clean`). It classifies disposable test-artifact directories
(`.pytest-tmp/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`) separately
from intentional, version-controlled evidence paths
(`data/predictions/`, `data/settlements/`, `data/prospective/`,
`data/processed/mini_loto_history.csv`, `data/processed/loto6_history.csv`,
`research/`), and only ever **reports recommendations** -- it does not edit
`.gitignore`, untrack, or delete anything itself. See the Final Report for
what it found in this repository.

## G. Live-data safety

Every test that needs real production/history/settlement/prospective data
hashes the six protected files (the same six named in the Stage 30 recovery
task) before calling any Stage 31 function against the live repository
paths, and asserts the hashes are identical afterward. Every other test
uses pytest's `tmp_path` fixture for a fully isolated, synthetic Stage 27
root -- no test ever writes under `data/processed/`, `data/predictions/`,
`data/settlements/`, or `data/prospective/stage27/`.

## H. Tests

`tests/test_stage31_operational_integrity_monitor.py` (19 tests) covers:
freeze-before-result integrity, future-data leakage prevention (both the
internal cutoff-field leak and the independent wall-clock leak), duplicate
cycle-and-audit idempotency, missing-draw detection (and its consistency
check against `metadata.json`), immutable frozen ranking (freeze-hash
tamper detection), immutable evaluated record (evaluation-hash tamper
detection), dataset-hash presence/stability, Stage 27 evidence-gate
behavior below threshold, Stage 29 integration-gate preview both below and
above threshold (including determinism of the preview and exclusion of the
`paired_random` control), MINI_LOTO-only guard, and three repository-hygiene
tests (missing `.gitignore`, a partially-covered `.gitignore`, and
no-disk-writes). Live-data safety is covered directly (see section G).

## Final Report (as of this audit)

> **CORRECTION (Stage 31B, 2026-10-05):** The paragraph below was computed
> by running this module against a stale cloud-sandbox clone of the
> repository (history frozen at draw #1403) and was incorrectly written up
> as if it reflected the live repository's actual state. It does not. The
> live repository's canonical Mini Loto history and Stage 27 records were
> independently verified to be at **draw #1406** (4 frozen targets, all 4
> `EVALUATED`, 0 integrity failures) at the time this correction was made.
> This was a one-time process error in how Stage 31 was invoked, not a bug
> in Stage 31's own code -- see
> `research/LotoSystem_V2_Stage31B_Live_Data_Reconciliation.md` for the full
> root-cause investigation, live re-audit, and corrected figures. The
> numbers below are preserved as-originally-written for the repository's
> own audit trail and must not be treated as current.

27-item report delivered in the conversation accompanying this file;
summarized here for the repository record. At the time of this audit:
Stage 27 had 2 frozen targets (one evaluated, one pending), 0 missing
targets, 0 integrity failures, and every signal's classification was
`INSUFFICIENT_DATA` (evaluated-draw sample size far below the preregistered
evidence gate) -- so `any_signal_reaches_evidence_gate=False` and the Stage
29 registration preview correctly returned no draft records and
`action_required: "none"`. The repository-hygiene audit found `.gitignore`
already covers `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/`, but is
missing `.pytest-tmp/` -- and a large number of files under `.pytest-tmp/`
(pytest `tmp_path` fixture output) are, as a result, currently tracked by
git and churn on every test run. Stage 31 recommends adding `.pytest-tmp/`
to `.gitignore` and then untracking (never deleting) those files in a
separate, deliberate commit; it does not do either itself.

**IMPORTANT:** The frozen/evaluated/pending counts and sample size above
were computed against a stale sandbox clone, not the live repository. See
the correction notice above.
