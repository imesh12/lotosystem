# LotoSystem V2 — Stage 31C: Restore Stage 27 Automation Continuity

## Purpose

Stage 31C is an operational-integrity repair, not a new prediction
experiment. It does two things only: (1) legitimately freezes the Stage 27
prospective target for Mini Loto #1407 before its 2026-10-06 result
becomes available, using the existing, unmodified Stage 27 freeze
mechanism, and (2) makes the minimum production-code correction so a
future automation cycle can never again silently complete while Stage 27's
own freeze/evaluate/summary lifecycle fails or is skipped. It does not
touch Stage 27's hypotheses or evidence gate, Stage 28/28B, Stage 29, or
Stage 30's conclusions, and it does not change production ticket-selection
logic or #1407's ticket numbers.

## B. Root cause of the missing #1407 freeze -- proved, not assumed

Stage 31B had already shown that Stage 27's own freeze function succeeds
against the real data in isolation. Stage 31C traced the actual execution
path on the live repository to find out why that success never reached
disk on 2026-09-29.

`run_post_draw_cycle` (`backend/app/research/operational_cycle.py`) always
calls `save_cycle_record(...)` before returning, on both its success path
(line 181, prior to this fix) and its outer `except Exception` path (line
222) -- a `CYCLE-MINI_LOTO-*.json` file is written every single time this
function runs, success or failure. The live repository's
`data/predictions/MINI_LOTO/cycles/` directory was inspected directly: the
most recent cycle record is dated **2026-09-23T23:43**, matching the cycle
that froze #1406. **There is no cycle record at all for 2026-09-29**, the
moment #1406 was evaluated, settled, and #1407's production prediction was
generated.

Likewise, `run_automation_once` (`backend/app/research/automation.py`)
always calls `_save_automation_record(...)` before returning. The live
`data/automation/runs/` directory's own timestamps show a roughly
three-hourly cadence that runs `...T09:00:06` on 2026-09-29 and then jumps
straight to `...T06:00:07` on 2026-09-30 -- a ~21 hour gap spanning
exactly the 23:46 timestamp at which #1406 was processed. **No automation
run record exists for that moment either.**

Both of the only two code paths that can invoke Stage 27 (`run-cycle` via
the CLI, and the scheduled `auto-run`) unconditionally persist a record of
having run, every time, including on failure. Since neither persisted a
record for the exact moment #1406 was processed, **neither of them is what
processed #1406 on 2026-09-29.** Something else -- outside both entry
points, most plausibly an ad-hoc/manual intervention after the ~21 hour
automation gap was noticed -- evaluated the production prediction and
Stage 27's #1406 record directly (both show the same 23:46:1x timestamp)
without ever calling the unified `run_stage27_cycle` orchestration that
also freezes the next target and rebuilds the summary. This is consistent
with every other observed fact: #1406 evaluated, #1407 never frozen,
`summary.json` stale since 2026-09-23, no cycle or automation record for
that moment.

**This is not a Stage 27 or Stage 31 defect.** When `run_post_draw_cycle`
/ `run_stage27_cycle` genuinely run (as they did for #1403 through #1406
originally, each with its own matching `CYCLE-MINI_LOTO-*.json` record and
`stage27_present=True`), freezing works correctly, proved directly in
Stage 31B by running the live freeze function against the real data. The
gap is specifically that the one occasion #1406 was processed bypassed
that orchestration entirely.

## C. Safely freezing #1407

All required conditions were verified directly against the live
repository immediately before acting, not assumed:

- canonical history ends at #1406 -- confirmed (864 rows, latest
  2026-09-29).
- #1407 result absent -- confirmed (no #1407 row in the canonical CSV).
- history cutoff = #1406, dataset hash corresponds exactly to history
  through #1406, ranking generated only from data through #1406 -- all
  three are guaranteed by construction by the freeze function itself
  (`freeze_next_stage27_record`), which derives every one of these fields
  from the `draws` tuple passed to it.
- freeze timestamp before result availability -- the freeze was created
  at 2026-10-05T04:37:44Z, the draw is scheduled 2026-10-06.
- integrity hash valid -- `stage27_freeze_hash` recomputed and compared
  equal to the stored hash immediately after creation.

All conditions held, so the **existing, unmodified**
`freeze_next_stage27_record` function (no hand-written JSON) was run
against an isolated, read-only copy of the live `data/processed/mini_loto_history.csv`
and `data/prospective/stage27/MINI_LOTO/` records (864 draws, latest
#1406). It produced:

```
status: FROZEN
existing_record: False
target_result_absent: True
draw_number: 1407
draw_date: 2026-10-06
created_at: 2026-10-05T04:37:44.456674+00:00
history_cutoff_draw: 1406
history_cutoff_date: 2026-09-29
history_dataset_hash: 86fe4688ca414da8b4f83e0f325ea48fe9f131c3f6d610b2add3680a5502e991
freeze_hash: 28de5c61d25f826612f4703d5ff9de8b1e0c62b5d6da55d3d80ad150d450c94b
```

Note `history_dataset_hash` matches, exactly, the `dataset_hash` already
recorded in production's own `data/predictions/MINI_LOTO/1407.json`
(`86fe4688...`) -- independent confirmation that Stage 27's freeze and
production's prediction were built from the identical canonical history
snapshot. This single resulting file was then written to the live
repository's `data/prospective/stage27/MINI_LOTO/1407.json` and
independently re-hashed on disk to confirm the write matched exactly; no
other file was touched.

## D. Automation fix

`backend/app/research/automation.py`'s `_run_due_cycle` already receives
the full `OperationalCycleResult`, including its `stage27` field, from
`runner(...)` (which defaults to `run_post_draw_cycle`). It simply never
forwarded that field into the dict it returns, and never folded a Stage 27
failure into the automation record's own `errors`. The minimum fix:

- `_run_due_cycle`'s success-path return now includes `"stage27":
  cycle_payload["stage27"]` and a new `"stage27_lifecycle_status"` field
  (`OK` / `ERROR` / `NOT_APPLICABLE`), computed by a new helper,
  `_stage27_lifecycle_outcome`. For a non-MINI_LOTO lottery this is always
  `NOT_APPLICABLE` (Stage 27 never runs for LOTO6, unchanged). For
  MINI_LOTO, it is `ERROR` whenever `stage27_payload` is `None` or carries
  `status == "ERROR"` (the exact shape `_stage27_cycle_payload` already
  produces when it catches a `ResearchValidationError`), and `OK`
  otherwise.
- Whenever the outcome is `ERROR`, a descriptive string is appended to
  this lottery's `errors` tuple (which `run_automation_once` already
  aggregates into the run-level `errors`), so a Stage 27 failure can no
  longer be silently absorbed while the rest of the payload reports a
  normal `RESULT_PROCESSED`/`CHECK_RESULT` action.
- The pre-existing `except ResearchValidationError` branch (an entire
  cycle failing before Stage 27 could even run) now also reports
  `"stage27": None` and the matching `stage27_lifecycle_status`, for
  schema consistency across every returned shape.
- `_no_action_payload`, `_disabled_payload`, and `_create_future_prediction`
  are deliberately left unchanged: none of them ever invoke Stage 27, so
  adding placeholder keys there would not reflect anything real. Any
  consumer of these records must already use `.get(...)` rather than
  assume every action type carries a `"stage27"` key.

No change was made to `run_post_draw_cycle`, `_stage27_cycle_payload`, or
anything in Stage 27's own module -- the orchestration, when it runs, was
already correct; only its *outcome* was being dropped on the floor by the
automation layer.

## E. Automation run record schema

`_save_automation_record` (unchanged) serializes whatever dict
`run_automation_once` builds, so the two new keys above
(`"stage27"`, `"stage27_lifecycle_status"`) now flow straight into every
future `AUTO-*.json`. This is purely additive: existing `AUTO-*.json`
files on disk, which lack both keys, remain valid JSON and are not
rewritten. Nothing in this repository reads a saved automation record's
`"stage27"` key elsewhere (confirmed by search), so there was no existing
consumer to break; any future consumer must use `.get("stage27")` /
`.get("stage27_lifecycle_status")`, exactly like every other optional
field on these records.

## F. Idempotency and immutability (proved by new tests, see H)

- Rerunning the automation entry point with no new draw to fetch leaves an
  already-frozen Stage 27 record byte-for-byte identical (same
  `freeze_hash`, same file bytes) -- no duplicate record is created for
  the same draw number.
- An evaluated Stage 27 record stays byte-for-byte identical across a
  later cycle that freezes a different target, and across a further
  idempotent rerun.
- The production prediction record (tickets, dataset hash, generated_at)
  is unchanged across an idempotent automation rerun.
- Every frozen target's `history_cutoff_draw` always equals the latest
  draw actually present in canonical history at freeze time and is always
  strictly less than the target's own `draw_number` -- no future-data
  leakage, checked across two successive real cycles.

## G. Live-data safety

Protected files were hashed immediately before this task's single
authorized live mutation and re-verified identical afterward (see the
final report for the full table). The *only* live file this task wrote
was `data/prospective/stage27/MINI_LOTO/1407.json`, and only after every
Section C condition was independently confirmed. All test work ran
against isolated `tmp_path` fixtures; no test touched the live repository.

## H. Tests

`tests/test_stage31c_automation_stage27_continuity.py` (8 tests, all
passing): Stage 27 lifecycle surfaced and the next target frozen
automatically through a real (non-mocked) `run_post_draw_cycle` wired
into `run_automation_once`; a synthetic Stage 27 failure surfaced in both
the per-lottery and run-level `errors` rather than hidden; the
whole-cycle `ResearchValidationError` path correctly reports
`stage27_lifecycle_status=ERROR` for MINI_LOTO; duplicate automation
execution does not duplicate or alter a frozen record; an evaluated
record stays immutable across a later cycle and a further idempotent
rerun; the production prediction is unchanged across an idempotent
rerun; every freeze's history cutoff is checked against the latest known
draw across two successive cycles (no leakage); and a pre-fix-shaped
automation record (missing both new keys) still parses and degrades
gracefully through `.get(...)`.

## Final report

See the conversation accompanying this file for the full, enumerated
final report; the live repository state after this repair matches the
expected healthy pre-result state exactly: latest canonical draw #1406,
latest evaluated Stage 27 target #1406, latest frozen Stage 27 target
#1407, one pending target (#1407), 0 missed, 0 integrity failures.
