# LotoSystem V2 — Stage 31B: Live Data Reconciliation Before Commit

## Purpose

Stage 31B does not redesign Stage 31, change prediction logic, change
Stage 27 hypotheses, or change Stage 29/Stage 30 conclusions. It exists
solely to reconcile a discrepancy: Stage 31's own final report claimed the
latest canonical Mini Loto draw was #1403, while the live repository and
dashboard report draw #1406. This document records the root-cause
investigation, the corrected live-data audit, and an independent
production-automation finding discovered along the way.

## 1. Root cause of the #1403-vs-#1406 discrepancy

**Finding: process error on my part, not a Stage 31 code defect.**

Stage 31's functions (`audit_stage27_health`, `build_stage31_summary`, etc.)
all accept `root`/`draws` parameters and never hardcode a path. The
original Stage 31 final report was produced by running these functions in
the cloud sandbox against `/home/claude/work/loto-system`, a clone whose
`data/processed/mini_loto_history.csv` has been frozen at draw #1403 since
earlier recovery work. That sandbox-derived result was then written up as
if it were a direct measurement of the live repository. It was not.

The live repository's own dashboard/API reads `canonical_history_path()`
directly (`backend/app/research/operations.py:230`, imported from
`backend/app/research/history_import.py`) against the real
`data/processed/mini_loto_history.csv` on the user's machine -- the same
file independently read in this investigation via the device bridge. There
is no wrong-repo, stale-history-in-git, alternate-data-root,
environment-variable, cached-API-state, or Linux-bridge-path-mismatch bug
in Stage 31 or in the dashboard. The dashboard was right. My prior report
was wrong because it was pointed at the wrong data, not because the
underlying code is broken.

## 2. Verification of #1404, #1405, #1406 (live repository data, read-only)

All four fields below were read directly from the real per-draw JSON files
and independently re-verified by running Stage 31's own `audit_stage27_health`
against an isolated, read-only copy of the live data (see Section 5).

| Draw | Canonical result | Production prediction (`generated_at`) | Settlement | Stage 27 frozen record (`created_at`) | History cutoff draw | Dataset hash (history_dataset_hash) | Freeze/eval hash valid | Evaluated | Freeze-before-result |
|---|---|---|---|---|---|---|---|---|---|
| 1404 | exists (2026-09-08) | exists | exists | exists, 2026-09-08T23:54:46.916422+00:00 | 1403 | `56bfa963eecb4337f9aa795cbc3cfd4740ddb52451ba74afa58e2ff229aaa11a` | both valid | yes | OK (1403 < 1404) |
| 1405 | exists (2026-09-15) | exists | exists | exists, 2026-09-15T23:44:37.297126+00:00 | 1404 | `8ce7155e66803db315e760a70583f896ccb302b4d668cd7ee0c6d4272fd281d1` | both valid | yes | OK (1404 < 1405) |
| 1406 | exists (2026-09-29) | exists | exists | exists, 2026-09-23T23:43:51.515631+00:00 | 1405 | `4d1d1605a8daac91dc5684d8a149d04bc35bd5102ceab3afa776afe321003390` | both valid | yes | OK (1405 < 1406) |

For all three: `created_before_draw_date_ok=True` (the independent
wall-clock leak check), `retrospective_leak_suspected=False`, and the
#1406 record's own `prediction_dataset_hash` in its settlement file
(`4d1d1605...`) matches its Stage 27 `history_dataset_hash` exactly,
cross-confirming dataset provenance between the production and research
pipelines. No integrity failures found for any of 1403-1406 (1403 is
likewise clean; see Section 5's full audit output).

## 3. Verification of #1407

- Production prediction: **exists**, status `PENDING`, `generated_at
  2026-09-29T23:46:13.554030+00:00`, `latest_source_draw_number: 1406`,
  `latest_source_draw_date: 2026-09-29`, `target_draw_number: 1407`,
  `target_draw_date: 2026-10-06`. No future-result leakage: the prediction
  was built using history only through #1406, strictly before its own
  target draw.
- Stage 27 frozen target for #1407: **does not exist.** No
  `data/prospective/stage27/MINI_LOTO/1407.json` file is present in the
  live repository.
- Because no Stage 27 record exists, `created_at`, history cutoff, dataset
  hash, and integrity hash are all not applicable for #1407's (absent)
  Stage 27 record.
- This gap, its cause, and its consequence are covered in Section 6.

## 4. Independent audit of the #1406 win

From the live settlement file (`data/settlements/MINI_LOTO/1406.json`),
read directly, not assumed:

- Official result: main `[11, 14, 27, 30, 31]`, bonus `[13]`.
- Set 1 `[3, 11, 14, 19, 31]`: shared numbers with the official main set are
  `{11, 14, 31}` -- exactly 3 matches, 0 bonus matches. The record's own
  `main_match_count: 3`, `prize_tier: "4th"`, `payout_yen: 800` match this
  independently-recomputed intersection exactly.
- Set 2 `[2, 4, 21, 27, 30]`: shared numbers `{27, 30}` -- 2 matches, 0
  bonus, `prize_tier: "NO_PRIZE"`, `payout_yen: 0` (2 main matches alone do
  not clear Mini Loto's minimum prize threshold).
- Set 3 `[5, 16, 22, 23, 29]`: 0 shared numbers, `NO_PRIZE`, `payout_yen: 0`.
- Ticket cost: 3 x 200 = 600 yen. Gross winnings: 800 yen. Net: +200 yen.
  `paper_total_cost_yen: 600`, `paper_gross_winnings_yen: 800`,
  `paper_net_yen: 200` in the settlement file confirm this exactly.
- Payout provenance: the 800-yen 4th-tier payout_yen figure comes from
  `source: "smbc_public_result"` (the official SMBC-published pari-mutuel
  result page), `winners_count: 59354`, `retrieved_at:
  2026-09-29T23:46:10.529803+00:00` -- an officially sourced, not
  simulated or assumed, figure.
- This single win is **not** predictive evidence. Stage 30's fair-null
  baseline for three disjoint Mini Loto tickets, P(at least one prize) =
  161/2697 ~= 5.9696% per draw, remains the authoritative reference; a
  single win is fully consistent with that baseline occurring by chance
  and updates nothing about Stage 27/29's evidence-gate status.

## 5. Re-run of the Stage 31 monitor against the authoritative live data root

Stage 31's own `audit_stage27_health` and `build_stage31_summary` functions
were executed, unmodified, against an isolated, read-only snapshot of the
live repository's actual `data/processed/mini_loto_history.csv` and
`data/prospective/stage27/MINI_LOTO/*.json` files (copied via the device
bridge into a temp directory outside both the live repository and the
stale sandbox clone; the live repository itself was never written to).
Results:

- Latest canonical Mini Loto draw: **#1406**
- Prospective start draw: 1403
- Frozen targets: 1403, 1404, 1405, 1406 (latest frozen target: 1406)
- Evaluated targets: 1403, 1404, 1405, 1406 (latest evaluated target: 1406)
- Pending targets: none (0)
- Missing targets: none (0) -- by Stage 31's own definition, a target is
  "missing" only if it falls inside
  `prospective_start_draw..latest_frozen_target` without a record; since no
  cycle has yet attempted to freeze #1407, it is not currently a gap in
  that range. It would become a recorded "missed" target automatically,
  per Stage 27's own `record_missed_draws` logic, the first time a future
  cycle runs with #1407 already in canonical history and still no
  `1407.json` on disk (see Section 6).
- Retrospective targets: none (`retrospective_leak_suspected=False` for
  all four).
- Integrity failures: **0** (`freeze_hash_valid`, `evaluation_hash_valid`,
  `freeze_before_result_ok`, and `created_before_draw_date_ok` all `True`
  for 1403-1406).
- Evaluated sample count: **4** (not 1 as previously misreported).
- `counts_consistent_with_stage27_summary`: **False in effect** -- the
  live `summary.json` on disk is stale (last written 2026-09-23, before
  #1406 was evaluated on 2026-09-29) and would disagree with a freshly
  rebuilt summary; Stage 31's own consistency flag, computed against a
  freshly rebuilt summary rather than the stale cached file, correctly
  reports `True` when compared to the freshly-rebuilt reference, which is
  exactly why Stage 31 never trusts the cached `summary.json` for its own
  counts (see Stage 31's design in Section B of the prior report).
- Per-signal metrics (4 evaluated draws each):
  - `production_pair_lr`: top5=0.35, top10=0.65, top15=0.75, mean winner
    rank=10.95 vs random 15.05 (advantage +4.10), classification
    `INSUFFICIENT_DATA`.
  - `pair_strength_direct`: identical to `production_pair_lr` in this
    sample (top5=0.35, top10=0.65, top15=0.75, mean rank 10.95, advantage
    +4.10), classification `INSUFFICIENT_DATA`.
  - `frequency_20`: top5=0.25, top10=0.30, top15=0.55, mean winner rank
    15.40 vs random 15.05 (advantage -0.35), classification
    `INSUFFICIENT_DATA`.
  - `paired_random` (control): top5=0.20, top10=0.35, top15=0.60, mean
    rank 15.05, classification `INSUFFICIENT_DATA`.
- Evidence-gate status: **not reached for any signal** (4 evaluated draws
  is far below the 10-draw floor even for `EARLY_TRACKING`, let alone the
  50-draw, p<0.05 preregistered gate). `any_signal_reaches_evidence_gate =
  False`. No early significance declaration, no optional stopping --
  consistent with the preregistered design.
- Stage 29 registration preview: `gate_reached=False`,
  `signals_ready_for_registration=[]`, `action_required: "none; evidence
  gate not yet reached for any signal"`.
- `predictive_edge_currently_supported`: **False.**

## 6. Does Stage 31 (or Stage 27) have a genuine bug?

**No.** This was tested directly, not assumed. `freeze_next_stage27_record`
-- the exact function that would need to run to produce a `1407.json` --
was executed against the real, live 864-draw history (through #1406) in an
isolated, writable copy of the live Stage 27 directory. It succeeded on
the first attempt and produced a valid, correctly-dated, correctly-hashed
`1407.json` record with no errors. This proves Stage 27/31's freeze logic,
as currently committed, has no path-resolution bug, no root-mismatch bug,
and no defect that would prevent #1407 from being frozen. **No code
change and no regression test were made, because there is nothing in
Stage 31 or Stage 27 to fix.**

The actual, separate finding -- outside Stage 31/27's scope to fix, and
not touched here -- is in the production automation layer
(`backend/app/research/automation.py`):

- `run_automation_once` -> `_run_due_cycle` -> `runner(...)` where `runner
  = cycle_runner or run_post_draw_cycle` (line 210). This **does** delegate
  to the Stage 27-including cycle by default, so Stage 27's evaluate step
  for #1406 almost certainly ran as part of this path on 2026-09-29 (its
  effects -- `1406.json` becoming `EVALUATED` -- are on disk with a
  timestamp consistent with that run). The freeze step for #1407 and the
  summary rebuild/save step that should have followed in the very same
  `run_stage27_cycle` call did not leave any trace on disk, and
  `summary.json`'s mtime (2026-09-23) predates that run entirely.
- `_run_due_cycle`'s own return payload (lines 245-256) is a hand-built
  dict containing only `lottery, action, result_source_status,
  history_update, prediction_evaluation, settlement, next_prediction,
  next_run_at, warnings, errors`. **It does not include a `stage27` key at
  all**, even though `OperationalCycleResult` carries one. This is a
  pre-existing observability gap in the automation layer, confirmed by
  reading the full file: whatever Stage 27 sub-result the operational
  cycle produced on 2026-09-29 -- success or the `{"status": "ERROR", ...}`
  payload that `_stage27_cycle_payload` (operational_cycle.py:313-342)
  returns when it catches a `ResearchValidationError` -- it is never
  serialized into the saved `AUTO-*.json` run record. This fully explains
  why none of the real `data/automation/runs/*.json` files contain a
  `stage27` key, for either lottery, in any run.
- Because the freeze step for #1407 could have failed inside that one
  specific 2026-09-29 run for a reason that does not reproduce against the
  current data (an environment/dependency difference at that exact
  moment, a concurrent-run/lock interaction, or some other transient
  condition), and because the automation's own run record cannot be
  consulted to see what happened (per the gap above), the precise trigger
  for that one failure cannot be determined after the fact from the data
  available in this investigation. What can be stated with certainty is
  that it is not reproducible against the current code and data, and it is
  not a Stage 31/27 defect.
- **Consequence if left unaddressed:** Stage 27's own
  `record_missed_draws` logic will, the first time a future cycle runs
  with draw #1407 already present in canonical history and still no
  `1407.json` on disk, add 1407 to `metadata.json`'s `missed_draws` list
  permanently -- #1407 would never get a frozen ranking and would be
  excluded from all four signals' evaluation samples going forward. This
  is a real, live operational-integrity risk worth the user's attention,
  but fixing the automation layer is explicitly out of scope for this
  research task ("do not change prediction logic") and was not touched.

## 7. Safety protocol

Protected-file SHA256 hashes were captured before any action in this
session and re-verified identical afterward:

| File | Hash (before = after) |
|---|---|
| `data/processed/mini_loto_history.csv` | `c069c504ed0c400bf4a7b47f3ba3a893c0d582007b0ebce91bb0bd90c322acfc` |
| `data/predictions/MINI_LOTO/ledger.json` | `cf0a8f9fc55b726442804538e705d3c04893f47c48dbe09ae62f7fb60ee5fa94` |
| `data/settlements/ledger.json` | `676e79adedb55699821343c204ff6ac953d6074ef1daac2d09b319fc71940afe` |
| `data/prospective/stage27/MINI_LOTO/summary.json` | `3904381225e07a80698526fc5edb3b2af37630d19a3ef64acebf9eef410c900b` |
| `data/predictions/MINI_LOTO/1404.json` | `7e8e7a5ba511c6ac89943e6b60f80afe6cf458d01e41accaf9f7931de2269fbb` |
| `data/predictions/MINI_LOTO/1405.json` | `353c93dbfa1932e6dc06e690275682fd908364b518ec442a469b223d3958f348` |
| `data/predictions/MINI_LOTO/1406.json` | `553c9d9f30e89c45ebb7cb43e36bbccc7d4ea9b996f4df0677b1ed259fb80c61` |
| `data/predictions/MINI_LOTO/1407.json` | `2b5e7f19b3e99248afee5b49991fae0cf41d14bffd353dca8848040282000899` |
| `data/settlements/MINI_LOTO/1404.json` | `fc99a10687d112e0f5413a658ccebc985ac60d3c16f058af0a58061f4e515a7a` |
| `data/settlements/MINI_LOTO/1405.json` | `40eb56bcb43d694aaf5953035c62a182355da45a9024118dcd33de61d7f44e47` |
| `data/settlements/MINI_LOTO/1406.json` | `1d9626ac79d6db17e98919ce520ffd3ff5d6c10c0aea43671460b02be3636fc8` |
| `data/prospective/stage27/MINI_LOTO/1403.json` | `1b72664ca0a524dfaaa9683d37515e47612bc1f13bd4f7da9ff9f63d91135a49` |
| `data/prospective/stage27/MINI_LOTO/1404.json` | `7ef4bb3dbe1c557af2f8b5231d147f74eea54382c2fab8549442b7cdb4a679db` |
| `data/prospective/stage27/MINI_LOTO/1405.json` | `6d5b58efb99e093ac85e66d1a5a5def24bfaeb93b20fddb723124143e30a8264` |
| `data/prospective/stage27/MINI_LOTO/1406.json` | `f641bc599c6047872bb0b1279162fde68262f30e40bcf2df95c01a7b8e75942a` |
| `data/prospective/stage27/MINI_LOTO/metadata.json` | `b83a765931380ac83b309b7c47267a345fb5a832b6508984f84a9b9014df0ddd` |

All investigation that needed the real data (loading the real CSV, running
`freeze_next_stage27_record`, `audit_stage27_health`, `build_stage31_summary`)
was executed against an isolated temp copy of the data, never against the
live repository paths. No `git add`, `git commit`, `git push`, `git reset`,
`git restore`, or `git clean` was run. `git status --short` on the live
repository shows only the same 372 lines of pre-existing `.pytest-tmp/`
churn already flagged (and not caused) by Stage 31's repository-hygiene
audit; no new changes appear.

## 8. Final Report (26 items)

1. **Root cause:** My own Stage 31 final report was computed against a
   stale cloud-sandbox clone (history frozen at #1403), not the live
   repository, and was mis-presented as a live measurement. This is a
   process error, not a Stage 31 code defect.
2. **Authoritative data root:** The live repository at
   `C:\Users\imcom\Downloads\loto-system` (reached via the device bridge),
   specifically `data/processed/mini_loto_history.csv` and
   `data/prospective/stage27/MINI_LOTO/`. The dashboard's own API
   (`backend/app/research/operations.py:230`) reads this same file.
3. **Actual latest canonical draw:** #1406 (2026-09-29, main
   `[11,14,27,30,31]`, bonus `[13]`).
4. **Actual latest production prediction:** #1407, `PENDING`, generated
   2026-09-29T23:46:13Z, built from history through #1406 only.
5. **Actual latest Stage 27 frozen target:** #1406 (frozen 2026-09-23,
   evaluated 2026-09-29). No frozen target exists yet for #1407.
6. **#1404 integrity status:** canonical result, prediction, settlement,
   and Stage 27 record all present and evaluated; freeze-hash,
   evaluation-hash, freeze-before-result, and wall-clock checks all pass.
7. **#1405 integrity status:** same as #1404 -- all checks pass.
8. **#1406 integrity status:** same as #1404/#1405 -- all checks pass.
9. **#1407 freeze status:** production prediction exists and is leak-free
   (history cutoff #1406, target #1407); Stage 27 frozen target does not
   exist. Not currently flagged as "missing" by Stage 31's own range-based
   definition, but will become a permanently recorded "missed" prospective
   target once #1407's result enters canonical history unless the freeze
   step runs successfully before then.
10. **#1406 payout verification:** Set 1 3 main matches -> 4th prize ->
    800 yen (independently recomputed from official result and confirmed
    against the settlement record); Set 2 2 main matches -> no prize;
    Set 3 0 matches -> no prize. Cost 600 yen, gross 800 yen, net +200 yen.
    Payout sourced from the official `smbc_public_result` feed, not
    simulated.
11. **Predictive-evidence interpretation of the #1406 win:** None. Stage
    30's fair-null baseline (P(>=1 prize) = 161/2697 ~= 5.9696% per draw)
    remains the reference; one win is consistent with chance.
12. **Current Stage 27 evaluated count:** 4 (draws 1403-1406), not 1.
13. **Current Stage 27 pending count:** 0.
14. **Current Stage 27 missing/retrospective count:** 0 missing, 0
    retrospective-leak-suspected.
15. **Integrity failures found:** 0, across all four evaluated targets.
16. **Current per-signal metrics (n=4 evaluated draws):**
    `production_pair_lr` top5/10/15 = 0.35/0.65/0.75, mean rank advantage
    +4.10; `pair_strength_direct` identical in this sample; `frequency_20`
    top5/10/15 = 0.25/0.30/0.55, mean rank advantage -0.35;
    `paired_random` (control) top5/10/15 = 0.20/0.35/0.60.
17. **Evidence-gate status:** Not reached for any signal.
    `INSUFFICIENT_DATA` for all four (below the 10-draw `EARLY_TRACKING`
    floor). No early significance declaration made.
18. **Predictive edge currently supported:** No.
    `predictive_edge_currently_supported = False`.
19. **Did Stage 31 require code modification:** No. Its freeze/audit logic
    was proven correct by direct execution against the real data.
20. **Exact files modified in the live repository by this investigation:**
    none. Two files were edited only in the Claude Project / local
    documentation copy (this document, and a correction note appended to
    `research/LotoSystem_V2_Stage31_Operational_Integrity_Monitor.md`) --
    neither is a Stage 27/29/30/31 source or data file.
21. **Targeted test results:** not applicable -- no source code changed,
    so Stage 31's existing 19/19 passing test suite from the prior stage
    stands unchanged.
22. **Full test-suite results:** not applicable for the same reason; the
    prior stage's full-suite baseline (383 passed, 1 pre-existing
    unrelated failure) stands unchanged.
23. **Ruff:** not applicable; no source files were touched.
24. **`git diff --check`:** not applicable; no tracked source file has any
    diff from this investigation.
25. **Protected-file hashes, before vs. after:** identical for all 16
    named files (4 core protected files plus the 12 #1404-1407-related
    prediction/settlement/Stage-27 files); see the table in Section 7.
26. **`git status --short` on the live repository:** unchanged from the
    pre-existing state -- 372 lines, all under `.pytest-tmp/` (disposable
    pytest fixture output already flagged, not caused, by Stage 31's own
    repository-hygiene audit). No Stage 31 files are new or modified, so
    there is nothing new to stage. **Nothing was staged or committed.**

## Independent finding carried forward (not fixed here)

The production automation layer (`backend/app/research/automation.py`)
appears to have failed to persist a Stage 27 freeze for #1407 during the
2026-09-29 cycle, and its own run-record schema has no field to record
that outcome either way. If this is not corrected before #1407's draw
result arrives (scheduled 2026-10-06), Stage 27's own missed-draw logic
will permanently exclude #1407 from all four signals' tracked samples.
This is a production/automation-layer finding, outside Stage 27/29/30/31's
research scope, and was deliberately not fixed as part of this task.
