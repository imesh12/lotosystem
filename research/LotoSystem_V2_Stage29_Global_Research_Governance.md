# LotoSystem V2 Stage 29: Global Research Governance Ledger

## Purpose

Stage 29 creates a cross-stage research governance ledger for LotoSystem. It
does not create a model, tune features, change production strategy, regenerate
predictions, modify Stage 27 prospective records, or alter canonical history,
settlements, scheduler, email, frontend, or configuration.

The governing question is:

```text
After accounting for the verified hypothesis-testing history of this project,
does any predictive result survive a global multiple-testing correction?
```

## Global Correction Family

The authoritative family is:

```text
Formal predictive or discovery hypotheses that were tested against historical
or prospective result data, produced a numeric raw p-value, could have
influenced a research conclusion or model-selection decision, have verifiable
code/output provenance, and are not duplicate reprints of the same test.
```

Included entries must have numeric raw p-values and defensible provenance.

Excluded entries remain visible in the audit ledger, but do not enter the
global correction family when they are:

- descriptive metrics with no hypothesis test
- report-only or unresolved claims
- duplicate reprints of the same test
- infrastructure or unit-test behavior
- already-adjusted p-values being repeated as if raw
- untestable placeholders such as Stage 28 with zero usable observations
- prospective records below their evidence gate

## Correction Methods

Stage 29 applies:

- Holm-Bonferroni as the authoritative global correction
- Benjamini-Hochberg as exploratory FDR reporting

Both are computed from raw p-values only. Local/stage-adjusted p-values are
preserved for audit context and never recursively corrected.

Tie handling is deterministic. Hypotheses are ordered by stage, lottery,
experiment family, hypothesis name, metric, and stable hypothesis ID. P-value
adjustments sort by p-value and stable hypothesis ID.

## Provenance Status

Each audit row records provenance explicitly:

- `VERIFIED_CODE_AND_OUTPUT`
- `VERIFIED_CODE_ONLY`
- `VERIFIED_OUTPUT_ONLY`
- `REPORT_ONLY`
- `LEGACY_OR_ABANDONED`
- `UNRESOLVED`
- `UNTESTABLE_CURRENT_DATA`
- `PROSPECTIVE_ACTIVE`

Only verified numeric entries can enter the authoritative family.

## Stage 24 Audit

Stage 24 is represented from the current live code/output:

- Code: `backend/app/research/stage24_temporal_research.py`
- Output: `data/exports/stage24/v2_stage24_temporal_research.json`
- Report: `research/LotoSystem_V2_Stage24_Temporal_Discovery.md`

The live output contains nine temporal signal tests. The committed report
documents the obsolete stale-ingestion hypothesis as contradicted by live
evidence, and the frozen decision identifies
`short_window_regime_concentration` as the strongest temporal signal while
classifying it as `NO_EVIDENCE`.

Stage 29 does not edit Stage 24. It registers the verified numeric Stage 24
signal tests as historical discovery hypotheses.

## Stage 27 Handling

Stage 27 is prospective and remains untouched. Current Stage 27 summary state
is active but below evidence gates. Its records are audited as
`PROSPECTIVE_ACTIVE` and excluded from global correction until legitimate
evaluated prospective observations produce formal p-values under the Stage 27
rules.

This report notes a future governance concern: repeated future looks after a
formal threshold can create optional-stopping exposure. Stage 29A documents
that risk but does not change Stage 27 behavior.

## Stage 28 / 28B Handling

Stage 28's popularity-vs-prize-split hypothesis is currently untestable because
there are no local `sales_amount_yen` observations. The safe placeholder
p-values in Stage 28 output are not treated as real hypothesis tests.

Stage 28B acquired available winner-count and payout fields from local
settlement copies but still found zero sales observations. It is audited as
data acquisition, not as a predictive hypothesis test.

## Current Result

The generated Stage 29 ledger is written under:

```text
data/exports/stage29/
```

Current audited result:

- total audit records: 199
- formal numeric hypotheses found: 187
- authoritative hypotheses included in global correction: 180
- duplicates excluded: 3
- report-only or unverified excluded: 3
- untestable excluded: 1
- prospective-not-yet-evaluable excluded: 4
- raw p < 0.05: 3
- original/local Holm surviving p < 0.05: 0
- global Holm surviving p < 0.05: 0
- global BH surviving p < 0.05: 0

The strongest raw p-value is a Stage 26 Mini Loto feature-information endpoint:

```text
frequency_momentum_5_20_top5_capture_rate
raw p = 0.01186704057853983
global Holm p = 1.0
global BH p = 0.8363163683631637
global classification = NO_EVIDENCE
```

## Scientific Conclusion

No predictive historical result currently survives authoritative global
Holm correction.

Mini Loto conclusion:

```text
NO_EVIDENCE after global correction.
```

LOTO6 conclusion:

```text
NO_EVIDENCE after global correction.
```

This is a successful governance result: it lowers false-discovery risk and
sets the standard for future preregistered experiments.

## Future Registration API

Stage 29 adds a small governance primitive for future stages:

- `register_hypothesis(...)`
- `record_result(...)`
- `compute_global_corrections(...)`
- `build_repository_global_ledger(...)`
- `save_global_ledger(...)`
- `load_global_ledger(...)`

Future stages should preregister hypotheses before execution, record raw
p-values once, and let Stage 29 compute global corrections across the verified
family.
