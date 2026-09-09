from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.statistical_evaluation import holm_adjust_p_values

STAGE29_SCHEMA_VERSION = "v2-stage29-global-experiment-ledger-v1"
DEFAULT_STAGE29_OUTPUT_DIR = Path("data") / "exports" / "stage29"
GLOBAL_FAMILY_ID = "verified_predictive_research_hypotheses_v1"
GLOBAL_FAMILY_DEFINITION = (
    "Formal predictive or discovery hypotheses that were tested against historical or "
    "prospective result data, produced a numeric raw p-value, could have influenced a "
    "research conclusion or model-selection decision, have verifiable code/output "
    "provenance, and are not duplicate reprints of the same test."
)

VERIFIED_CODE_AND_OUTPUT = "VERIFIED_CODE_AND_OUTPUT"
VERIFIED_OUTPUT_ONLY = "VERIFIED_OUTPUT_ONLY"
REPORT_ONLY = "REPORT_ONLY"
UNTESTABLE_CURRENT_DATA = "UNTESTABLE_CURRENT_DATA"
PROSPECTIVE_ACTIVE = "PROSPECTIVE_ACTIVE"

EXACT_DUPLICATE = "EXACT_DUPLICATE"
SAME_HYPOTHESIS_DIFFERENT_REPORT = "SAME_HYPOTHESIS_DIFFERENT_REPORT"
SAME_RAW_TEST_DIFFERENT_LABEL = "SAME_RAW_TEST_DIFFERENT_LABEL"
DISTINCT_HYPOTHESIS = "DISTINCT_HYPOTHESIS"


@dataclass(frozen=True, slots=True)
class GlobalHypothesisRecord:
    schema_version: str
    hypothesis_id: str
    stage: str
    lottery: str
    experiment_family: str
    hypothesis_name: str
    description: str
    metric: str
    direction: str
    sample_size: int | None
    effect: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    raw_p_value: float | None
    original_holm_p_value: float | None
    original_bh_p_value: float | None
    original_classification: str | None
    source_file: str | None
    source_report: str | None
    source_output: str | None
    provenance_status: str
    historical_or_prospective: str
    discovery_or_confirmation: str
    dataset_cutoff_draw: int | None
    dataset_hash: str | None
    preregistered: bool
    primary_or_secondary: str
    notes: str
    duplicate_status: str = DISTINCT_HYPOTHESIS
    global_family_id: str | None = GLOBAL_FAMILY_ID
    eligible_for_global_correction: bool = False
    global_holm_p_value: float | None = None
    global_bh_p_value: float | None = None
    global_classification: str | None = None
    classification_changed: bool = False
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GlobalLedgerSummary:
    total_audit_records: int
    formal_numeric_hypotheses_found: int
    authoritative_hypotheses_included: int
    duplicates_excluded: int
    report_only_or_unverified_excluded: int
    untestable_excluded: int
    prospective_not_yet_evaluable_excluded: int
    raw_significant_count: int
    original_holm_surviving_count: int
    global_holm_surviving_count: int
    global_bh_surviving_count: int
    classification_changes: tuple[dict[str, Any], ...]
    strongest_by_raw_p: str | None
    strongest_by_global_holm: str | None
    any_predictive_result_survives_global_holm: bool


@dataclass(frozen=True, slots=True)
class GlobalResearchLedger:
    schema_version: str
    global_family_id: str
    global_family_definition: str
    holm_implementation: str
    bh_implementation: str
    records: tuple[GlobalHypothesisRecord, ...]
    summary: GlobalLedgerSummary
    stages_audited: tuple[str, ...]
    source_files_inspected: tuple[str, ...]
    source_reports_inspected: tuple[str, ...]
    source_outputs_inspected: tuple[str, ...]


def deterministic_hypothesis_id(
    *,
    stage: str,
    lottery: str,
    experiment_family: str,
    hypothesis_name: str,
    metric: str,
    source_output: str | None = None,
) -> str:
    payload = "|".join(
        (stage, lottery, experiment_family, hypothesis_name, metric, source_output or "")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"GH-{digest[:16]}"


def make_hypothesis_record(
    *,
    stage: str,
    lottery: str,
    experiment_family: str,
    hypothesis_name: str,
    description: str,
    metric: str,
    direction: str = "positive_is_better",
    sample_size: int | None = None,
    effect: float | None = None,
    confidence_interval_low: float | None = None,
    confidence_interval_high: float | None = None,
    raw_p_value: float | None = None,
    original_holm_p_value: float | None = None,
    original_bh_p_value: float | None = None,
    original_classification: str | None = None,
    source_file: str | None = None,
    source_report: str | None = None,
    source_output: str | None = None,
    provenance_status: str = VERIFIED_OUTPUT_ONLY,
    historical_or_prospective: str = "historical",
    discovery_or_confirmation: str = "discovery",
    dataset_cutoff_draw: int | None = None,
    dataset_hash: str | None = None,
    preregistered: bool = False,
    primary_or_secondary: str = "secondary",
    notes: str = "",
    eligible_for_global_correction: bool | None = None,
    exclusion_reason: str | None = None,
) -> GlobalHypothesisRecord:
    if raw_p_value is not None and not 0 <= raw_p_value <= 1:
        raise ValueError("raw_p_value must be in 0..1")
    eligible = (
        raw_p_value is not None
        and provenance_status in {VERIFIED_CODE_AND_OUTPUT, VERIFIED_OUTPUT_ONLY}
        and exclusion_reason is None
        if eligible_for_global_correction is None
        else eligible_for_global_correction
    )
    if not eligible and exclusion_reason is None:
        exclusion_reason = _default_exclusion_reason(provenance_status, raw_p_value)
    return GlobalHypothesisRecord(
        schema_version=STAGE29_SCHEMA_VERSION,
        hypothesis_id=deterministic_hypothesis_id(
            stage=stage,
            lottery=lottery,
            experiment_family=experiment_family,
            hypothesis_name=hypothesis_name,
            metric=metric,
            source_output=source_output,
        ),
        stage=stage,
        lottery=lottery,
        experiment_family=experiment_family,
        hypothesis_name=hypothesis_name,
        description=description,
        metric=metric,
        direction=direction,
        sample_size=sample_size,
        effect=effect,
        confidence_interval_low=confidence_interval_low,
        confidence_interval_high=confidence_interval_high,
        raw_p_value=raw_p_value,
        original_holm_p_value=original_holm_p_value,
        original_bh_p_value=original_bh_p_value,
        original_classification=original_classification,
        source_file=source_file,
        source_report=source_report,
        source_output=source_output,
        provenance_status=provenance_status,
        historical_or_prospective=historical_or_prospective,
        discovery_or_confirmation=discovery_or_confirmation,
        dataset_cutoff_draw=dataset_cutoff_draw,
        dataset_hash=dataset_hash,
        preregistered=preregistered,
        primary_or_secondary=primary_or_secondary,
        notes=notes,
        eligible_for_global_correction=eligible,
        exclusion_reason=exclusion_reason,
    )


def compute_global_corrections(
    records: tuple[GlobalHypothesisRecord, ...] | list[GlobalHypothesisRecord],
) -> GlobalResearchLedger:
    ordered = tuple(sorted(records, key=_record_sort_key))
    duplicate_marked = _mark_duplicates(ordered)
    eligible = tuple(
        record
        for record in duplicate_marked
        if record.eligible_for_global_correction
        and record.duplicate_status == DISTINCT_HYPOTHESIS
        and record.raw_p_value is not None
    )
    raw = {record.hypothesis_id: float(record.raw_p_value) for record in eligible}
    holm = holm_adjust_p_values(raw)
    bh = benjamini_hochberg_adjust_p_values(raw)
    corrected_by_id = {}
    for record in eligible:
        global_holm = holm[record.hypothesis_id]
        global_bh = bh[record.hypothesis_id]
        global_classification = classify_global_result(
            effect=record.effect,
            global_holm_p_value=global_holm,
            original_classification=record.original_classification,
        )
        corrected_by_id[record.hypothesis_id] = replace(
            record,
            global_holm_p_value=global_holm,
            global_bh_p_value=global_bh,
            global_classification=global_classification,
            classification_changed=_normalize_classification(record.original_classification)
            != _normalize_classification(global_classification),
        )
    corrected = tuple(
        corrected_by_id[record.hypothesis_id]
        if record.eligible_for_global_correction
        and record.duplicate_status == DISTINCT_HYPOTHESIS
        and record.hypothesis_id in corrected_by_id
        else record
        for record in duplicate_marked
    )
    return GlobalResearchLedger(
        schema_version=STAGE29_SCHEMA_VERSION,
        global_family_id=GLOBAL_FAMILY_ID,
        global_family_definition=GLOBAL_FAMILY_DEFINITION,
        holm_implementation=(
            "Holm-Bonferroni over eligible raw p-values sorted by p-value then "
            "hypothesis_id; adjusted values are monotone running maxima."
        ),
        bh_implementation=(
            "Benjamini-Hochberg exploratory FDR adjustment over eligible raw p-values "
            "sorted by p-value then hypothesis_id; reported separately from Holm."
        ),
        records=corrected,
        summary=summarize_global_ledger(corrected),
        stages_audited=tuple(sorted({record.stage for record in corrected}, key=_stage_sort_key)),
        source_files_inspected=tuple(
            sorted({record.source_file for record in corrected if record.source_file})
        ),
        source_reports_inspected=tuple(
            sorted({record.source_report for record in corrected if record.source_report})
        ),
        source_outputs_inspected=tuple(
            sorted({record.source_output for record in corrected if record.source_output})
        ),
    )


def classify_global_result(
    *,
    effect: float | None,
    global_holm_p_value: float,
    original_classification: str | None,
) -> str:
    normalized = _normalize_classification(original_classification)
    if effect is not None and effect < 0 and global_holm_p_value < 0.05:
        return "NEGATIVE"
    if global_holm_p_value < 0.05:
        return "EVIDENCE"
    if global_holm_p_value < 0.10 and normalized in {"WEAK_SIGNAL", "EVIDENCE"}:
        return "WEAK_SIGNAL"
    return "NO_EVIDENCE"


def summarize_global_ledger(records: tuple[GlobalHypothesisRecord, ...]) -> GlobalLedgerSummary:
    numeric = tuple(record for record in records if record.raw_p_value is not None)
    included = tuple(
        record
        for record in records
        if record.eligible_for_global_correction and record.duplicate_status == DISTINCT_HYPOTHESIS
    )
    classification_changes = tuple(
        {
            "hypothesis_id": record.hypothesis_id,
            "stage": record.stage,
            "lottery": record.lottery,
            "hypothesis_name": record.hypothesis_name,
            "original_classification": record.original_classification,
            "global_classification": record.global_classification,
            "raw_p_value": record.raw_p_value,
            "global_holm_p_value": record.global_holm_p_value,
        }
        for record in records
        if record.classification_changed
    )
    raw_significant = tuple(
        record
        for record in included
        if record.raw_p_value is not None and record.raw_p_value < 0.05
    )
    original_holm = tuple(
        record
        for record in included
        if record.original_holm_p_value is not None and record.original_holm_p_value < 0.05
    )
    global_holm = tuple(
        record
        for record in included
        if record.global_holm_p_value is not None and record.global_holm_p_value < 0.05
    )
    global_bh = tuple(
        record
        for record in included
        if record.global_bh_p_value is not None and record.global_bh_p_value < 0.05
    )
    return GlobalLedgerSummary(
        total_audit_records=len(records),
        formal_numeric_hypotheses_found=len(numeric),
        authoritative_hypotheses_included=len(included),
        duplicates_excluded=sum(
            record.duplicate_status != DISTINCT_HYPOTHESIS for record in records
        ),
        report_only_or_unverified_excluded=sum(
            record.provenance_status == REPORT_ONLY for record in records
        ),
        untestable_excluded=sum(
            record.provenance_status == UNTESTABLE_CURRENT_DATA
            or record.exclusion_reason == "NO_USABLE_OBSERVATIONS"
            for record in records
        ),
        prospective_not_yet_evaluable_excluded=sum(
            record.provenance_status == PROSPECTIVE_ACTIVE for record in records
        ),
        raw_significant_count=len(raw_significant),
        original_holm_surviving_count=len(original_holm),
        global_holm_surviving_count=len(global_holm),
        global_bh_surviving_count=len(global_bh),
        classification_changes=classification_changes,
        strongest_by_raw_p=_strongest(included, "raw_p_value"),
        strongest_by_global_holm=_strongest(included, "global_holm_p_value"),
        any_predictive_result_survives_global_holm=bool(global_holm),
    )


def build_repository_global_ledger(
    *,
    export_root: str | Path = Path("data") / "exports",
    research_root: str | Path = Path("research"),
) -> GlobalResearchLedger:
    export_root = Path(export_root)
    research_root = Path(research_root)
    records: list[GlobalHypothesisRecord] = []
    records.extend(_stage05_records(export_root))
    records.extend(_stage06_records(export_root))
    records.extend(_stage07_records(export_root))
    records.extend(_stage08_records(export_root))
    records.extend(_stage09_records(export_root))
    records.extend(_stage18_records(export_root))
    records.extend(_stage19_records(export_root))
    records.extend(_stage20_records(export_root))
    records.extend(_stage21_records())
    records.extend(_stage22_records(export_root))
    records.extend(_stage23_records())
    records.extend(_stage24_records(export_root, research_root))
    records.extend(_stage25_records(export_root, research_root))
    records.extend(_stage26_records(export_root, research_root))
    records.extend(_stage27_records(Path("data") / "prospective" / "stage27"))
    records.extend(_stage28_records(export_root, research_root))
    records.extend(_stage28b_records(export_root, research_root))
    return compute_global_corrections(tuple(records))


def save_global_ledger(
    ledger: GlobalResearchLedger,
    output_dir: str | Path = DEFAULT_STAGE29_OUTPUT_DIR,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "global_research_ledger.json"
    summary_path = destination / "global_research_ledger_summary.json"
    csv_path = destination / "global_research_ledger.csv"
    ledger_path.write_text(research_result_json(ledger), encoding="utf-8")
    summary_path.write_text(research_result_json(ledger.summary), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = tuple(to_jsonable(ledger.records[0]).keys()) if ledger.records else ()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in ledger.records:
            writer.writerow(to_jsonable(record))
    return {"ledger": str(ledger_path), "summary": str(summary_path), "csv": str(csv_path)}


def load_global_ledger(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def register_hypothesis(**kwargs: Any) -> GlobalHypothesisRecord:
    return make_hypothesis_record(**kwargs)


def record_result(record: GlobalHypothesisRecord, **updates: Any) -> GlobalHypothesisRecord:
    allowed = {
        "sample_size",
        "effect",
        "confidence_interval_low",
        "confidence_interval_high",
        "raw_p_value",
        "original_holm_p_value",
        "original_bh_p_value",
        "original_classification",
        "source_output",
        "provenance_status",
        "notes",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"unsupported result fields: {sorted(unknown)}")
    return replace(record, **updates)


def _stage05_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    paths = sorted(export_root.glob("stage05*_baseline_report.json"))
    return tuple(
        make_hypothesis_record(
            stage="05",
            lottery=_json(path).get("lottery", "ALL"),
            experiment_family="real_random_baseline",
            hypothesis_name="baseline_distribution",
            description="Stage 05 establishes random baseline metrics without a formal p-value.",
            metric="descriptive_baseline",
            source_file="backend/app/research/baseline_benchmark.py",
            source_output=_rel(path),
            provenance_status=VERIFIED_CODE_AND_OUTPUT,
            eligible_for_global_correction=False,
            exclusion_reason="DESCRIPTIVE_NO_HYPOTHESIS_TEST",
        )
        for path in paths
    )


def _stage06_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    records: list[GlobalHypothesisRecord] = []
    for path in sorted(export_root.glob("stage06_*_statistical_evaluation.json")):
        if path.name == "stage06_statistical_evaluation.json":
            continue
        data = _json(path)
        for strategy, payload in data.get("strategies", {}).items():
            for metric_key, metric_name in (
                ("mean_matches", "average_matches_per_ticket"),
                ("prize_qualified_rate", "prize_qualified_rate"),
            ):
                comparison = payload.get(metric_key, {})
                records.append(
                    _comparison_record(
                        stage="06",
                        lottery=data["lottery"],
                        family="deterministic_strategy_vs_random",
                        name=f"{strategy}_vs_random",
                        description=f"Stage 06 {strategy} strategy vs paired random baseline.",
                        metric=metric_name,
                        comparison=comparison,
                        sample_size=payload.get("target_draws"),
                        original_classification=payload.get("conclusion"),
                        source_file="backend/app/research/statistical_evaluation.py",
                        source_output=_rel(path),
                        dataset_hash=data.get("dataset_hash"),
                        dataset_cutoff_draw=_last_draw(data),
                        preregistered=True,
                        primary_or_secondary="primary"
                        if metric_key == "mean_matches"
                        else "secondary",
                    )
                )
    return tuple(records)


def _stage07_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    records: list[GlobalHypothesisRecord] = []
    for path in sorted(export_root.glob("stage07_*_ml_baseline.json")):
        data = _json(path)
        for model, payload in data.get("models", {}).items():
            records.append(
                _comparison_record(
                    stage="07",
                    lottery=data["lottery"],
                    family="first_ml_baseline_vs_random",
                    name=f"{model}_vs_random",
                    description=f"Stage 07 {model} ML baseline vs paired random baseline.",
                    metric="average_matches_per_ticket",
                    comparison=payload.get("mean_matches", {}),
                    sample_size=payload.get("metrics", {}).get("draws_evaluated"),
                    original_classification=payload.get("conclusion"),
                    source_file="backend/app/research/ml_baseline.py",
                    source_output=_rel(path),
                    dataset_hash=data.get("dataset_hash"),
                    dataset_cutoff_draw=_last_draw(data),
                    preregistered=True,
                    primary_or_secondary="primary",
                )
            )
    return tuple(records)


def _stage08_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    records: list[GlobalHypothesisRecord] = []
    for path in sorted(export_root.glob("stage08_*_feature_evaluation.json")):
        data = _json(path)
        for group, group_payload in data.get("ablation_results", {}).items():
            for model, payload in group_payload.get("models", {}).items():
                records.append(
                    _comparison_record(
                        stage="08",
                        lottery=data["lottery"],
                        family="feature_group_ablation_vs_random",
                        name=f"{group}_{model}_vs_random",
                        description=f"Stage 08 {group} feature group with {model} vs random.",
                        metric="average_matches_per_ticket",
                        comparison=payload.get("mean_matches", {}),
                        sample_size=payload.get("metrics", {}).get("draws_evaluated"),
                        original_classification=payload.get("conclusion"),
                        source_file="backend/app/research/feature_evaluation.py",
                        source_output=_rel(path),
                        dataset_hash=data.get("dataset_hash"),
                        dataset_cutoff_draw=_last_draw(data),
                        preregistered=True,
                        primary_or_secondary="primary",
                    )
                )
    return tuple(records)


def _stage09_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    records: list[GlobalHypothesisRecord] = []
    for path in sorted(export_root.glob("stage09_*_portfolio_evaluation.json")):
        data = _json(path)
        for method, payload in data.get("method_results", {}).items():
            records.append(
                _comparison_record(
                    stage="09",
                    lottery=data["lottery"],
                    family="portfolio_method_vs_random",
                    name=f"{method}_portfolio_vs_random",
                    description=f"Stage 09 {method} portfolio construction vs random.",
                    metric="average_matches_per_portfolio",
                    comparison=payload.get("comparison_vs_random", {}),
                    sample_size=payload.get("metrics", {}).get("draws_evaluated"),
                    original_classification=payload.get("conclusion"),
                    source_file="backend/app/research/portfolio_evaluation.py",
                    source_output=_rel(path),
                    dataset_hash=data.get("dataset_hash"),
                    dataset_cutoff_draw=_last_draw(data),
                    preregistered=True,
                    primary_or_secondary="primary",
                )
            )
    return tuple(records)


def _stage18_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "experiments" / "v2_experiment_ledger.json"
    if not path.exists():
        return (
            make_hypothesis_record(
                stage="18",
                lottery="ALL",
                experiment_family="experiment_governance",
                hypothesis_name="stage18_ledger_absent",
                description="Stage 18 governance design not found as an output ledger.",
                metric="governance",
                provenance_status=REPORT_ONLY,
                eligible_for_global_correction=False,
                exclusion_reason="GOVERNANCE_INFRASTRUCTURE_NO_RAW_TEST",
            ),
        )
    data = _json(path)
    governance_record = make_hypothesis_record(
        stage="18",
        lottery="ALL",
        experiment_family="experiment_governance",
        hypothesis_name="v2_experiment_ledger_design",
        description="Stage 18 provides local experiment-ledger governance, not a predictive test.",
        metric="governance",
        source_file="backend/app/research/extra_trees_evaluation.py",
        source_output=_rel(path),
        provenance_status=VERIFIED_OUTPUT_ONLY,
        eligible_for_global_correction=False,
        exclusion_reason="GOVERNANCE_INFRASTRUCTURE_NO_RAW_TEST",
    )
    entry_records = tuple(
        make_hypothesis_record(
            stage=str(entry.get("stage", "18")),
            lottery=entry.get("lottery", "UNKNOWN"),
            experiment_family="stage18_v2_experiment_ledger",
            hypothesis_name=entry.get("hypothesis", entry.get("comparison", "unknown")),
            description=entry.get("hypothesis", "Stage 18 ledger entry."),
            metric=entry.get("comparison", "ledger_entry"),
            sample_size=None,
            effect=None,
            raw_p_value=_float_or_none(entry.get("raw_p_value")),
            original_holm_p_value=_float_or_none(entry.get("ledger_adjusted_p_value")),
            original_bh_p_value=_float_or_none(entry.get("bh_exploratory_p_value")),
            original_classification=entry.get("conclusion") or entry.get("status"),
            source_file="backend/app/research/extra_trees_evaluation.py",
            source_output=_rel(path),
            provenance_status=VERIFIED_OUTPUT_ONLY,
            dataset_hash=entry.get("dataset_hash"),
            preregistered=True,
            primary_or_secondary="primary",
            notes="Stage 18 ledger entry retained; duplicate status may exclude reprinted tests.",
        )
        for entry in data.get("entries", [])
    )
    return (governance_record, *entry_records)


def _stage19_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    return (
        make_hypothesis_record(
            stage="19",
            lottery="ALL",
            experiment_family="prospective_ledger",
            hypothesis_name="prospective_evaluation_diagnostics",
            description="Stage 19 creates prospective diagnostics; no formal global p-value output found.",  # noqa: E501
            metric="prospective_diagnostic",
            source_file="backend/app/research/prospective.py",
            provenance_status=REPORT_ONLY,
            eligible_for_global_correction=False,
            exclusion_reason="NO_FORMAL_NUMERIC_HYPOTHESIS_OUTPUT_FOUND",
        ),
    )


def _stage20_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    records: list[GlobalHypothesisRecord] = []
    for path in sorted(export_root.glob("v2_stage20_*_extra_trees.json")):
        data = _json(path)
        for model_key in ("current_champion", "extra_trees"):
            payload = data.get(model_key, {})
            if payload.get("comparison_vs_random"):
                records.append(
                    _comparison_record(
                        stage="20",
                        lottery=data["lottery"],
                        family="extra_trees_robustness_vs_random",
                        name=f"{model_key}_vs_random",
                        description=f"Stage 20 {model_key} vs paired random.",
                        metric="average_matches_per_ticket",
                        comparison=payload["comparison_vs_random"],
                        sample_size=payload.get("sample_size"),
                        original_classification=payload.get("conclusion"),
                        source_file="backend/app/research/extra_trees_evaluation.py",
                        source_output=_rel(path),
                        dataset_hash=data.get("dataset_hash"),
                        dataset_cutoff_draw=_last_draw(data),
                        preregistered=True,
                        primary_or_secondary="primary",
                    )
                )
            if payload.get("comparison_vs_champion"):
                records.append(
                    _comparison_record(
                        stage="20",
                        lottery=data["lottery"],
                        family="extra_trees_robustness_vs_champion",
                        name=f"{model_key}_vs_champion",
                        description=f"Stage 20 {model_key} direct paired comparison vs champion.",
                        metric="average_matches_per_ticket",
                        comparison=payload["comparison_vs_champion"],
                        sample_size=payload.get("sample_size"),
                        original_classification=payload.get("conclusion"),
                        source_file="backend/app/research/extra_trees_evaluation.py",
                        source_output=_rel(path),
                        dataset_hash=data.get("dataset_hash"),
                        dataset_cutoff_draw=_last_draw(data),
                        preregistered=True,
                        primary_or_secondary="primary",
                    )
                )
    return tuple(records)


def _stage21_records() -> tuple[GlobalHypothesisRecord, ...]:
    return (
        make_hypothesis_record(
            stage="21",
            lottery="ALL",
            experiment_family="shadow_challenger_framework",
            hypothesis_name="shadow_framework_infrastructure",
            description="Stage 21 is framework-only and registers no predictive hypothesis test.",
            metric="infrastructure",
            source_file="backend/app/research/shadow.py",
            provenance_status=REPORT_ONLY,
            eligible_for_global_correction=False,
            exclusion_reason="INFRASTRUCTURE_NO_HYPOTHESIS_TEST",
        ),
    )


def _stage22_records(export_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "v2_stage22_mini_loto_pair_network.json"
    if not path.exists():
        return ()
    data = _json(path)
    return (
        _comparison_record(
            stage="22",
            lottery=data["lottery"],
            family="pair_network_feature_vs_champion",
            name="pair_network_v1_vs_pair_only_champion",
            description="Stage 22 pair-network feature challenger vs champion.",
            metric="average_matches_per_ticket",
            comparison=data["challenger_vs_champion"],
            sample_size=data.get("challenger", {}).get("metrics", {}).get("draws_evaluated"),
            original_classification=data.get("governance", {}).get("unified_conclusion"),
            source_file="backend/app/research/stage22_pair_network.py",
            source_report="research/LotoSystem_V2_Stage22_Pair_Network.md",
            source_output=_rel(path),
            dataset_hash=data.get("dataset_hash"),
            dataset_cutoff_draw=_last_draw(data),
            preregistered=True,
            primary_or_secondary="primary",
        ),
        _comparison_record(
            stage="22",
            lottery=data["lottery"],
            family="pair_network_feature_vs_random",
            name="pair_network_v1_vs_random",
            description="Stage 22 pair-network feature challenger vs random.",
            metric="average_matches_per_ticket",
            comparison=data["challenger_vs_random"],
            sample_size=data.get("challenger", {}).get("metrics", {}).get("draws_evaluated"),
            original_classification=data.get("governance", {}).get("unified_conclusion"),
            source_file="backend/app/research/stage22_pair_network.py",
            source_output=_rel(path),
            dataset_hash=data.get("dataset_hash"),
            dataset_cutoff_draw=_last_draw(data),
            preregistered=True,
            primary_or_secondary="secondary",
        ),
    )


def _stage23_records() -> tuple[GlobalHypothesisRecord, ...]:
    return (
        make_hypothesis_record(
            stage="23",
            lottery="MINI_LOTO",
            experiment_family="portfolio_calibration_research",
            hypothesis_name="portfolio_calibration_report_only",
            description="Stage 23 was referenced as completed but no committed output was found.",
            metric="portfolio_calibration",
            provenance_status=REPORT_ONLY,
            historical_or_prospective="historical",
            eligible_for_global_correction=False,
            exclusion_reason="REPORT_ONLY_NO_VERIFIED_OUTPUT",
        ),
    )


def _stage24_records(export_root: Path, research_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "stage24" / "v2_stage24_temporal_research.json"
    if not path.exists():
        return ()
    data = _json(path)
    report = research_root / "LotoSystem_V2_Stage24_Temporal_Discovery.md"
    records = []
    for signal, payload in sorted(data.get("signals", {}).items()):
        records.append(
            _comparison_record(
                stage="24",
                lottery=data["lottery"],
                family="mini_loto_temporal_signal_vs_random",
                name=f"{signal}_vs_random",
                description=f"Stage 24 temporal signal {signal} vs paired random.",
                metric="matches_per_draw",
                comparison={
                    "difference": payload.get("difference"),
                    "difference_ci": payload.get("difference_ci"),
                    "raw_p_value": payload.get("raw_p_value"),
                    "adjusted_p_value": payload.get("adjusted_p_value"),
                    "bh_p_value": payload.get("bh_exploratory_p_value"),
                },
                sample_size=len(payload.get("observations", [])),
                original_classification=payload.get("classification"),
                source_file="backend/app/research/stage24_temporal_research.py",
                source_report=_rel(report) if report.exists() else None,
                source_output=_rel(path),
                dataset_hash=data.get("discovery_dataset_hash"),
                dataset_cutoff_draw=data.get("discovery_cutoff_draw"),
                preregistered=True,
                primary_or_secondary="primary"
                if signal == data.get("frozen_decision", {}).get("strongest_signal")
                else "secondary",
                notes=(
                    "Stage 24 code/output contain nine signal tests. The research report headline "
                    "matches the frozen decision; stale-ingestion discussion is explicitly "
                    "obsolete."
                ),
            )
        )
    return tuple(records)


def _stage25_records(export_root: Path, research_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "stage25" / "v2_stage25_ranking_discrimination.json"
    if not path.exists():
        return ()
    data = _json(path)
    report = research_root / "LotoSystem_V2_Stage25_Ranking_Discrimination.md"
    records = []
    for config, payload in sorted(data.get("regularization_results", {}).items()):
        for endpoint, comparison in sorted(payload.get("primary_comparisons", {}).items()):
            record = _comparison_record(
                stage="25",
                lottery=data["lottery"],
                family="mini_loto_ranking_discrimination_regularization",
                name=f"{config}_{endpoint}",
                description=f"Stage 25 regularization/calibration config {config} on {endpoint}.",
                metric=endpoint,
                comparison={
                    "difference": comparison.get("difference"),
                    "difference_ci": comparison.get("difference_ci"),
                    "raw_p_value": comparison.get("raw_p_value"),
                    "adjusted_p_value": comparison.get("holm_p_value"),
                    "bh_p_value": comparison.get("bh_p_value"),
                },
                sample_size=data.get("winner_rank_discrimination", {}).get("sample_size"),
                original_classification=comparison.get("classification"),
                source_file="backend/app/research/stage25_ranking_discrimination.py",
                source_report=_rel(report) if report.exists() else None,
                source_output=_rel(path),
                dataset_hash=data.get("discovery_dataset_hash"),
                dataset_cutoff_draw=data.get("discovery_cutoff_draw"),
                preregistered=True,
                primary_or_secondary="primary",
            )
            if comparison.get("classification") == "BASELINE":
                record = replace(
                    record,
                    eligible_for_global_correction=False,
                    exclusion_reason="BASELINE_REFERENCE_NOT_NEW_HYPOTHESIS",
                )
            records.append(record)
    return tuple(records)


def _stage26_records(export_root: Path, research_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "stage26" / "v2_stage26_feature_information.json"
    if not path.exists():
        return ()
    data = _json(path)
    report = research_root / "LotoSystem_V2_Stage26_Feature_Information_Audit.md"
    records = []
    for feature, payload in sorted(data.get("features", {}).items()):
        endpoints = payload.get("direct_ranker", {}).get("primary_endpoints", {})
        for endpoint, comparison in sorted(endpoints.items()):
            records.append(
                _comparison_record(
                    stage="26",
                    lottery=data["lottery"],
                    family="mini_loto_single_feature_information",
                    name=f"{feature}_{endpoint}",
                    description=f"Stage 26 direct feature ranker {feature} on {endpoint}.",
                    metric=endpoint,
                    comparison={
                        "difference": comparison.get("difference"),
                        "difference_ci": comparison.get("confidence_interval"),
                        "raw_p_value": comparison.get("raw_p_value"),
                        "adjusted_p_value": comparison.get("holm_p_value"),
                        "bh_p_value": comparison.get("bh_p_value"),
                    },
                    sample_size=payload.get("direct_ranker", {}).get("sample_size"),
                    original_classification=comparison.get("classification"),
                    source_file="backend/app/research/stage26_feature_information.py",
                    source_report=_rel(report) if report.exists() else None,
                    source_output=_rel(path),
                    dataset_hash=data.get("discovery_dataset_hash"),
                    dataset_cutoff_draw=data.get("discovery_cutoff_draw"),
                    preregistered=True,
                    primary_or_secondary="primary",
                )
            )
    return tuple(records)


def _stage27_records(root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = root / "MINI_LOTO" / "summary.json"
    if not path.exists():
        return ()
    data = _json(path)
    records = []
    for signal, payload in sorted(data.get("signals", {}).items()):
        records.append(
            make_hypothesis_record(
                stage="27",
                lottery=data.get("lottery", "MINI_LOTO"),
                experiment_family="prospective_signal_tracking",
                hypothesis_name=f"{signal}_prospective_tracking",
                description=f"Stage 27 prospective tracking for {signal}.",
                metric="mean_winner_rank_advantage",
                sample_size=payload.get("evaluated_draw_count"),
                effect=payload.get("mean_rank_advantage"),
                raw_p_value=payload.get("paired_permutation_p_value"),
                original_classification=payload.get("classification"),
                source_file="backend/app/research/stage27_prospective_signals.py",
                source_report="research/LotoSystem_V2_Stage27_Prospective_Signal_Tracking.md",
                source_output=_rel(path),
                provenance_status=PROSPECTIVE_ACTIVE,
                historical_or_prospective="prospective",
                discovery_or_confirmation="confirmation",
                dataset_cutoff_draw=data.get("prospective_start_draw"),
                preregistered=True,
                eligible_for_global_correction=False,
                exclusion_reason="PROSPECTIVE_NOT_YET_EVALUABLE",
                notes=(
                    "Prospective record is active; evaluated draw count is below evidence gate. "
                    "Repeated future looks can create optional-stopping exposure after thresholds."
                ),
            )
        )
    return tuple(records)


def _stage28_records(export_root: Path, research_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "stage28" / "mini_loto_ticket_popularity_report.json"
    if not path.exists():
        return ()
    data = _json(path)
    primary = data.get("primary_association", {})
    report = research_root / "LotoSystem_V2_Stage28_Ticket_Popularity.md"
    return (
        make_hypothesis_record(
            stage="28",
            lottery=data.get("lottery", "MINI_LOTO"),
            experiment_family="ticket_popularity_prize_split",
            hypothesis_name="popularity_score_vs_sales_normalized_first_winner_rate",
            description="Stage 28 preregistered popularity proxy association test.",
            metric=primary.get("endpoint", "sales_normalized_first_winner_rate"),
            sample_size=primary.get("usable_observations"),
            effect=primary.get("effect"),
            raw_p_value=primary.get("raw_p_value"),
            original_holm_p_value=primary.get("holm_p_value"),
            original_bh_p_value=primary.get("bh_p_value"),
            original_classification="UNTESTABLE",
            source_file="backend/app/research/stage28_ticket_popularity.py",
            source_report=_rel(report) if report.exists() else None,
            source_output=_rel(path),
            provenance_status=UNTESTABLE_CURRENT_DATA,
            historical_or_prospective="historical",
            discovery_or_confirmation="discovery",
            dataset_hash=data.get("dataset_hash"),
            preregistered=True,
            primary_or_secondary="primary",
            eligible_for_global_correction=False,
            exclusion_reason="NO_USABLE_OBSERVATIONS",
            notes="Stage 28 p=1 values are safe placeholders for an unrun N=0 test.",
        ),
    )


def _stage28b_records(export_root: Path, research_root: Path) -> tuple[GlobalHypothesisRecord, ...]:
    path = export_root / "stage28" / "mini_loto_economic_history.csv"
    if not path.exists():
        return ()
    report = research_root / "LotoSystem_V2_Stage28_Ticket_Popularity.md"
    return (
        make_hypothesis_record(
            stage="28B",
            lottery="MINI_LOTO",
            experiment_family="ticket_popularity_economic_data_acquisition",
            hypothesis_name="economic_history_acquisition",
            description="Stage 28B acquired available winner-count/payout data but no sales observations.",  # noqa: E501
            metric="data_availability",
            source_file="backend/app/research/stage28_economic_history.py",
            source_report=_rel(report) if report.exists() else None,
            source_output=_rel(path),
            provenance_status=VERIFIED_CODE_AND_OUTPUT,
            eligible_for_global_correction=False,
            exclusion_reason="DATA_ACQUISITION_NO_HYPOTHESIS_TEST",
        ),
    )


def _comparison_record(
    *,
    stage: str,
    lottery: str,
    family: str,
    name: str,
    description: str,
    metric: str,
    comparison: dict[str, Any],
    sample_size: int | None,
    original_classification: str | None,
    source_file: str,
    source_output: str,
    source_report: str | None = None,
    dataset_hash: str | None = None,
    dataset_cutoff_draw: int | None = None,
    preregistered: bool = False,
    primary_or_secondary: str = "secondary",
    notes: str = "",
) -> GlobalHypothesisRecord:
    ci = comparison.get("difference_ci") or comparison.get("confidence_interval") or {}
    return make_hypothesis_record(
        stage=stage,
        lottery=lottery,
        experiment_family=family,
        hypothesis_name=name,
        description=description,
        metric=metric,
        sample_size=sample_size,
        effect=_float_or_none(comparison.get("difference")),
        confidence_interval_low=_float_or_none(ci.get("lower")),
        confidence_interval_high=_float_or_none(ci.get("upper")),
        raw_p_value=_float_or_none(comparison.get("raw_p_value")),
        original_holm_p_value=_float_or_none(
            comparison.get("adjusted_p_value", comparison.get("holm_p_value"))
        ),
        original_bh_p_value=_float_or_none(comparison.get("bh_p_value")),
        original_classification=original_classification,
        source_file=source_file,
        source_report=source_report,
        source_output=source_output,
        provenance_status=VERIFIED_CODE_AND_OUTPUT,
        historical_or_prospective="historical",
        discovery_or_confirmation="discovery",
        dataset_cutoff_draw=dataset_cutoff_draw,
        dataset_hash=dataset_hash,
        preregistered=preregistered,
        primary_or_secondary=primary_or_secondary,
        notes=notes,
    )


def _mark_duplicates(
    records: tuple[GlobalHypothesisRecord, ...],
) -> tuple[GlobalHypothesisRecord, ...]:
    by_exact: dict[tuple[Any, ...], str] = {}
    verified_raw_tests: set[tuple[Any, ...]] = set()
    output: list[GlobalHypothesisRecord] = []
    for record in records:
        status = DISTINCT_HYPOTHESIS
        exclusion = record.exclusion_reason
        eligible = record.eligible_for_global_correction
        exact = (
            record.stage,
            record.lottery,
            record.experiment_family,
            record.hypothesis_name,
            record.metric,
            record.raw_p_value,
            record.effect,
        )
        raw_test = (
            record.stage,
            record.lottery,
            record.metric,
            record.raw_p_value,
            record.effect,
            record.sample_size,
            record.dataset_hash,
        )
        if exact in by_exact:
            status = EXACT_DUPLICATE
        elif (
            record.source_output == "data/exports/experiments/v2_experiment_ledger.json"
            and record.stage != "18"
            and record.raw_p_value is not None
        ):
            status = SAME_HYPOTHESIS_DIFFERENT_REPORT
        else:
            by_exact[exact] = record.hypothesis_id
        if (
            status == DISTINCT_HYPOTHESIS
            and record.provenance_status == VERIFIED_CODE_AND_OUTPUT
            and record.raw_p_value is not None
        ):
            verified_raw_tests.add(raw_test)
        if status != DISTINCT_HYPOTHESIS:
            eligible = False
            exclusion = status
        output.append(
            replace(
                record,
                duplicate_status=status,
                eligible_for_global_correction=eligible,
                exclusion_reason=exclusion,
            )
        )
    return tuple(output)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _last_draw(data: dict[str, Any]) -> int | None:
    dataset_range = data.get("dataset_range") or data.get("discovery_range") or {}
    return dataset_range.get("last_draw_number")


def _record_sort_key(record: GlobalHypothesisRecord) -> tuple[Any, ...]:
    return (
        _stage_sort_key(record.stage),
        record.lottery,
        record.experiment_family,
        record.hypothesis_name,
        record.metric,
        record.hypothesis_id,
    )


def _stage_sort_key(stage: str) -> tuple[int, str]:
    digits = "".join(character for character in str(stage) if character.isdigit())
    return (int(digits) if digits else 999, str(stage))


def _default_exclusion_reason(provenance_status: str, raw_p_value: float | None) -> str | None:
    if raw_p_value is None:
        return "NO_NUMERIC_RAW_P_VALUE"
    if provenance_status == REPORT_ONLY:
        return "REPORT_ONLY_NO_VERIFIED_OUTPUT"
    if provenance_status == UNTESTABLE_CURRENT_DATA:
        return "NO_USABLE_OBSERVATIONS"
    if provenance_status == PROSPECTIVE_ACTIVE:
        return "PROSPECTIVE_NOT_YET_EVALUABLE"
    return None


def _normalize_classification(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.upper()
    if normalized in {"NO_EVIDENCE", "NO_FEATURE_IMPROVEMENT", "NO_PORTFOLIO_IMPROVEMENT"}:
        return "NO_EVIDENCE"
    if normalized in {"WEAK_FEATURE_SIGNAL", "WEAK_SIGNAL"}:
        return "WEAK_SIGNAL"
    return normalized


def _strongest(records: tuple[GlobalHypothesisRecord, ...], field: str) -> str | None:
    candidates = tuple(record for record in records if getattr(record, field) is not None)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (
            float(getattr(record, field)),
            float(record.raw_p_value) if record.raw_p_value is not None else 1.0,
            record.hypothesis_id,
        ),
    ).hypothesis_id
