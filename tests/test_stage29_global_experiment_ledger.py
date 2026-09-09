from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.research.persistence import research_result_json
from backend.app.research.stage29_global_experiment_ledger import (
    DISTINCT_HYPOTHESIS,
    EXACT_DUPLICATE,
    GLOBAL_FAMILY_DEFINITION,
    PROSPECTIVE_ACTIVE,
    REPORT_ONLY,
    UNTESTABLE_CURRENT_DATA,
    build_repository_global_ledger,
    classify_global_result,
    compute_global_corrections,
    deterministic_hypothesis_id,
    load_global_ledger,
    make_hypothesis_record,
    record_result,
    register_hypothesis,
    save_global_ledger,
)


def _record(name: str, p: float, *, effect: float = 0.1, stage: str = "99"):
    return make_hypothesis_record(
        stage=stage,
        lottery="MINI_LOTO",
        experiment_family="test_family",
        hypothesis_name=name,
        description=name,
        metric="mean_matches",
        sample_size=100,
        effect=effect,
        raw_p_value=p,
        original_holm_p_value=0.001,
        original_bh_p_value=0.001,
        original_classification="WEAK_SIGNAL",
        source_file="test.py",
        source_output="test.json",
        provenance_status="VERIFIED_CODE_AND_OUTPUT",
        preregistered=True,
        primary_or_secondary="primary",
    )


def test_deterministic_hypothesis_ids_and_ordering() -> None:
    left = deterministic_hypothesis_id(
        stage="24",
        lottery="MINI_LOTO",
        experiment_family="temporal",
        hypothesis_name="signal",
        metric="mean",
        source_output="out.json",
    )
    right = deterministic_hypothesis_id(
        stage="24",
        lottery="MINI_LOTO",
        experiment_family="temporal",
        hypothesis_name="signal",
        metric="mean",
        source_output="out.json",
    )
    ledger = compute_global_corrections((_record("b", 0.2), _record("a", 0.1)))

    assert left == right
    assert tuple(record.hypothesis_name for record in ledger.records) == ("a", "b")


def test_holm_and_bh_use_raw_p_values_not_original_adjusted_values() -> None:
    ledger = compute_global_corrections(
        (
            _record("a", 0.01),
            _record("b", 0.04),
            _record("c", 0.20),
        )
    )
    by_name = {record.hypothesis_name: record for record in ledger.records}

    assert by_name["a"].global_holm_p_value == pytest.approx(0.03)
    assert by_name["b"].global_holm_p_value == pytest.approx(0.08)
    assert by_name["c"].global_holm_p_value == pytest.approx(0.20)
    assert by_name["a"].global_bh_p_value == pytest.approx(0.03)
    assert by_name["b"].global_bh_p_value == pytest.approx(0.06)


def test_exact_duplicate_excluded_from_authoritative_family() -> None:
    duplicate = _record("same", 0.01)
    ledger = compute_global_corrections((duplicate, duplicate))

    assert ledger.summary.total_audit_records == 2
    assert ledger.summary.authoritative_hypotheses_included == 1
    assert ledger.summary.duplicates_excluded == 1
    assert ledger.records[1].duplicate_status == EXACT_DUPLICATE
    assert ledger.records[1].eligible_for_global_correction is False


def test_correlated_but_distinct_hypotheses_preserved() -> None:
    ledger = compute_global_corrections((_record("feature_a", 0.2), _record("feature_b", 0.2)))

    assert all(record.duplicate_status == DISTINCT_HYPOTHESIS for record in ledger.records)
    assert ledger.summary.authoritative_hypotheses_included == 2


def test_untestable_stage28_placeholder_p_value_excluded() -> None:
    record = make_hypothesis_record(
        stage="28",
        lottery="MINI_LOTO",
        experiment_family="ticket_popularity",
        hypothesis_name="sales_normalized_test",
        description="No sales observations.",
        metric="winner_rate",
        raw_p_value=1.0,
        provenance_status=UNTESTABLE_CURRENT_DATA,
        eligible_for_global_correction=False,
        exclusion_reason="NO_USABLE_OBSERVATIONS",
    )
    ledger = compute_global_corrections((record,))

    assert ledger.summary.untestable_excluded == 1
    assert ledger.summary.authoritative_hypotheses_included == 0


def test_prospective_insufficient_data_excluded() -> None:
    record = make_hypothesis_record(
        stage="27",
        lottery="MINI_LOTO",
        experiment_family="prospective",
        hypothesis_name="frequency_20",
        description="Pending prospective signal.",
        metric="mean_rank",
        raw_p_value=None,
        provenance_status=PROSPECTIVE_ACTIVE,
        historical_or_prospective="prospective",
        eligible_for_global_correction=False,
        exclusion_reason="PROSPECTIVE_NOT_YET_EVALUABLE",
    )
    ledger = compute_global_corrections((record,))

    assert ledger.summary.prospective_not_yet_evaluable_excluded == 1
    assert ledger.records[0].global_holm_p_value is None


def test_report_only_nonverifiable_entry_excluded_and_provenance_preserved() -> None:
    record = make_hypothesis_record(
        stage="23",
        lottery="MINI_LOTO",
        experiment_family="report_only",
        hypothesis_name="missing_output",
        description="No output found.",
        metric="portfolio",
        raw_p_value=0.01,
        provenance_status=REPORT_ONLY,
        eligible_for_global_correction=False,
        exclusion_reason="REPORT_ONLY_NO_VERIFIED_OUTPUT",
    )
    ledger = compute_global_corrections((record,))

    assert ledger.summary.report_only_or_unverified_excluded == 1
    assert ledger.records[0].provenance_status == REPORT_ONLY


def test_classification_change_is_deterministic() -> None:
    ledger = compute_global_corrections((_record("tiny_raw", 0.04), _record("large", 0.5)))
    changed = tuple(record for record in ledger.records if record.classification_changed)

    assert len(changed) == 1
    assert changed[0].global_classification == "NO_EVIDENCE"
    assert (
        classify_global_result(
            effect=-0.1,
            global_holm_p_value=0.5,
            original_classification="NO_EVIDENCE",
        )
        == "NO_EVIDENCE"
    )


def test_byte_identical_export_without_timestamps(tmp_path: Path) -> None:
    ledger = compute_global_corrections((_record("a", 0.2), _record("b", 0.3)))

    first_paths = save_global_ledger(ledger, tmp_path / "first")
    second_paths = save_global_ledger(ledger, tmp_path / "second")

    assert Path(first_paths["ledger"]).read_text(encoding="utf-8") == Path(
        second_paths["ledger"]
    ).read_text(encoding="utf-8")
    assert load_global_ledger(first_paths["ledger"])["schema_version"]


def test_empty_ledger_and_p_value_edges_are_safe() -> None:
    empty = compute_global_corrections(())
    edge = compute_global_corrections((_record("zero", 0.0), _record("one", 1.0)))

    assert empty.summary.total_audit_records == 0
    assert edge.summary.global_holm_surviving_count == 1
    assert edge.summary.global_bh_surviving_count == 1


def test_future_preregistration_api_roundtrip() -> None:
    registered = register_hypothesis(
        stage="30",
        lottery="MINI_LOTO",
        experiment_family="future",
        hypothesis_name="future_signal",
        description="Preregistered placeholder.",
        metric="mean_rank",
        provenance_status=REPORT_ONLY,
        eligible_for_global_correction=False,
    )
    recorded = record_result(
        registered,
        raw_p_value=0.5,
        original_classification="NO_EVIDENCE",
        provenance_status="VERIFIED_OUTPUT_ONLY",
    )

    assert registered.raw_p_value is None
    assert recorded.raw_p_value == 0.5


def test_repository_ledger_handles_stage27_and_stage28_without_runtime_mutation() -> None:
    protected_paths = (
        Path("data/prospective/stage27/MINI_LOTO/1403.json"),
        Path("data/predictions/MINI_LOTO/1403.json"),
        Path("data/predictions/LOTO6/2136.json"),
        Path("data/processed/mini_loto_history.csv"),
        Path("data/settlements/ledger.json"),
    )
    before = {path: path.read_bytes() for path in protected_paths if path.exists()}

    ledger = build_repository_global_ledger()

    after = {path: path.read_bytes() for path in protected_paths if path.exists()}
    assert after == before
    assert "24" in ledger.stages_audited
    assert "27" in ledger.stages_audited
    assert "28" in ledger.stages_audited
    assert ledger.summary.prospective_not_yet_evaluable_excluded >= 1
    assert ledger.summary.untestable_excluded >= 1
    assert ledger.global_family_definition == GLOBAL_FAMILY_DEFINITION
    assert ledger.summary.global_holm_surviving_count == 0


def test_repository_export_payload_is_json_serializable(tmp_path: Path) -> None:
    ledger = build_repository_global_ledger()
    paths = save_global_ledger(ledger, tmp_path / "stage29")
    payload = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

    assert payload["total_audit_records"] == ledger.summary.total_audit_records
    assert research_result_json(ledger)
