from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import research_result_json
from backend.app.research.stage27_prospective_signals import (
    STAGE27_SIGNALS,
    STATUS_EVALUATED,
    load_stage27_record,
    run_stage27_cycle,
    stage27_record_path,
)
from backend.app.research.stage29_global_experiment_ledger import VERIFIED_OUTPUT_ONLY
from backend.app.research.stage31_operational_integrity_monitor import (
    EVIDENCE_GATE_CLASSIFICATION,
    RECOMMENDED_DISPOSABLE_IGNORES,
    audit_repository_hygiene,
    audit_stage27_health,
    build_stage31_summary,
    preview_stage27_global_registration,
)

PROTECTED_LIVE_FILES = (
    Path("data/predictions/MINI_LOTO/1404.json"),
    Path("data/predictions/MINI_LOTO/ledger.json"),
    Path("data/prospective/stage27/MINI_LOTO/1404.json"),
    Path("data/prospective/stage27/MINI_LOTO/summary.json"),
    Path("data/processed/mini_loto_history.csv"),
    Path("data/settlements/ledger.json"),
)


def _draws(
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    start_number: int = 1368,
    count: int = 40,
    start_date: date = date(2026, 1, 6),
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for index in range(count):
        base = ((index * 11) % lottery.number_max) + 1
        main = tuple(
            sorted(
                ((base + offset * 6 - 1) % lottery.number_max) + 1
                for offset in range(lottery.numbers_per_ticket)
            )
        )
        bonus = next(
            number
            for number in range(lottery.number_min, lottery.number_max + 1)
            if number not in main
        )
        rows.append(
            HistoricalDraw(
                lottery=lottery,
                draw_number=start_number + index,
                draw_date=start_date + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(rows)


def _timestamp(offset_days: int = 0) -> datetime:
    return datetime(2026, 8, 26, 0, 0, tzinfo=UTC) + timedelta(days=offset_days)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Live-data safety (section G): hash before/after, tmp_path isolation
# ---------------------------------------------------------------------------


def test_audit_against_live_repository_does_not_mutate_anything() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    existing = {
        path: repo_root / path for path in PROTECTED_LIVE_FILES if (repo_root / path).exists()
    }
    before = {path: _file_hash(full) for path, full in existing.items()}

    from backend.app.research.data import load_draws_csv

    draws = load_draws_csv(repo_root / "data/processed/mini_loto_history.csv", MINI_LOTO)
    audit_stage27_health(draws, MINI_LOTO, root=repo_root / "data/prospective/stage27")
    build_stage31_summary(draws, MINI_LOTO, root=repo_root / "data/prospective/stage27")
    audit_repository_hygiene(repo_root)

    after = {path: _file_hash(full) for path, full in existing.items()}
    assert after == before


def test_isolated_fixtures_use_tmp_path_not_live_root(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    health = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")
    assert health.latest_canonical_draw == draws[-1].draw_number
    # Nothing was written outside tmp_path.
    assert not (Path("data") / "prospective" / "stage27" / "MINI_LOTO" / "9999999.json").exists()


# ---------------------------------------------------------------------------
# B/H: freeze-before-result integrity, future-data leakage prevention
# ---------------------------------------------------------------------------


def test_freeze_before_result_ok_for_normal_cycle(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())

    health = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    assert health.integrity_failure_targets == ()
    for target in health.targets:
        assert target.freeze_before_result_ok is True
        assert target.retrospective_leak_suspected is False
        assert target.created_before_draw_date_ok is True


def test_detects_retrospective_leak_when_cutoff_reaches_target_draw(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    path = stage27_record_path(tmp_path / "stage27", MINI_LOTO, draws[-1].draw_number + 1)
    record = load_stage27_record(path)

    # Tamper: claim the freeze saw history through (and including) its own
    # target draw number -- a retrospective-generation leak.
    leaked = replace(record, history_cutoff_draw=record.draw_number)
    path.write_text(research_result_json(leaked), encoding="utf-8")

    health = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    leaked_audit = next(t for t in health.targets if t.draw_number == record.draw_number)
    assert leaked_audit.retrospective_leak_suspected is True
    assert leaked_audit.freeze_before_result_ok is False
    assert record.draw_number in health.integrity_failure_targets


def test_detects_frozen_after_target_draw_date_from_created_at(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    path = stage27_record_path(tmp_path / "stage27", MINI_LOTO, draws[-1].draw_number + 1)
    record = load_stage27_record(path)

    # Tamper: created_at now claims the record was written after its own
    # target draw date -- suspicious independent of history_cutoff_draw.
    after_target = replace(
        record,
        created_at=datetime.fromisoformat(record.draw_date).replace(tzinfo=UTC).isoformat(),
    )
    path.write_text(research_result_json(after_target), encoding="utf-8")

    health = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    flagged = next(t for t in health.targets if t.draw_number == record.draw_number)
    assert flagged.created_before_draw_date_ok is False
    assert record.draw_number in health.integrity_failure_targets


# ---------------------------------------------------------------------------
# H: duplicate-run idempotency
# ---------------------------------------------------------------------------


def test_running_cycle_and_audit_twice_is_idempotent(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    first = build_stage31_summary(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp(1))

    # Re-run the exact same cycle again with the same inputs.
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    second = build_stage31_summary(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp(1))

    assert first.prospective_draws_frozen == second.prospective_draws_frozen
    assert first.prospective_draws_evaluated == second.prospective_draws_evaluated
    assert first.health.targets == second.health.targets
    assert first.integrity_failures == second.integrity_failures == 0


def test_running_audit_itself_twice_does_not_change_result(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())

    first = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")
    second = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    assert first == second


# ---------------------------------------------------------------------------
# H: missing-draw detection, duplicate-target detection
# ---------------------------------------------------------------------------


def test_missing_draw_detection(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    more_draws = _draws(count=36)
    run_stage27_cycle(more_draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp(7))

    # Simulate a gap: delete one (now-evaluated) frozen record file without
    # updating metadata, as if a cycle silently failed to persist it. A
    # second, later target still exists, so the expected-range check below
    # has something to detect the gap against.
    gap_number = draws[-1].draw_number + 1
    stage27_record_path(tmp_path / "stage27", MINI_LOTO, gap_number).unlink()

    health = audit_stage27_health(more_draws, MINI_LOTO, root=tmp_path / "stage27")

    assert gap_number in health.missing_targets
    # Metadata was never told about this gap, so the cross-check must fail.
    assert health.missing_vs_metadata_consistent is False


def test_missed_draw_recorded_via_record_missed_draws_is_consistent(tmp_path: Path) -> None:
    from backend.app.research.stage27_prospective_signals import record_missed_draws

    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    more_draws = _draws(count=36)
    run_stage27_cycle(more_draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp(7))

    gap_number = draws[-1].draw_number + 1
    stage27_record_path(tmp_path / "stage27", MINI_LOTO, gap_number).unlink()

    # The production path for acknowledging a missed draw: record it in
    # metadata via Stage 27's own function (never hand-edited).
    record_missed_draws(more_draws, MINI_LOTO, root=tmp_path / "stage27")

    health = audit_stage27_health(more_draws, MINI_LOTO, root=tmp_path / "stage27")

    assert gap_number in health.missing_targets
    assert gap_number in health.metadata_missed_draws
    assert health.missing_vs_metadata_consistent is True


# ---------------------------------------------------------------------------
# H: immutable frozen ranking / immutable evaluated record / hash validation
# ---------------------------------------------------------------------------


def test_tampered_frozen_ranking_fails_freeze_hash_check(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    path = stage27_record_path(tmp_path / "stage27", MINI_LOTO, draws[-1].draw_number + 1)
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Tamper with a frozen signal's ranking without touching freeze_hash.
    signal = next(iter(payload["signals"].values()))
    signal["ranking"] = list(reversed(signal["ranking"]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    health = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    tampered = next(t for t in health.targets if t.draw_number == draws[-1].draw_number + 1)
    assert tampered.freeze_hash_valid is False
    assert tampered.draw_number in health.integrity_failure_targets


def test_tampered_evaluated_record_fails_evaluation_hash_check(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())
    more_draws = _draws(count=36)
    run_stage27_cycle(more_draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp(7))

    evaluated_number = more_draws[-1].draw_number
    path = stage27_record_path(tmp_path / "stage27", MINI_LOTO, evaluated_number)
    record = load_stage27_record(path)
    assert record.status == STATUS_EVALUATED

    # The audit recomputes the evaluation payload itself from (record,
    # actual draw, lottery) rather than trusting the stored evaluation
    # dict's contents, so tamper-evidence lives in the hash field: corrupt
    # it directly to simulate the stored record being inconsistent with
    # what re-evaluating it would produce.
    tampered = replace(record, evaluation_hash="0" * 64)
    path.write_text(research_result_json(tampered), encoding="utf-8")

    health = audit_stage27_health(more_draws, MINI_LOTO, root=tmp_path / "stage27")
    tampered_audit = next(t for t in health.targets if t.draw_number == evaluated_number)
    assert tampered_audit.evaluation_hash_valid is False
    assert evaluated_number in health.integrity_failure_targets


def test_dataset_hash_present_and_stable_across_audits(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())

    first = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")
    second = audit_stage27_health(draws, MINI_LOTO, root=tmp_path / "stage27")

    for target in first.targets:
        assert target.dataset_hash
    assert {t.draw_number: t.dataset_hash for t in first.targets} == {
        t.draw_number: t.dataset_hash for t in second.targets
    }


# ---------------------------------------------------------------------------
# H: Stage27 evidence-gate behavior / Stage29 integration gate
# ---------------------------------------------------------------------------


def test_evidence_gate_not_reached_with_small_sample(tmp_path: Path) -> None:
    draws = _draws(count=35)
    run_stage27_cycle(draws, MINI_LOTO, root=tmp_path / "stage27", now=_timestamp())

    summary = build_stage31_summary(draws, MINI_LOTO, root=tmp_path / "stage27")

    assert summary.any_signal_reaches_evidence_gate is False
    assert summary.predictive_edge_currently_supported is False
    for signal in summary.signals:
        assert signal.evidence_gate_reached is False
        assert signal.declares_predictive_evidence is False
    assert summary.stage29_registration_preview["gate_reached"] is False
    assert summary.stage29_registration_preview["draft_hypothesis_records"] == ()


def test_stage29_registration_preview_never_marks_gate_reached_below_threshold() -> None:
    fake_summary = {
        "lottery": "MINI_LOTO",
        "prospective_start_draw": 1403,
        "signals": {
            signal_id: {
                "classification": "PRELIMINARY_POSITIVE",
                "evaluated_draw_count": 20,
                "mean_rank_advantage": 2.0,
                "paired_permutation_p_value": 0.2,
            }
            for signal_id in STAGE27_SIGNALS
        },
    }
    preview = preview_stage27_global_registration(fake_summary)
    assert preview["gate_reached"] is False
    assert preview["signals_ready_for_registration"] == ()
    assert preview["draft_hypothesis_records"] == ()
    assert "none" in preview["action_required"]


def test_stage29_registration_preview_builds_deterministic_draft_once_gate_reached() -> None:
    fake_summary = {
        "lottery": "MINI_LOTO",
        "prospective_start_draw": 1403,
        "signals": {
            "production_pair_lr": {
                "classification": EVIDENCE_GATE_CLASSIFICATION,
                "evaluated_draw_count": 55,
                "mean_rank_advantage": 3.4,
                "paired_permutation_p_value": 0.01,
            },
            "pair_strength_direct": {
                "classification": "PRELIMINARY_POSITIVE",
                "evaluated_draw_count": 30,
                "mean_rank_advantage": 1.1,
                "paired_permutation_p_value": 0.3,
            },
            "frequency_20": {
                "classification": "INSUFFICIENT_DATA",
                "evaluated_draw_count": 3,
                "mean_rank_advantage": None,
                "paired_permutation_p_value": None,
            },
            "paired_random": {
                "classification": EVIDENCE_GATE_CLASSIFICATION,
                "evaluated_draw_count": 55,
            },
        },
    }

    preview_a = preview_stage27_global_registration(fake_summary)
    preview_b = preview_stage27_global_registration(fake_summary)

    assert preview_a == preview_b  # deterministic, reproducible
    assert preview_a["gate_reached"] is True
    assert preview_a["signals_ready_for_registration"] == ("production_pair_lr",)
    assert len(preview_a["draft_hypothesis_records"]) == 1
    draft = preview_a["draft_hypothesis_records"][0]
    assert draft["provenance_status"] == VERIFIED_OUTPUT_ONLY
    assert draft["preregistered"] is True
    assert "ONE-TIME" in draft["notes"]
    # paired_random is a control, never itself registered as a hypothesis.
    assert all(
        "paired_random" not in record["hypothesis_name"]
        for record in preview_a["draft_hypothesis_records"]
    )


def test_stage27_is_mini_loto_only_for_health_audit(tmp_path: Path) -> None:
    draws = _draws(LOTO6, start_number=2100, count=10)
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        audit_stage27_health(draws, LOTO6, root=tmp_path / "stage27")


# ---------------------------------------------------------------------------
# F: repository hygiene audit (read-only)
# ---------------------------------------------------------------------------


def test_repository_hygiene_audit_is_read_only_and_finds_no_gitignore(tmp_path: Path) -> None:
    audit = audit_repository_hygiene(tmp_path)
    assert audit.gitignore_exists is False
    assert set(RECOMMENDED_DISPOSABLE_IGNORES) == set(audit.missing_recommended_ignores)
    assert not (tmp_path / ".gitignore").exists()  # never created by the audit


def test_repository_hygiene_audit_flags_disposable_pattern_gap(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n", encoding="utf-8"
    )
    audit = audit_repository_hygiene(tmp_path)
    assert audit.missing_recommended_ignores == (".pytest-tmp/",)
    assert "Add to .gitignore: .pytest-tmp/" in audit.recommendations


def test_repository_hygiene_audit_never_writes_to_disk(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())
    audit_repository_hygiene(tmp_path)
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
