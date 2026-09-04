from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.stage27_prospective_signals import (
    STAGE27_SIGNALS,
    STATUS_EVALUATED,
    STATUS_FROZEN,
    STATUS_MISSED,
    build_signal_rankings,
    build_stage27_record,
    evaluate_stage27_records,
    evaluated_stage27_record,
    freeze_next_stage27_record,
    freeze_stage27_record,
    initialize_stage27,
    load_stage27_record,
    rebuild_stage27_summary,
    record_missed_draws,
    run_stage27_cycle,
    stage27_freeze_hash,
    stage27_record_path,
)


def _draws(
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    start_number: int = 1368,
    count: int = 34,
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


def _timestamp() -> datetime:
    return datetime(2026, 8, 26, 0, 0, tzinfo=UTC)


def _freeze(tmp_path: Path, draws: tuple[HistoricalDraw, ...]):
    return freeze_next_stage27_record(
        draws,
        MINI_LOTO,
        root=tmp_path / "stage27",
        seed=123456,
        created_at=_timestamp(),
    )


def test_stage27_is_mini_loto_only(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        initialize_stage27(
            _draws(LOTO6, start_number=2100, count=34),
            LOTO6,
            root=tmp_path / "stage27",
        )


def test_prospective_boundary_is_latest_known_draw_plus_one(tmp_path: Path) -> None:
    draws = _draws()

    metadata = initialize_stage27(draws, MINI_LOTO, root=tmp_path / "stage27")

    assert metadata["latest_known_draw_at_initialization"] == 1401
    assert metadata["prospective_start_draw"] == 1402


def test_no_retroactive_freeze_after_result_exists(tmp_path: Path) -> None:
    draws = _draws(count=35)

    result = freeze_stage27_record(
        draws,
        MINI_LOTO,
        target_draw_number=1402,
        root=tmp_path / "stage27",
        created_at=_timestamp(),
    )

    assert result.status == STATUS_MISSED
    assert result.record is None
    assert result.target_result_absent is False


def test_freeze_uses_only_history_before_target_and_hashes_it(tmp_path: Path) -> None:
    draws = _draws()

    result = _freeze(tmp_path, draws)

    assert result.status == STATUS_FROZEN
    assert result.record is not None
    assert result.record.draw_number == 1402
    assert result.record.history_cutoff_draw == 1401
    assert result.record.history_dataset_hash == calculate_dataset_hash(draws)
    assert result.target_result_absent is True
    assert result.record.freeze_hash == stage27_freeze_hash(result.record)


def test_target_and_future_mutations_do_not_change_existing_frozen_record(
    tmp_path: Path,
) -> None:
    draws = _draws()
    frozen = _freeze(tmp_path, draws).record
    assert frozen is not None
    mutated_future = _draws(count=36)
    loaded_before = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    evaluation = evaluate_stage27_records(mutated_future, MINI_LOTO, root=tmp_path / "stage27")
    loaded_after = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    assert evaluation.evaluated_count == 1
    assert loaded_before.freeze_hash == loaded_after.freeze_hash == frozen.freeze_hash
    assert loaded_before.signals == loaded_after.signals


def test_signal_rankings_are_deterministic_and_complete_permutations() -> None:
    draws = _draws()

    first = build_signal_rankings(
        draws,
        MINI_LOTO,
        target_draw_number=1402,
        target_draw_date="2026-09-01",
        seed=123456,
    )
    second = build_signal_rankings(
        draws,
        MINI_LOTO,
        target_draw_number=1402,
        target_draw_date="2026-09-01",
        seed=123456,
    )

    assert set(first) == set(STAGE27_SIGNALS)
    assert first == second
    for signal in first.values():
        assert sorted(signal.ranking) == list(range(1, 32))
        assert len(signal.top5) == 5
        assert len(signal.top20) == 20


def test_random_ranking_seed_is_target_specific() -> None:
    draws = _draws()

    target_1402 = build_signal_rankings(
        draws,
        MINI_LOTO,
        target_draw_number=1402,
        target_draw_date="2026-09-01",
        seed=123456,
    )["paired_random"]
    target_1403 = build_signal_rankings(
        draws,
        MINI_LOTO,
        target_draw_number=1403,
        target_draw_date="2026-09-08",
        seed=123456,
    )["paired_random"]

    assert target_1402.ranking != target_1403.ranking
    assert sorted(target_1402.ranking) == list(range(1, 32))
    assert sorted(target_1403.ranking) == list(range(1, 32))


def test_freeze_hash_is_deterministic_and_evaluation_does_not_alter_it() -> None:
    history = _draws()
    actual = _draws(count=35)[-1]
    first = build_stage27_record(
        history,
        MINI_LOTO,
        target_draw_number=1402,
        target_draw_date="2026-09-01",
        prospective_start_draw=1402,
        seed=123456,
        created_at=_timestamp(),
    )
    second = build_stage27_record(
        history,
        MINI_LOTO,
        target_draw_number=1402,
        target_draw_date="2026-09-01",
        prospective_start_draw=1402,
        seed=123456,
        created_at=_timestamp(),
    )

    evaluated = evaluated_stage27_record(first, actual, MINI_LOTO)

    assert first.freeze_hash == second.freeze_hash
    assert evaluated.freeze_hash == first.freeze_hash
    assert evaluated.status == STATUS_EVALUATED
    assert evaluated.evaluation_hash is not None


def test_duplicate_freeze_is_idempotent_and_conflicting_file_is_rejected(tmp_path: Path) -> None:
    draws = _draws()
    first = _freeze(tmp_path, draws)
    second = _freeze(tmp_path, draws)
    path = Path(first.record_path or "")

    assert second.existing_record is True
    assert first.record is not None and second.record is not None
    assert first.record.freeze_hash == second.record.freeze_hash

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["freeze_hash"] = "corrupt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchValidationError, match="freeze hash"):
        _freeze(tmp_path, draws)


def test_duplicate_evaluation_is_idempotent_and_conflict_is_rejected(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]
    _freeze(tmp_path, draws)
    first = evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    second = evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    path = stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402)

    assert first.evaluated_count == 1
    assert second.evaluated_count == 0
    assert second.skipped_count == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_hash"] = "corrupt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchValidationError, match="evaluation hash"):
        evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")


def test_winner_rank_evaluation_and_bonus_exclusion(tmp_path: Path) -> None:
    draws = _draws()
    actual = HistoricalDraw(
        lottery=MINI_LOTO,
        draw_number=1402,
        draw_date=date(2026, 9, 1),
        main_numbers=(1, 4, 20, 25, 29),
        bonus_numbers=(22,),
    )

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    record = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    assert record.evaluation is not None
    production = record.evaluation["signal_results"]["production_pair_lr"]
    assert {int(number) for number in production["winner_ranks"]} == {1, 4, 20, 25, 29}
    assert "22" not in production["winner_ranks"]
    assert record.evaluation["bonus_excluded_from_main_capture"] is True


def test_paired_random_differences_are_recorded(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    record = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    comparisons = record.evaluation["paired_comparisons_vs_random"]  # type: ignore[index]
    assert set(comparisons) == {
        "production_pair_lr",
        "pair_strength_direct",
        "frequency_20",
    }
    assert "mean_winner_rank_advantage" in comparisons["frequency_20"]
    assert "top15_capture_advantage" in comparisons["production_pair_lr"]


def test_champion_direct_equality_diagnostics(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    record = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    equality = record.evaluation["production_vs_pair_strength_direct"]  # type: ignore[index]
    assert set(equality) == {
        "full_rank_equality",
        "spearman",
        "top5_equality",
        "top15_equality",
    }
    assert -1 <= equality["spearman"] <= 1


def test_summary_rebuild_is_deterministic_and_derived_from_records(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")
    first = rebuild_stage27_summary(MINI_LOTO, root=tmp_path / "stage27", bootstrap_replications=10)
    second = rebuild_stage27_summary(
        MINI_LOTO, root=tmp_path / "stage27", bootstrap_replications=10
    )

    assert first == second
    assert first["source"] == "derived from immutable Stage 27 draw records"
    assert first["evaluated_draw_count"] == 1
    assert first["pending_draw_count"] == 0
    assert first["classification"] == "INSUFFICIENT_DATA"


def test_missed_draw_cannot_be_backfilled_and_catch_up_freezes_next(
    tmp_path: Path,
) -> None:
    initial = _draws()
    root = tmp_path / "stage27"
    initialize_stage27(initial, MINI_LOTO, root=root, initialized_at=_timestamp())
    caught_up = _draws(count=35)

    missed = record_missed_draws(caught_up, MINI_LOTO, root=root)
    freeze = freeze_next_stage27_record(
        caught_up,
        MINI_LOTO,
        root=root,
        seed=123456,
        created_at=_timestamp(),
    )

    assert missed == (1402,)
    assert freeze.record is not None
    assert freeze.record.draw_number == 1403
    assert freeze.missed_draws == (1402,)


def test_run_cycle_freezes_pending_record_and_reports_insufficient_data(tmp_path: Path) -> None:
    result = run_stage27_cycle(
        _draws(),
        MINI_LOTO,
        root=tmp_path / "stage27",
        seed=123456,
        now=_timestamp(),
    )
    summary = rebuild_stage27_summary(MINI_LOTO, root=tmp_path / "stage27")

    assert result.prospective_start_draw == 1402
    assert result.freeze.record is not None
    assert result.freeze.record.status == STATUS_FROZEN
    assert result.evaluated.evaluated_count == 0
    assert summary["pending_draw_count"] == 1
    assert summary["classification"] == "INSUFFICIENT_DATA"


def test_evidence_gate_prevents_premature_promotion(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]
    root = tmp_path / "stage27"

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=root)
    summary = rebuild_stage27_summary(MINI_LOTO, root=root, bootstrap_replications=10)

    for signal_id in ("production_pair_lr", "pair_strength_direct", "frequency_20"):
        assert summary["signals"][signal_id]["classification"] == "INSUFFICIENT_DATA"


def test_stage27_functions_do_not_touch_production_or_settlement_roots(tmp_path: Path) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]
    production_root = Path("data/predictions")
    settlement_root = Path("data/settlements")
    prediction_snapshot = sorted(path.as_posix() for path in production_root.glob("**/*.json"))
    settlement_snapshot = sorted(path.as_posix() for path in settlement_root.glob("**/*.json"))

    _freeze(tmp_path, draws)
    evaluate_stage27_records((*draws, actual), MINI_LOTO, root=tmp_path / "stage27")

    assert (
        sorted(path.as_posix() for path in production_root.glob("**/*.json")) == prediction_snapshot
    )
    assert (
        sorted(path.as_posix() for path in settlement_root.glob("**/*.json")) == settlement_snapshot
    )


def test_target_mutation_before_freeze_changes_candidate_record_but_not_existing_freeze(
    tmp_path: Path,
) -> None:
    draws = _draws()
    actual = _draws(count=35)[-1]
    mutated_actual = replace(actual, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))

    frozen = _freeze(tmp_path, draws).record
    assert frozen is not None
    evaluate_stage27_records((*draws, mutated_actual), MINI_LOTO, root=tmp_path / "stage27")
    loaded = load_stage27_record(stage27_record_path(tmp_path / "stage27", MINI_LOTO, 1402))

    assert loaded.freeze_hash == frozen.freeze_hash
    assert {signal_id: signal.ranking for signal_id, signal in loaded.signals.items()} == {
        signal_id: signal.ranking for signal_id, signal in frozen.signals.items()
    }
