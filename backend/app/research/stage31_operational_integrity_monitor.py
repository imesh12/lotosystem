"""Stage 31 -- Operational Integrity + Prospective Evidence Monitor.

This module is NOT a new prediction experiment. It introduces no new
predictive features, ML models, number-selection strategies, or
retrospective hypothesis searches. It is a read-only auditor and
machine-readable summary builder that proves (or flags doubt about) the
operational integrity of Stage 27's prospective tracking and its
relationship to Stage 29's global hypothesis governance.

It never modifies:
  * the mathematical conclusions of Stage 27, Stage 28/28B, Stage 29, or
    Stage 30,
  * production ticket generation,
  * any live prediction, settlement, prospective, or history file.

Everything here either reads existing live/evaluated data and reports on
it, or (for the Stage 29 integration path) builds an in-memory *preview*
of what a future, one-time registration would look like -- it never
writes that preview anywhere.

If integrity cannot be proven for a record, this module marks it in its
report. It never silently repairs, rewrites, or deletes evidence.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.stage27_prospective_signals import (
    STAGE27_ROOT,
    STAGE27_SIGNALS,
    STATUS_EVALUATED,
    STATUS_FROZEN,
    Stage27Record,
    evaluated_stage27_record,
    load_stage27_record,
    rebuild_stage27_summary,
    stage27_freeze_hash,
)
from backend.app.research.stage29_global_experiment_ledger import (
    VERIFIED_OUTPUT_ONLY,
    make_hypothesis_record,
)

STAGE31_SCHEMA_VERSION = "v2-stage31-operational-integrity-monitor-v1"
DEFAULT_STAGE31_OUTPUT_DIR = Path("data") / "exports" / "stage31"

# Stage 27 (and Stage 29's consumption of Stage 27) already define the
# preregistered evidence gate as the classification produced by
# ``rebuild_stage27_summary`` once a signal has enough evaluated draws, a
# significant Holm/permutation p-value, and a positive effect. Stage 31
# reuses that exact, already-tested classification rather than
# duplicating or re-deriving the threshold, so there is no risk of the
# gate drifting between Stage 27/29 and this monitor.
EVIDENCE_GATE_CLASSIFICATION = "ELIGIBLE_FOR_REVIEW"

RECOMMENDED_DISPOSABLE_IGNORES = (
    ".pytest-tmp/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
)

EVIDENCE_PATH_HINTS = (
    "data/predictions/",
    "data/settlements/",
    "data/prospective/",
    "data/processed/mini_loto_history.csv",
    "data/processed/loto6_history.csv",
    "research/",
)


# ---------------------------------------------------------------------------
# B. Stage 27 health monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stage27TargetAudit:
    draw_number: int
    status: str
    history_cutoff_draw: int
    history_cutoff_date: str
    dataset_hash: str
    created_at: str
    freeze_hash_valid: bool
    evaluation_hash_valid: bool | None
    freeze_before_result_ok: bool
    retrospective_leak_suspected: bool
    created_before_draw_date_ok: bool


@dataclass(frozen=True, slots=True)
class Stage27HealthAudit:
    lottery: str
    latest_canonical_draw: int | None
    prospective_start_draw: int | None
    frozen_targets: tuple[int, ...]
    evaluated_targets: tuple[int, ...]
    pending_targets: tuple[int, ...]
    missing_targets: tuple[int, ...]
    duplicate_targets: tuple[int, ...]
    latest_frozen_target: int | None
    latest_evaluated_target: int | None
    targets: tuple[Stage27TargetAudit, ...]
    integrity_failure_targets: tuple[int, ...]
    metadata_missed_draws: tuple[int, ...]
    missing_vs_metadata_consistent: bool


def audit_stage27_health(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    root: str | Path = STAGE27_ROOT,
) -> Stage27HealthAudit:
    """Read-only audit of the live Stage 27 records. Never writes anything.

    Independently re-derives frozen/evaluated/pending/missing/duplicate
    counts straight from the per-draw JSON files and metadata.json (it does
    not simply trust ``summary.json``), and recomputes each record's
    freeze/evaluation hash plus a freeze-before-result check so tampering
    or accidental retrospective generation would be caught rather than
    assumed away.
    """
    if str(lottery.code) != str(MINI_LOTO.code):
        raise ResearchValidationError("Stage 31 Stage27 health monitor supports MINI_LOTO only")
    ordered = validate_lottery_dataset(draws, lottery)
    by_number = {draw.draw_number: draw for draw in ordered}
    latest_canonical = ordered[-1].draw_number if ordered else None

    directory = Path(root) / str(lottery.code)
    metadata = _load_json_if_exists(directory / "metadata.json")
    prospective_start = int(metadata["prospective_start_draw"]) if metadata is not None else None
    metadata_missed = tuple(
        int(value) for value in (metadata.get("missed_draws", ()) if metadata else ())
    )

    records, duplicates = _load_all_stage27_records(directory)

    frozen_targets = tuple(sorted(records))
    evaluated_targets = tuple(
        sorted(number for number, record in records.items() if record.status == STATUS_EVALUATED)
    )
    pending_targets = tuple(
        sorted(number for number, record in records.items() if record.status == STATUS_FROZEN)
    )

    missing_targets: tuple[int, ...] = ()
    if prospective_start is not None and frozen_targets:
        expected = range(prospective_start, max(frozen_targets) + 1)
        missing_targets = tuple(number for number in expected if number not in records)
    missing_vs_metadata_consistent = set(missing_targets) == set(metadata_missed)

    target_audits: list[Stage27TargetAudit] = []
    integrity_failures: list[int] = []
    for number in frozen_targets:
        record = records[number]
        freeze_ok = stage27_freeze_hash(record) == record.freeze_hash
        leak = record.history_cutoff_draw >= record.draw_number
        created_ok = _created_before_draw_date_ok(record)
        eval_ok: bool | None = None
        if record.status == STATUS_EVALUATED:
            actual = by_number.get(record.draw_number)
            if actual is None:
                eval_ok = False
            else:
                try:
                    evaluated_stage27_record(record, actual, lottery)
                    eval_ok = True
                except ResearchValidationError:
                    eval_ok = False
        failing = (not freeze_ok) or leak or (not created_ok) or (eval_ok is False)
        if failing:
            integrity_failures.append(number)
        target_audits.append(
            Stage27TargetAudit(
                draw_number=number,
                status=record.status,
                history_cutoff_draw=record.history_cutoff_draw,
                history_cutoff_date=record.history_cutoff_date,
                dataset_hash=record.history_dataset_hash,
                created_at=record.created_at,
                freeze_hash_valid=freeze_ok,
                evaluation_hash_valid=eval_ok,
                freeze_before_result_ok=not leak,
                retrospective_leak_suspected=leak,
                created_before_draw_date_ok=created_ok,
            )
        )

    return Stage27HealthAudit(
        lottery=str(lottery.code),
        latest_canonical_draw=latest_canonical,
        prospective_start_draw=prospective_start,
        frozen_targets=frozen_targets,
        evaluated_targets=evaluated_targets,
        pending_targets=pending_targets,
        missing_targets=missing_targets,
        duplicate_targets=tuple(sorted(set(duplicates))),
        latest_frozen_target=max(frozen_targets) if frozen_targets else None,
        latest_evaluated_target=max(evaluated_targets) if evaluated_targets else None,
        targets=tuple(target_audits),
        integrity_failure_targets=tuple(integrity_failures),
        metadata_missed_draws=metadata_missed,
        missing_vs_metadata_consistent=missing_vs_metadata_consistent,
    )


def _created_before_draw_date_ok(record: Stage27Record) -> bool:
    """Independent wall-clock check: was this frozen before its target draw
    date, judged by the record's own ``created_at`` timestamp rather than
    its self-reported ``history_cutoff_draw``? Catches a record that claims
    a safe cutoff draw but was, by its own creation timestamp, actually
    written on or after the day its target result became available.
    """
    try:
        created = datetime.fromisoformat(record.created_at).astimezone(UTC).date()
        target = date.fromisoformat(record.draw_date)
    except ValueError:
        return False
    return created < target


def _load_all_stage27_records(
    directory: Path,
) -> tuple[dict[int, Stage27Record], tuple[int, ...]]:
    records: dict[int, Stage27Record] = {}
    duplicates: list[int] = []
    for path in _list_stage27_record_paths(directory):
        record = load_stage27_record(path)
        filename_draw = int(path.stem)
        if filename_draw != record.draw_number:
            raise ResearchValidationError(f"Stage 27 record filename/content mismatch: {path}")
        if record.draw_number in records:
            duplicates.append(record.draw_number)
        records[record.draw_number] = record
    return records, tuple(duplicates)


def _list_stage27_record_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(
        path
        for path in sorted(directory.glob("*.json"))
        if path.name not in {"metadata.json", "summary.json"}
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# D. Evidence dashboard / summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalEvidenceSnapshot:
    signal_id: str
    frozen_draw_count: int
    evaluated_draw_count: int
    pending_draw_count: int
    top5_capture_average: float | None
    top10_capture_average: float | None
    top15_capture_average: float | None
    mean_winner_rank: float | None
    random_mean_winner_rank: float | None
    mean_rank_advantage: float | None
    classification: str | None
    evidence_gate_reached: bool
    declares_predictive_evidence: bool


@dataclass(frozen=True, slots=True)
class Stage31Summary:
    schema_version: str
    generated_at: str
    lottery: str
    health: Stage27HealthAudit
    prospective_draws_frozen: int
    prospective_draws_evaluated: int
    prospective_draws_pending: int
    prospective_draws_missed: int
    integrity_failures: int
    earliest_target: int | None
    latest_target: int | None
    latest_evaluated_target: int | None
    counts_consistent_with_stage27_summary: bool
    signals: tuple[SignalEvidenceSnapshot, ...]
    stage29_registration_preview: dict[str, Any]
    any_signal_reaches_evidence_gate: bool
    predictive_edge_currently_supported: bool


def build_stage31_summary(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    root: str | Path = STAGE27_ROOT,
    now: datetime | None = None,
) -> Stage31Summary:
    health = audit_stage27_health(draws, lottery, root=root)
    directory = Path(root) / str(lottery.code)
    records, _duplicates = _load_all_stage27_records(directory)

    # Stage 27's own public, already-tested statistics (permutation
    # p-values, bootstrap CIs, classification) are reused verbatim rather
    # than re-derived, so the evidence-gate decision can never drift from
    # Stage 27/29's own authoritative computation.
    fresh_summary = to_jsonable(rebuild_stage27_summary(lottery, root=root))

    counts_consistent = (
        fresh_summary.get("frozen_draw_count") == len(health.frozen_targets)
        and fresh_summary.get("evaluated_draw_count") == len(health.evaluated_targets)
        and fresh_summary.get("pending_draw_count") == len(health.pending_targets)
    )

    signals = tuple(
        _signal_snapshot(signal_id, records, fresh_summary.get("signals", {}))
        for signal_id in STAGE27_SIGNALS
    )
    gate_reached = any(signal.evidence_gate_reached for signal in signals)

    return Stage31Summary(
        schema_version=STAGE31_SCHEMA_VERSION,
        generated_at=(now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        lottery=str(lottery.code),
        health=health,
        prospective_draws_frozen=len(health.frozen_targets),
        prospective_draws_evaluated=len(health.evaluated_targets),
        prospective_draws_pending=len(health.pending_targets),
        prospective_draws_missed=len(health.missing_targets),
        integrity_failures=len(health.integrity_failure_targets),
        earliest_target=min(health.frozen_targets) if health.frozen_targets else None,
        latest_target=health.latest_frozen_target,
        latest_evaluated_target=health.latest_evaluated_target,
        counts_consistent_with_stage27_summary=counts_consistent,
        signals=signals,
        stage29_registration_preview=preview_stage27_global_registration(fresh_summary, root=root),
        any_signal_reaches_evidence_gate=gate_reached,
        # Stage 31 never itself declares predictive evidence. Even if a
        # single signal's gate is reached, that is a one-time registration
        # decision for a human maintainer via Stage 29 (see
        # stage29_registration_preview), never an automatic claim.
        predictive_edge_currently_supported=False,
    )


def _signal_snapshot(
    signal_id: str,
    records: dict[int, Stage27Record],
    fresh_signal_summaries: dict[str, Any],
) -> SignalEvidenceSnapshot:
    evaluated = [record for record in records.values() if record.status == STATUS_EVALUATED]
    frozen_count = sum(signal_id in record.signals for record in records.values())
    pending_count = sum(
        record.status == STATUS_FROZEN and signal_id in record.signals
        for record in records.values()
    )
    per_draw = [
        record.evaluation["signal_results"][signal_id]
        for record in evaluated
        if record.evaluation is not None
        and signal_id in record.evaluation.get("signal_results", {})
    ]
    top10 = _avg_key(per_draw, "top10_capture")

    fresh = fresh_signal_summaries.get(signal_id, {})
    classification = fresh.get("classification")
    return SignalEvidenceSnapshot(
        signal_id=signal_id,
        frozen_draw_count=frozen_count,
        evaluated_draw_count=len(per_draw),
        pending_draw_count=pending_count,
        top5_capture_average=fresh.get("top5_capture_average"),
        top10_capture_average=top10,
        top15_capture_average=fresh.get("top15_capture_average"),
        mean_winner_rank=fresh.get("mean_winner_rank"),
        random_mean_winner_rank=fresh.get("random_mean_winner_rank"),
        mean_rank_advantage=fresh.get("mean_rank_advantage"),
        classification=classification,
        evidence_gate_reached=classification == EVIDENCE_GATE_CLASSIFICATION,
        declares_predictive_evidence=False,
    )


def _avg_key(values: list[dict[str, Any]], key: str) -> float | None:
    if not values:
        return None
    return mean(float(value[key]) for value in values)


def save_stage31_summary(
    summary: Stage31Summary,
    output_dir: str | Path = DEFAULT_STAGE31_OUTPUT_DIR,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "stage31_summary.json"
    path.write_text(research_result_json(summary), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# E. Stage 29 integration: deterministic one-time registration preview
# ---------------------------------------------------------------------------


def preview_stage27_global_registration(
    summary: dict[str, Any],
    *,
    root: str | Path = STAGE27_ROOT,
) -> dict[str, Any]:
    """Pure preview. Never writes to any Stage 29 export, never mutates
    Stage 27 data, never changes Stage 29's current conclusion. Returns the
    exact, deterministic draft hypothesis record(s) that a human maintainer
    would register with Stage 29 -- EXACTLY ONCE -- for each signal whose
    preregistered evidence gate has been reached. While no signal's gate is
    reached (the current, live state), this always returns an empty,
    no-action result; it performs no optional stopping or cherry-picking of
    interim draws on its own.
    """
    ready_signals: list[str] = []
    draft_records: list[dict[str, Any]] = []
    for signal_id, payload in sorted(summary.get("signals", {}).items()):
        if signal_id == "paired_random":
            continue
        if payload.get("classification") != EVIDENCE_GATE_CLASSIFICATION:
            continue
        ready_signals.append(signal_id)
        draft_records.append(
            to_jsonable(
                make_hypothesis_record(
                    stage="27",
                    lottery=str(summary.get("lottery", "MINI_LOTO")),
                    experiment_family="prospective_signal_tracking",
                    hypothesis_name=f"{signal_id}_prospective_tracking_FINAL",
                    description=(
                        f"Stage 27 FINAL preregistered prospective result for {signal_id}, "
                        "registered once the preregistered evidence gate was reached."
                    ),
                    metric="mean_winner_rank_advantage",
                    sample_size=payload.get("evaluated_draw_count"),
                    effect=payload.get("mean_rank_advantage"),
                    raw_p_value=payload.get("paired_permutation_p_value"),
                    original_classification=payload.get("classification"),
                    source_file="backend/app/research/stage27_prospective_signals.py",
                    source_report="research/LotoSystem_V2_Stage27_Prospective_Signal_Tracking.md",
                    source_output=str(Path(root) / "MINI_LOTO" / "summary.json"),
                    provenance_status=VERIFIED_OUTPUT_ONLY,
                    historical_or_prospective="prospective",
                    discovery_or_confirmation="confirmation",
                    dataset_cutoff_draw=summary.get("prospective_start_draw"),
                    preregistered=True,
                    primary_or_secondary="primary",
                    notes=(
                        "DRAFT PREVIEW ONLY, generated by Stage 31. Not written to any Stage "
                        "29 export. Represents the ONE-TIME final registration this "
                        "preregistered hypothesis is entitled to now that its evidence gate "
                        "has been reached. Registering it twice, or registering an earlier "
                        "interim look, would be optional stopping / cherry-picking and must "
                        "not be done."
                    ),
                )
            )
        )
    return {
        "evidence_gate_classification": EVIDENCE_GATE_CLASSIFICATION,
        "gate_reached": bool(ready_signals),
        "signals_ready_for_registration": tuple(ready_signals),
        "draft_hypothesis_records": tuple(draft_records),
        "action_required": (
            "none; evidence gate not yet reached for any signal"
            if not ready_signals
            else (
                "A human maintainer must manually add these draft record(s) to "
                "build_repository_global_ledger()'s Stage 27 section EXACTLY ONCE, "
                "replacing the PROSPECTIVE_ACTIVE placeholder for the ready signal(s), then "
                "re-run Stage 29. Stage 31 does not perform this registration itself."
            )
        ),
    }


# ---------------------------------------------------------------------------
# F. Repository safety: .gitignore / disposable-artifact audit (read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepositoryHygieneAudit:
    gitignore_path: str
    gitignore_exists: bool
    missing_recommended_ignores: tuple[str, ...]
    disposable_tracked_file_count: int
    disposable_tracked_sample: tuple[str, ...]
    recommendations: tuple[str, ...]


def audit_repository_hygiene(repo_root: str | Path = ".") -> RepositoryHygieneAudit:
    """Read-only. Runs only `git ls-files` (a listing command; never `git
    add`/`commit`/`push`/`reset`/`restore`/`clean`). Reports recommendations;
    never edits .gitignore or untracks/deletes anything itself.
    """
    root = Path(repo_root)
    gitignore_path = root / ".gitignore"
    gitignore_text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    missing = tuple(
        pattern for pattern in RECOMMENDED_DISPOSABLE_IGNORES if pattern not in gitignore_text
    )

    tracked = _git_ls_files(root)
    disposable_prefixes = tuple(
        pattern.rstrip("/") + "/" for pattern in RECOMMENDED_DISPOSABLE_IGNORES
    )
    disposable_tracked = sorted(
        path for path in tracked if any(path.startswith(prefix) for prefix in disposable_prefixes)
    )

    recommendations: list[str] = []
    if missing:
        recommendations.append("Add to .gitignore: " + ", ".join(missing))
    if disposable_tracked:
        sample = ", ".join(disposable_tracked[:3])
        recommendations.append(
            f"{len(disposable_tracked)} disposable test-artifact file(s) are currently "
            f"tracked by git under disposable directories (e.g. {sample}). Recommended fix: "
            "add the missing pattern(s) above to .gitignore, then `git rm -r --cached` on "
            "those paths in a separate, deliberate commit so future pytest runs stop "
            "generating repository diffs. Stage 31 does not run this itself and does not "
            "delete anything from disk -- only untracking, never deleting, is recommended."
        )
    recommendations.append(
        "Evidence paths that are intentionally version-controlled and must NOT be added to "
        ".gitignore: " + ", ".join(EVIDENCE_PATH_HINTS)
    )
    return RepositoryHygieneAudit(
        gitignore_path=str(gitignore_path),
        gitignore_exists=gitignore_path.exists(),
        missing_recommended_ignores=missing,
        disposable_tracked_file_count=len(disposable_tracked),
        disposable_tracked_sample=tuple(disposable_tracked[:10]),
        recommendations=tuple(recommendations),
    )


def _git_ls_files(repo_root: Path) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    return tuple(line for line in completed.stdout.splitlines() if line)
