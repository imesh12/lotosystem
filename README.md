# LotoSystem

LotoSystem is a research-oriented Python system for studying Japanese LOTO6 and Mini Loto data. The project is designed to test whether statistical and machine learning strategies can rank lottery candidates differently from carefully constructed random baselines.

This repository is at Version 1.0.0. It includes the local operational lifecycle, paper-trading ledger, notifications, settings/API, simple frontend, and Windows deployment scripts. It intentionally does not include AI/LLM agents, new model families, database integration, authentication, cloud hosting, purchasing, payment, or real-money claims.

## Current Features

- FastAPI application shell
- `/api/health` endpoint
- Pydantic Settings configuration
- Structured logging setup
- pytest and Ruff configuration
- Initial documentation and repository structure
- LOTO6 and Mini Loto rule definitions
- Ticket, draw, prediction, and experiment domain models
- Domain validation tests
- Offline CSV historical draw loading
- Deterministic frequency, recency, distribution, pair, gap, and transition statistics
- Deterministic feature generation, research candidate generation, scoring, and walk-forward backtesting
- Prize/match classification for LOTO6 and Mini Loto
- Seeded uniform-random baseline replications
- Evaluation date ranges and strategy-vs-random comparison
- Deterministic dataset hashing and provenance summary
- JSON research result persistence
- `loto-research` CLI
- Authoritative Mizuho Bank historical import for LOTO6 and Mini Loto
- Canonical historical CSV exports under `data/processed/`
- Real-data two-ticket random baseline benchmark with theoretical sanity checks
- Paired statistical evaluation, bootstrap confidence intervals, multiple-comparison correction, and file-based experiment records
- Leakage-safe number-level Logistic Regression and Random Forest baseline evaluation
- Feature audit, temporal stability diagnostics, controlled `number-features-v2`, and ML feature ablation
- Two-ticket portfolio construction audit with overlap, coverage, and paired random comparison
- Immutable paper-trading prediction records and pending-result evaluation
- Manual post-draw operational cycle for updating history, evaluating pending predictions, and creating the next pending record
- Optional SMTP notifications generated from stored operational records
- Local operational settings API and simple dashboard/settings frontend
- Windows PowerShell wrappers for server startup, one-shot automation, Task Scheduler installation, status, and backup

## Requirements

- Python 3.12+

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` for local overrides. Do not commit `.env`.

## Run The API

```powershell
uvicorn backend.app.main:app --reload
```

On Windows, the Version 1 startup wrapper is preferred:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/start-server.ps1
```

Health check:

```text
GET http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Quality Checks

```powershell
pytest
ruff check .
ruff format --check .
```

## Version 1 Windows Operation

Version: 1.0.0

Manual status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/status.ps1
```

Manual one-shot automation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run-auto-cycle.ps1
```

Install local Windows scheduled tasks explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/install-scheduled-tasks.ps1
```

Created tasks:

- `LotoSystem-AutoRun`: wakes every 3 hours and runs Python `auto-run --lottery ALL`.
- `LotoSystem-Web`: starts the local FastAPI/frontend at user logon.

Uninstall:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/uninstall-scheduled-tasks.ps1
```

Create a local operational backup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/backup-data.ps1
```

See `docs/version-1-runbook.md` for setup, recovery, email, backup, and troubleshooting.

## Research CLI

After installing the project, the deterministic research CLI can run offline against a CSV file:

```powershell
loto-research --data data/raw/sample.csv --lottery LOTO6 validate-data
loto-research --data data/raw/sample.csv --lottery LOTO6 calculate-statistics
loto-research --data data/raw/sample.csv --lottery LOTO6 generate-features
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid generate-candidates
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid --seed 123 --baseline-replications 100 backtest
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid --seed 123 --baseline-replications 100 --evaluation-start 2024-01-01 --output data/exports/research.json run-research
```

CSV input must include `draw_number` and `draw_date`. The `lottery` column is supported and must match the requested lottery when present. Numbers can be provided as packed fields such as `main_numbers` and `bonus_numbers`, or as columns like `n1` through `n6` plus `bonus`/`bonus1`. Mini Loto uses `n1` through `n5`.

## Authoritative History Import

Stage 04 adds a Mizuho Bank collector behind the existing collector abstraction. Mizuho Bank's lottery winning-number archive is used as the authoritative source for historical winning numbers.

Update canonical exports:

```powershell
loto-research --lottery LOTO6 --output data/processed/loto6_history.csv update-history
loto-research --lottery MINI_LOTO --output data/processed/mini_loto_history.csv update-history
```

If Mizuho requires JavaScript-rendered archive tables, use the local browser bootstrap. Playwright uses installed Chrome/Edge when available, or its managed Chromium after one-time setup:

```powershell
python -m playwright install chromium
loto-research --lottery LOTO6 --output data/processed/loto6_history.csv browser-bootstrap-history
loto-research --lottery MINI_LOTO --output data/processed/mini_loto_history.csv browser-bootstrap-history
```

Use `--headed` to run the browser visibly:

```powershell
loto-research --lottery LOTO6 --headed --output data/processed/loto6_history.csv browser-bootstrap-history
```

The default import period is `2010-01-01` through the latest completed draw available from the source. Use `--history-start` and `--history-end` for controlled verification runs:

```powershell
loto-research --lottery LOTO6 --history-start 2010-01-01 --history-end 2024-12-31 --output data/processed/loto6_history.csv update-history
```

Canonical exports use:

```text
lottery,draw_number,draw_date,n1,n2,n3,n4,n5,n6,bonus,source,source_url,retrieved_at,content_hash
```

Mini Loto leaves `n6` blank. Repeated updates preserve unchanged existing draws, append new draws canonically, and fail if fetched authoritative records conflict with existing canonical draw data. Dataset hashes are calculated from canonical lottery/date/draw/number data, not file paths or retrieval timestamps.

If Mizuho blocks automated HTTP access from the local environment, save official Mizuho archive pages or CSV files under a local folder such as `data/raw/mizuho/`, then run the same update command with `--source-dir`:

```powershell
loto-research --lottery LOTO6 --source-dir data/raw/mizuho --output data/processed/loto6_history.csv update-history
loto-research --lottery MINI_LOTO --source-dir data/raw/mizuho --output data/processed/mini_loto_history.csv update-history
```

The manual bootstrap parser derives draw numbers, dates, winning numbers, and bonus numbers from file content, not filenames. It combines files, removes exact duplicates, detects conflicting records, applies the 2010 cutoff, and writes the same canonical CSV schema.

Historical statistical patterns do not guarantee future lottery outcomes. Candidates produced by this engine are research artifacts, not winning-number claims.

## Stage 05 Baseline Benchmark

The Stage 05 benchmark is the first real-data benchmark future ML must beat. It evaluates exactly two distinct tickets per draw, using seeded uniform random selection and the existing prize/match engine. It also computes independent combinatorial probabilities as a sanity check.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --baseline-replications 1000 --tickets-per-draw 2 --output data/exports/stage05_loto6_baseline_report.json baseline-benchmark
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --seed 123456 --baseline-replications 1000 --tickets-per-draw 2 --output data/exports/stage05_mini_loto_baseline_report.json baseline-benchmark
```

The benchmark reports match-count rates, prize-category counts, simulated ticket cost at 200 yen per ticket, Monte Carlo distribution summaries, and existing deterministic strategy comparisons. It does not calculate payout, ROI, or statistical significance. Random outcomes and historical strategy results do not imply future predictability.

## Stage 06 Statistical Evaluation

Stage 06 compares each deterministic strategy against a seeded random two-ticket baseline on the same target draws. It uses seeded bootstrap confidence intervals, a paired permutation test for mean-match differences, Holm multiple-comparison adjustment, and period stability checks.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage06_loto6_statistical_evaluation.json statistical-evaluation
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage06_mini_loto_statistical_evaluation.json statistical-evaluation
```

Experiment records are written under `data/exports/experiments/` when `--output` is used. Conclusion labels are deliberately conservative and never claim a proven predictive or winning model.

## Stage 07 ML Baseline

Stage 07 introduces the first supervised ML baseline. The ML problem is number-level: for each target draw and each valid lottery number, features are calculated only from draws before the target draw. The label is `1` only when the number appears in the target draw's main numbers; bonus numbers are not positive targets.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage07_loto6_ml_baseline.json ml-baseline
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage07_mini_loto_ml_baseline.json ml-baseline
```

The command evaluates Logistic Regression and Random Forest with walk-forward training, converts ranked number scores into exactly two distinct tickets per draw, and compares ML results against the paired seeded random baseline with the same statistical controls used in Stage 06. ML scores are ranking values, not winning probabilities.

## Stage 08 Feature Evaluation

Stage 08 audits `number-features-v1`, adds a small historical-only `number-features-v2`, and evaluates explicit feature groups through ablation with the same Logistic Regression and Random Forest models. It reports constant or near-constant features, highly correlated feature pairs, temporal shifts, feature importance diagnostics, leakage checks, and comparison against Stage 07.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage08_loto6_feature_evaluation.json feature-evaluation
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage08_mini_loto_feature_evaluation.json feature-evaluation
```

Feature expansion is treated as hypothesis testing. A stronger historical result is not accepted as proof of future predictability.

## Stage 09 Portfolio Evaluation

Stage 09 keeps the selected Stage 08 number-scoring model/features unchanged and audits how scores become exactly two tickets. It compares the current top-ranked construction with coverage, diversified, and predefined overlap-penalty portfolio methods.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage09_loto6_portfolio_evaluation.json portfolio-evaluation
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage09_mini_loto_portfolio_evaluation.json portfolio-evaluation
```

Portfolio objective scores are construction heuristics only. They are not probabilities and do not imply guaranteed improvement.

## Stage 10 Paper-Trading Predictions

Stage 10 generates production-style future-draw records using the current conservative research selections. Records are saved under `data/predictions/` and are designed to be immutable operational history.

```powershell
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --tickets-per-draw 3 --seed 123456 generate-next
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv --tickets-per-draw 3 --seed 123456 generate-next
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv evaluate-predictions
loto-research --lottery MINI_LOTO --data data/processed/mini_loto_history.csv evaluate-predictions
```

Prediction records are paper-trading only. They record source-dataset provenance, target draw metadata, model/features, portfolio method, seed/config, generated tickets, and cost at 200 yen per ticket. Evaluation fills match/prize-category fields after the actual result appears in canonical history. Payout and ROI remain unavailable until official draw-specific payout collection exists.

## Stage 11 Post-Draw Cycle

Stage 11 adds a single manual command for the normal post-draw operating loop:

```powershell
loto-research --lottery LOTO6 --tickets-per-draw 3 --seed 123456 run-cycle
loto-research --lottery MINI_LOTO --tickets-per-draw 3 --seed 123456 run-cycle
```

The cycle updates the canonical Mizuho history CSV, validates the updated data, evaluates any saved `PENDING` prediction whose result is now present, and then creates the next future `PENDING` prediction only when one does not already exist. Cycle audit records are written under `data/predictions/<LOTTERY>/cycles/`.

Repeated runs with no new official result are intended to be no-ops: no appended draws, no duplicate evaluation, and no regenerated prediction. Browser/history update failures stop the cycle before evaluation or next-prediction generation.

Stage 11B keeps Mizuho as the primary source and adds SMBC's public lottery result XML as the configured secondary latest-result source. If both automated sources fail, append an emergency result manually through the same validation and conflict-detection pipeline:

```powershell
loto-research --lottery LOTO6 --draw-number 2131 --draw-date 2026-08-24 --numbers "01,02,03,04,05,06" --bonus "07" --confirm-manual add-result
```

Manual entries are provenance-marked as `manual` and do not bypass lottery validation.

## Stage 12 Paper Financial Settlement

Stage 12 settles evaluated paper predictions into simulated financial records under `data/settlements/`. Official payout data is attached when available from the result source; otherwise the settlement is marked `PAYOUT_PENDING` and paper gross/net values remain unavailable.

```powershell
loto-research --lottery LOTO6 financial-summary
loto-research --lottery MINI_LOTO financial-summary
loto-research --lottery ALL financial-summary
loto-research --lottery LOTO6 --draw-number 2131 --tier 5th --payout 1000 --confirm-manual add-payout
```

Financial fields are explicitly paper-trading fields: `paper_total_cost_yen`, `paper_gross_winnings_yen`, and `paper_net_yen`. They do not represent purchased tickets or actual bank/account earnings.

## Stage 13 Automated Lifecycle

Stage 13 adds one-shot automation commands that can be run unattended by a future cron or Task Scheduler entry:

```powershell
loto-research automation-status
loto-research --lottery ALL --tickets-per-draw 3 --seed 123456 auto-run
```

Automation uses Asia/Tokyo time, checks results after the configured evening result-check window, and then delegates to the existing `run-cycle` flow. It is idempotent: no duplicate predictions, duplicate history rows, duplicate settlements, or retroactive predictions are created. A local lock under `data/automation/` prevents concurrent runs and supports stale-lock recovery.

If no official result is available, automation returns a clean retry recommendation. If result sources fail, it reports `SOURCE_FAILURE` and does not mutate history, predictions, or settlements.

## Stage 14 Email Notifications

Stage 14 adds reusable operational reports and optional SMTP email delivery. Email is downstream of the lifecycle: history updates, prediction evaluation, settlements, and next prediction generation are never rolled back because delivery fails.

```powershell
loto-research notification-status
loto-research send-pending-notifications
loto-research test-email
```

Email is disabled by default and configured only through environment variables:

```text
LOTO_EMAIL_ENABLED=false
LOTO_EMAIL_FROM=
LOTO_EMAIL_TO=
LOTO_SMTP_HOST=
LOTO_SMTP_PORT=587
LOTO_SMTP_USERNAME=
LOTO_SMTP_PASSWORD=
LOTO_SMTP_USE_TLS=true
```

Notification records are stored under `data/notifications/`. Supported types are `DRAW_PROCESSED`, `SOURCE_FAILURE`, and `PAYOUT_COMPLETED`. Reports are plain text, concise, and clearly labeled `PAPER TRADING / SIMULATED`. If payout information is unavailable, paper winnings and net remain `pending`; pending payout is not treated as zero.

## Stage 15 Settings And Operational API

Stage 15 adds file-backed operational settings and a local FastAPI read surface for a future Version 1 dashboard.

Settings are stored in `config/operational_settings.json` when changed:

```json
{
  "LOTO6": {"enabled": true, "tickets_per_draw": 3},
  "MINI_LOTO": {"enabled": true, "tickets_per_draw": 3},
  "email_enabled": false
}
```

Changing `tickets_per_draw` affects only newly created future prediction records. Existing `PENDING` or `EVALUATED` prediction records are not rewritten.

Operational API endpoints:

```text
GET /api/status
GET /api/settings
PUT /api/settings
GET /api/lotteries
GET /api/lotteries/{lottery}/latest
GET /api/lotteries/{lottery}/next-prediction
GET /api/lotteries/{lottery}/history?limit=20&offset=0
GET /api/financial/summary?lottery=ALL&period=all_time
GET /api/notifications/status
```

The API does not expose SMTP credentials or arbitrary filesystem paths. It returns paper-trading fields only and does not create predictions retroactively.

## Blueprint

The master architecture is documented in `fullblueprint.md`. Development stages must follow that file and avoid implementing future-stage functionality early.
