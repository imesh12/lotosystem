# LotoSystem

LotoSystem is a research-oriented Python system for studying Japanese LOTO6 and Mini Loto data. The project is designed to test whether statistical and machine learning strategies can rank lottery candidates differently from carefully constructed random baselines.

This repository is currently at Stage 07: First ML Baseline. It intentionally does not include deep learning, scheduling, database integration, LLM integrations, frontend work, or automatic ticket purchasing.

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

## Blueprint

The master architecture is documented in `fullblueprint.md`. Development stages must follow that file and avoid implementing future-stage functionality early.
