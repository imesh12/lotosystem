# Deterministic Research Engine

Stage 04 keeps the deterministic research engine independent while adding authoritative historical import. It uses validated historical draw data to calculate deterministic statistics, generate deterministic features, produce explainable research candidates, evaluate those candidates with walk-forward backtesting, classify match/prize tiers, and compare strategies against seeded uniform-random baselines.

Historical statistical patterns do not guarantee future lottery outcomes. This project is for statistical research, entertainment, and experimental evaluation.

## Data Model

`HistoricalDraw` represents one draw:

- draw number
- draw date
- sorted main numbers
- sorted bonus numbers
- optional source metadata

Validation rejects invalid draw identifiers, wrong number counts, out-of-range values, duplicate main numbers, invalid bonus numbers, duplicate draw identities, duplicate draw numbers, source rows arriving out of chronological order, invalid lottery codes, and mixed-lottery input.

CSV loading supports packed fields such as `main_numbers` and `bonus_numbers`, or numbered columns such as `n1` through `n6` plus `bonus`/`bonus1`. Mini Loto uses `n1` through `n5`.

## Provenance

Optional provenance fields are supported:

- `source`
- `source_url`
- `retrieved_at`
- `content_hash`

Fixtures in `tests/fixtures` are clearly marked synthetic structural fixtures. They are not real historical draw records.

## Authoritative Import

The Mizuho Bank collector is implemented behind `CollectorInterface` and separate from statistics, features, candidate generation, scoring, and backtesting. It parses Mizuho archive responses into `HistoricalDraw` records for:

- LOTO6
- Mini Loto

The default production import range is:

```text
2010-01-01 through latest completed draw available from source
```

Canonical exports are:

```text
data/processed/loto6_history.csv
data/processed/mini_loto_history.csv
```

Canonical rows include lottery code, draw number, draw date, main numbers, bonus number, source identifier, source URL, retrieval timestamp, and content hash. Dataset hashes depend on canonical research data only, so equivalent normalized datasets produce the same hash even when imported at different times.

Incremental updates preserve unchanged records, append newly fetched records, and fail when a fetched record conflicts with existing canonical draw data. The importer validates every written dataset with the same Stage 03 draw validation used by research commands.

When automated access to Mizuho is blocked, locally saved official Mizuho archive pages can be imported with `--source-dir`. Files may be `.html`, `.htm`, `.csv`, or `.txt` and may be mixed in one folder. The parser derives research facts from file content, not filenames, then deduplicates, detects conflicts, applies the configured date range, and writes the same canonical processed CSV.

## Statistics

The engine calculates:

- overall and windowed frequency
- recency and recent-window activity
- currently absent numbers, recently frequent numbers, and recently inactive numbers
- odd/even counts and patterns
- low/high counts and patterns using configurable thresholds
- sum, mean, minimum, maximum, median, sum frequencies, and configured percentiles
- consecutive pair counts, consecutive group counts, and maximum consecutive group length
- unordered pair co-occurrence with configurable minimum observations
- historical gaps and current gaps
- draw-to-draw overlaps, entering numbers, and leaving numbers

The engine reports statistics only. It does not label numbers as due, lucky, guaranteed, or predictive.

## Features

Candidate feature generation is separate from scoring. Features include:

- frequency total
- recent frequency total
- recency total
- average gap total
- pair strength
- odd/even balance
- low/high balance
- total sum
- distance from historical median sum
- consecutive-pair count

## Candidate Strategies

Supported deterministic strategies:

- `fixed-baseline`
- `frequency`
- `recency`
- `balanced`
- `pair`
- `hybrid`

Candidates are research artifacts. They are not claims about future winning numbers.

## Scoring

Scoring is transparent and deterministic. Each score contains:

- frequency component
- recency component
- gap component
- pair component
- distribution component
- pattern component
- total score

Weights are intentionally simple and not optimized in this stage.

## Backtesting

Backtesting uses a walk-forward process:

```text
target draw N
training data = draws before N
generate candidates from training data only
compare candidate numbers to target draw
```

This prevents look-ahead bias. The evaluation records average matches, maximum matches, and a full match-count distribution, then compares the selected strategy to a simple fixed deterministic baseline.

Evaluation can be restricted with `evaluation_start` and `evaluation_end`. Earlier history is still available for training if it appears before the evaluation window.

`backtest_candidate_count` evaluates up to that many generated candidates for each target draw.

## Prize Classification

The prize engine uses set membership and official rank conditions represented in `LotteryDefinition.prize_tiers`.

LOTO6:

- 1st: 6 main matches
- 2nd: 5 main matches plus bonus
- 3rd: 5 main matches
- 4th: 4 main matches
- 5th: 3 main matches

Mini Loto:

- 1st: 5 main matches
- 2nd: 4 main matches plus bonus
- 3rd: 4 main matches
- 4th: 3 main matches

Prize tier classification is separate from payout amounts.

## Random Baseline

The random baseline uniformly samples valid lottery tickets with a local `random.Random(seed)` instance. It never uses global random state. Replications derive seeds from the configured master seed:

```text
replication_seed = seed + replication_index
```

The baseline summary reports mean matches, median matches, standard deviation, max matches, match distribution, 3+/4+/5+ match rates, and prize-qualified rate. These are descriptive comparisons, not statistical-significance claims.

## Stage 05 Real-Data Benchmark

Stage 05 establishes the real-data benchmark future ML must beat. For each historical draw it generates exactly two distinct valid random tickets using a local seeded RNG. LOTO6 tickets contain 6 unique numbers from 1 through 43. Mini Loto tickets contain 5 unique numbers from 1 through 31.

The benchmark defaults are:

```text
seed = 123456
baseline_replications = 1000
tickets_per_draw = 2
ticket_price = 200 yen
```

Each ticket is evaluated with the same prize/match engine used by walk-forward backtesting. The benchmark aggregates match-count rates, prize-qualified rates, prize-category counts, Monte Carlo distribution summaries, and simulated cost. It does not calculate payout amounts, net profit, or ROI.

An independent combinatorial sanity check calculates exact uniform-random match probabilities using combinations/hypergeometric reasoning. This check is separate from the random generator so it can catch obvious simulation bugs.

Existing deterministic strategies (`frequency`, `recency`, `pair`, `balanced`, and `hybrid`) are evaluated with two distinct candidates per target draw where walk-forward history is available. Every target draw uses only earlier draws. Strategy performance is descriptive historical evidence only; it is not a guarantee of future performance and is not a statistical-significance claim.

## Stage 06 Statistical Validation

Stage 06 evaluates whether observed Stage 05 strategy-vs-random differences are likely to be meaningful or random variation. The comparison is paired by target draw: each strategy's two-ticket result is compared with a seeded random two-ticket portfolio for the same historical draw.

The statistical evaluation uses:

- seeded bootstrap confidence intervals
- paired permutation/randomization tests for mean-match differences
- paired rate differences for 3+, 4+, 5+, and prize-qualified outcomes
- Holm correction across multiple strategy comparisons
- period stability checks across 2010-2014, 2015-2019, 2020-2023, and 2024-latest

Conclusion labels are conservative:

- `no_evidence`
- `weak_signal`
- `statistically_detectable_small_effect`
- `unstable_effect`
- `needs_more_validation`

The engine must not use labels such as proven predictive, winning model, or guaranteed advantage.

## Stage 07 ML Baseline

Stage 07 adds the first leakage-safe supervised ML baseline with scikit-learn:

- Logistic Regression
- Random Forest

The ML task is number-level. For each historical target draw and each valid lottery number, the feature row is calculated from draws strictly before the target draw. The target label is `1` only when that number appears in the target draw's main numbers. Bonus numbers are never positive labels.

Feature version `number-features-v1` includes deterministic number-level features such as lifetime frequency rate, recent window frequencies, current gap, mean gap, gap standard deviation, maximum observed gap, recent activity, and pair-strength aggregates.

Walk-forward evaluation fits each model only on earlier labeled rows for the current target draw. Model scores rank valid numbers, then the two-ticket baseline converts ranked scores into exactly two distinct canonical tickets. The comparison against random uses the same paired bootstrap, permutation-test, effect-size, and Holm-adjustment approach established in Stage 06.

ML results remain experimental. A higher historical score is not a winning probability and does not prove future predictability.

## Stage 08 Feature Audit And Expansion

Stage 08 asks whether feature quality, rather than model choice, is limiting the first ML baseline. It keeps the same Logistic Regression and Random Forest models and evaluates feature changes under the same leakage-safe walk-forward rules.

The feature audit reports per-feature missing rates, min/max/mean/standard deviation, constant or near-constant status, highly correlated pairs at absolute correlation `>= 0.95`, and temporal mean shifts across the Stage 06 periods.

Feature version `number-features-v2` adds a small set of historical-only hypotheses:

- frequency momentum and short-vs-long ratios
- normalized current-gap features
- gap z-score with zero-variance protection
- seen-rate and pair-strength aggregates
- previous-draw presence and previous-draw pair-strength context

Feature groups are evaluated by ablation, including v1, frequency-only, gap-only, pair-only, v2, and v2 with major expansion groups removed. The conclusion labels remain conservative: no feature improvement, weak feature signal, unstable feature signal, promising for next model stage, or needs more validation.

## Stage 09 Portfolio Construction

Stage 09 audits the two-ticket construction step while keeping Stage 08 number-scoring models and features fixed:

- LOTO6: `gap_only` features with Random Forest
- Mini Loto: `pair_only` features with Logistic Regression

The current construction is deterministic: Ticket 1 uses the top K scored numbers and Ticket 2 uses the next K scored numbers, with score-descending and number-ascending tie-breaking.

Controlled portfolio methods compare:

- `top_ranked`
- `diversified`
- `coverage`
- `overlap_penalty_0`, `overlap_penalty_0.25`, `overlap_penalty_0.5`, `overlap_penalty_1`

Each method produces exactly two distinct valid tickets. Metrics include average ticket overlap, unique-number coverage, average matches per ticket, average matches per two-ticket portfolio, best-ticket matches per draw, portfolio prize-qualified rate, and paired statistical comparison against two distinct uniformly random tickets.

Portfolio objective scores are ticket-construction heuristics, not winning probabilities.

## Stage 10 Paper-Trading Prediction Records

Stage 10 creates the first production-style future-draw workflow without automation, purchasing, payout fetching, or a database. The current conservative strategy is:

- LOTO6: Random Forest, `gap_only`, `top_ranked`
- Mini Loto: Logistic Regression, `pair_only`, `top_ranked`

The `generate-next` command trains only on the canonical history available before the next scheduled draw, creates a future target draw number/date, and saves an immutable `PENDING` prediction record under `data/predictions/<LOTTERY>/<draw_number>.json`.

Records include dataset hash, latest source draw, generated timestamp, model/features, portfolio method/version, seed/config, configurable `tickets_per_draw`, generated tickets, cost at 200 yen per ticket, and warnings that this is paper-trading only.

The `evaluate-predictions` command checks pending records against canonical history. When the actual target draw exists, it records actual numbers, match counts, bonus matches, prize categories, best match count, and prize-qualified ticket count, then marks the record `EVALUATED`. It does not calculate payout or ROI.

## Stage 11 Post-Draw Operational Cycle

Stage 11 wraps the existing authoritative history update and Stage 10 paper-trading functions into one manual operational command:

```powershell
loto-research --lottery LOTO6 --tickets-per-draw 3 --seed 123456 run-cycle
loto-research --lottery MINI_LOTO --tickets-per-draw 3 --seed 123456 run-cycle
```

The cycle order is fixed:

1. Load the current canonical history CSV.
2. Run the existing Mizuho browser-backed incremental history update.
3. Validate the updated canonical history.
4. Evaluate saved `PENDING` prediction records whose actual target draw now exists.
5. Generate the next future prediction if no valid future `PENDING` record already exists.
6. Save a compact cycle audit record under `data/predictions/<LOTTERY>/cycles/`.

Catch-up behavior appends all newly validated missing official draws, evaluates only predictions that were previously saved, and does not fabricate retroactive predictions for draws where no pre-draw record existed. If the history update fails or a conflict is detected, the cycle fails before prediction evaluation or next-prediction generation.

Stage 11B makes latest-result ingestion resilient:

- primary source: Mizuho browser-backed latest/backnumber pages
- secondary source: SMBC public lottery result XML for latest LOTO6 and Mini Loto results
- manual fallback: explicit CLI entry through canonical validation and merge logic

`NO_NEW_RESULT` is treated as a successful no-op. `SOURCE_FAILURE` triggers fallback to the next configured automated source. If all automated sources fail, the cycle stops without mutating history or predictions and requires manual input.

```powershell
loto-research --lottery LOTO6 --draw-number 2131 --draw-date 2026-08-24 --numbers "01,02,03,04,05,06" --bonus "07" --confirm-manual add-result
```

## Stage 12 Paper Financial Settlement

Stage 12 extends evaluated paper predictions with draw-specific settlement records. Settlement uses the existing prize classifier, then attaches official payout amounts where available from the configured result source. If payout data is unavailable for a winning tier, the settlement remains `PAYOUT_PENDING`; no gross or net value is fabricated.

Settlement records are file-based:

```text
data/settlements/
  LOTO6/<draw>.json
  MINI_LOTO/<draw>.json
  ledger.json
```

Financial fields use paper-trading terminology:

- `paper_total_cost_yen = ticket_count * 200`
- `paper_gross_winnings_yen = sum(ticket payout_yen)` when all winning-ticket payouts are known
- `paper_net_yen = paper_gross_winnings_yen - paper_total_cost_yen`

Manual payout completion is available only for an existing evaluated settlement and requires explicit confirmation:

```powershell
loto-research --lottery LOTO6 --draw-number 2131 --tier 5th --payout 1000 --confirm-manual add-payout
loto-research --lottery ALL financial-summary
```

These are simulated accounting records. They do not claim that tickets were purchased or that money was actually won.

## Stage 13 Automated Lifecycle

Stage 13 keeps `run-cycle` as the operational source of truth and adds a thin, timezone-aware automation layer around it. The automation layer decides whether work is due, acquires a local lock, runs only the currently due lottery lifecycle, and saves a compact run audit record under `data/automation/runs/`.

Commands:

```powershell
loto-research automation-status
loto-research --lottery ALL --tickets-per-draw 3 --seed 123456 auto-run
```

Scheduling is based on Asia/Tokyo. LOTO6 uses the Monday/Thursday lottery schedule; Mini Loto uses Tuesday. Results are not assumed to be available immediately at draw time, so the default one-shot policy waits until a configurable evening result-check time and then recommends bounded retries after `NO_NEW_RESULT` or `SOURCE_FAILURE`.

Automation preserves paper-trading integrity:

- saved predictions are not overwritten
- evaluated predictions are not rewritten
- settlements are idempotent
- `PAYOUT_PENDING` does not block next prediction generation
- missing historical prediction records are not fabricated during catch-up
- source failures stop mutation and require a later retry or manual result entry

## Stage 14 Operational Reports And Email

Stage 14 creates deterministic operational report DTOs from stored prediction and settlement records. Email rendering is a downstream presentation layer; it does not recalculate results independently and does not mutate history, prediction, settlement, or next-prediction state.

Commands:

```powershell
loto-research notification-status
loto-research send-pending-notifications
loto-research test-email
```

SMTP delivery is optional and configured only through environment variables documented in `.env.example`. Notification records are stored under `data/notifications/` with delivery status, attempt count, last error, and sent timestamp. Successfully sent `DRAW_PROCESSED` notifications are not resent by repeated automation runs; failed or disabled records can be retried later.

Notification types:

- `DRAW_PROCESSED`: result, saved prediction tickets, match/prize results, paper cost/gross/net, and next prediction.
- `SOURCE_FAILURE`: compact operational alert with latest history, pending prediction, attempted sources, and manual action reminder.
- `PAYOUT_COMPLETED`: financial update when a previously pending payout is completed.

All notification text is labeled `PAPER TRADING / SIMULATED`. `PAYOUT_PENDING` keeps paper winnings and net as `pending`; pending payout is never treated as zero.

## Stage 15 Settings And Operational API

Stage 15 adds a small settings service and dashboard-ready FastAPI endpoints. Settings are file-backed at `config/operational_settings.json` and are intentionally limited to operational preferences:

- LOTO6: `enabled`, `tickets_per_draw`
- Mini Loto: `enabled`, `tickets_per_draw`
- global: `email_enabled`

`tickets_per_draw` is validated as a positive integer up to 20. Unknown settings fields are rejected. SMTP credentials are not stored in this file.

Automation reads these settings before taking action:

- `enabled=false` skips automated work for that lottery while preserving existing records.
- changing `tickets_per_draw` affects only newly created future predictions.
- existing immutable prediction records are not rewritten when settings change.

Operational API endpoints:

```text
GET /api/status
GET /api/settings
PUT /api/settings
GET /api/lotteries
GET /api/lotteries/{lottery}/latest
GET /api/lotteries/{lottery}/next-prediction
GET /api/lotteries/{lottery}/history
GET /api/financial/summary
GET /api/notifications/status
```

Routes are thin wrappers around research services. They expose summary DTOs for the future frontend and avoid model internals, filesystem paths, SMTP secrets, and full-history bulk responses.

## Persistence

Research results can be saved as JSON. Saved output includes:

- dataset version
- dataset hash
- configuration
- strategy
- generated candidates and scores
- backtest results
- persistence timestamp

The dataset hash and configuration allow repeated runs to be compared for reproducibility.

Deterministic scientific payload serialization is available separately from operational persistence metadata. Saved files retain `persisted_at`, but deterministic comparison should use the result payload.

## CLI

Commands:

```powershell
loto-research --lottery LOTO6 --output data/processed/loto6_history.csv update-history
loto-research --lottery MINI_LOTO --output data/processed/mini_loto_history.csv update-history
loto-research --lottery LOTO6 --source-dir data/raw/mizuho --output data/processed/loto6_history.csv update-history
loto-research --lottery MINI_LOTO --source-dir data/raw/mizuho --output data/processed/mini_loto_history.csv update-history
loto-research --data data/raw/sample.csv --lottery LOTO6 validate-data
loto-research --data data/raw/sample.csv --lottery LOTO6 calculate-statistics
loto-research --data data/raw/sample.csv --lottery LOTO6 generate-features
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid generate-candidates
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid backtest
loto-research --data data/raw/sample.csv --lottery LOTO6 --strategy hybrid --output data/exports/research.json run-research
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --baseline-replications 1000 --tickets-per-draw 2 --output data/exports/stage05_loto6_baseline_report.json baseline-benchmark
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage06_loto6_statistical_evaluation.json statistical-evaluation
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage07_loto6_ml_baseline.json ml-baseline
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage08_loto6_feature_evaluation.json feature-evaluation
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --seed 123456 --bootstrap-replications 10000 --output data/exports/stage09_loto6_portfolio_evaluation.json portfolio-evaluation
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv --tickets-per-draw 3 --seed 123456 generate-next
loto-research --lottery LOTO6 --data data/processed/loto6_history.csv evaluate-predictions
loto-research --lottery LOTO6 --tickets-per-draw 3 --seed 123456 run-cycle
loto-research automation-status
loto-research --lottery ALL --tickets-per-draw 3 --seed 123456 auto-run
loto-research notification-status
loto-research send-pending-notifications
```

## Known Limitations

- Mizuho import depends on the current official archive and CSV response shapes.
- Parser fixtures are intentionally small and do not duplicate the complete web archive.
- Prize classification and paper settlement are included, but payout collection is limited to configured result-source availability or manual completion.
- Stage 10-15 prediction, cycle, automation, settlement, notification, and settings records are file-based; no database registry or OS scheduler installation is included.
- Stage 05 tracks simulated ticket cost, but does not calculate payout, net profit, or ROI.
- No database integration is included.
- Stage 07 includes only Logistic Regression and Random Forest baselines; no model optimization or advanced ML is included.
- Stage 06/07/08/09 add bootstrap and permutation diagnostics, but no formal production-readiness claim is included.
- No LLM, prompt, embedding, vector database, or agent code is included.
