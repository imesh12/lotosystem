Absolutely. Below is the complete fullblueprint.md. You can copy everything from the code block and save it as:

C:\Users\imcom\Downloads\loto-system\fullblueprint.md
# LotoSystem — Full Blueprint


## 1. Project Overview


**Project name:** LotoSystem


**Local project path:**


`C:\Users\imcom\Downloads\loto-system`


**GitHub repository:**


`https://github.com/imesh12/lotosystem.git`


LotoSystem is a research-oriented AI/ML platform for Japanese:


- LOTO6
- Mini Loto


The system will research whether statistical analysis, machine learning, simulation, ensemble methods, and ticket optimization can produce candidate tickets that perform better than properly designed random baselines.


The weekly production target is:


```text
LOTO6
├── Ticket 1
└── Ticket 2


Mini Loto
├── Ticket 1
└── Ticket 2

Total:

2 × LOTO6
2 × Mini Loto

The system is intended for research and entertainment. It must never claim guaranteed winning numbers.

2. Main Research Hypothesis

The system should investigate:

Can historical lottery data contain measurable statistical signals that allow an AI/ML system to rank candidate combinations better than random selection?

We do NOT assume that this hypothesis is true.

The system must prove or disprove it through:

historical testing
walk-forward backtesting
out-of-sample testing
random baselines
statistical analysis
future real-world tracking

The goal is not:

"AI says these numbers will win."

The goal is:

"According to our tested model, these candidates have the highest
relative score under the current strategy."
3. Important Scientific Principle

Lottery draws should be treated as random unless evidence demonstrates otherwise.

Therefore:

Pattern found
      ↓
Hypothesis
      ↓
Backtest
      ↓
Out-of-sample test
      ↓
Compare against random
      ↓
Statistical evaluation
      ↓
Only then consider the pattern useful

Do not automatically treat:

hot numbers
cold numbers
overdue numbers
repeated pairs
consecutive numbers
odd/even patterns
number sums
historical frequency

as predictive.

They are features/hypotheses to test.

4. Supported Lotteries
LOTO6

Initial configuration:

Name:
LOTO6


Number range:
1–43


Main numbers:
6


Bonus numbers:
1


Ticket price:
¥200
Mini Loto

Initial configuration:

Name:
Mini Loto


Number range:
1–31


Main numbers:
5


Bonus numbers:
1


Ticket price:
¥200

Official rules and prize classifications must be verified against authoritative current sources before production use.

5. Generic Lottery Architecture

Do not hard-code LOTO6 logic throughout the application.

Use a generic lottery definition:

LotteryDefinition
├── code
├── name
├── number_min
├── number_max
├── numbers_per_ticket
├── bonus_enabled
├── ticket_price
├── draw_frequency
└── draw_schedule

Example:

LotteryDefinition(
    code="LOTO6",
    name="LOTO6",
    number_min=1,
    number_max=43,
    numbers_per_ticket=6,
    bonus_enabled=True,
    ticket_price=200,
)

Mini Loto should use the same architecture.

6. Core Architecture

The system is divided into:

DATA
  ↓
VALIDATION
  ↓
STATISTICS
  ↓
FEATURE ENGINEERING
  ↓
MODEL TRAINING
  ↓
ENSEMBLE
  ↓
CANDIDATE GENERATION
  ↓
TICKET OPTIMIZATION
  ↓
BACKTESTING
  ↓
EVALUATION
  ↓
WEEKLY PREDICTION

LLM functionality is a separate layer:

LLM
 ↓
Research
 ↓
Experiment proposals
 ↓
Explanation
 ↓
Reports

The LLM is NOT the core mathematical prediction engine.

7. LLM Provider Architecture

The system must support multiple AI providers.

Initial provider:

Ollama

Future provider:

OpenAI API

Potential future providers:

Gemini
Anthropic
Other compatible providers

Use an abstraction:

Application
     ↓
LLMProvider
     ↓
Provider Factory
     ├── Ollama
     ├── OpenAI
     └── Gemini

Business logic must never directly call Ollama or OpenAI.

Example:

provider = llm_factory.create()
result = provider.generate(prompt)

Configuration:

LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

Later:

LLM_PROVIDER=openai
LLM_MODEL=<configured-model>
OPENAI_API_KEY=<secret>

Changing providers must not require changes to:

database
prediction engine
backtesting
ticket optimizer
statistical engine
8. Repository Structure

Recommended:

loto-system/
│       ├── ensemble/
│       │   ├── scorer.py
│       │   ├── voting.py
│       │   └── calibration.py
│       │
│       ├── backtesting/
│       │   ├── engine.py
│       │   ├── walk_forward.py
│       │   ├── metrics.py
│       │   ├── baselines.py
│       │   └── leakage_checks.py
│       │
│       ├── evaluation/
│       │   ├── prize_calculator.py
│       │   ├── statistical_tests.py
│       │   ├── confidence.py
│       │   └── reports.py
│       │
│       ├── agents/
│       │   ├── manager.py
│       │   ├── research.py
│       │   ├── analysis.py
│       │   ├── experiment.py
│       │   ├── explanation.py
│       │   └── report.py
│       │
│       ├── llm/
│       │   ├── interface.py
│       │   ├── factory.py
│       │   ├── ollama.py
│       │   ├── openai.py
│       │   └── gemini.py
│       │
│       ├── scheduler/
│       │   ├── scheduler.py
│       │   └── jobs.py
│       │
│       └── services/
│           ├── draw_service.py
│           ├── prediction_service.py
│           ├── backtest_service.py
│           ├── experiment_service.py
│           └── report_service.py
│
├── frontend/
│
├── scripts/
│   ├── import_history.py
│   ├── update_draws.py
│   ├── calculate_features.py
│   ├── train_models.py
│   ├── run_backtest.py
│   └── generate_weekly.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── exports/
│
├── models/
│
├── notebooks/
│
└── tests/
    ├── unit/
    ├── integration/
    └── backtest/

The structure may evolve, but responsibilities must remain separated.

9. Database

Production database:

PostgreSQL

Development may support:

SQLite

Core tables:

lotteries
draws
draw_numbers
tickets
predictions
prediction_numbers
features
model_runs
model_scores
backtests
backtest_results
experiments
weekly_reports
data_sources
10. Lottery Table

Fields:

id
code
name
number_min
number_max
numbers_per_ticket
bonus_enabled
ticket_price
draw_frequency
draw_schedule
active
created_at
updated_at
11. Draw Table

Fields:

id
lottery_id
draw_number
draw_date
source_id
created_at
updated_at

Draw number should be unique per lottery.

12. Draw Numbers

Fields:

id
draw_id
number
position
is_bonus

position is source metadata only.

Winning calculations must use number membership, not ticket display order.

13. Ticket Representation

All tickets must be normalized.

Example:

03 08 15 24 31 42

Canonical representation:

03-08-15-24-31-42

These are the same ticket:

03 08 15 24 31 42


42 31 24 15 08 03


24 03 42 08 15 31

Internally use sorted sets.

Do not treat number position as a predictive feature.

14. Prize Matching Engine

Create a generic matching engine.

For each ticket:

ticket numbers
        ↓
winning main numbers
        ↓
set intersection
        ↓
match count
        ↓
bonus match
        ↓
prize classification

Example:

Winning:
03 08 15 24 31 42


Bonus:
19


Ticket:
42 31 15 08 03 24


Main matches:
6

The ticket wins the appropriate top prize according to official rules.

The engine must be independently tested.

15. Data Collection

Collector architecture:

CollectorInterface
       │
       ├── OfficialSourceCollector
       ├── SecondarySourceCollector
       └── CSV/ManualCollector

Every record should maintain provenance:

source
source_url
retrieved_at
content_hash

The system must never silently overwrite historical records.

16. Historical Data

The target dataset should include:

All available LOTO6 historical draws


+


All available Mini Loto historical draws

The requested research period includes:

2025
2026

but the model should use older history where available.

This gives the system more data for:

long-term frequency
gap analysis
pair analysis
seasonality experiments
model training
backtesting
17. Data Validation

Every imported draw must be checked for:

valid lottery
valid draw number
valid date
valid number range
no duplicate main numbers
valid bonus
no duplicate draw
source consistency

If data is invalid:

Do not train.
Do not produce a production prediction.
Report the problem.
18. Data Leakage

This is one of the most important rules in the entire project.

When predicting draw X, the system may only use information available before draw X.

Example:

Prediction:
2025-08-06

The model may use:

2025-08-05 and earlier

It must NOT use:

2025-08-07
2025-08-08
...

Do not calculate features using the complete historical dataset before splitting.

Every feature must have an "as-of" date.

19. Walk-Forward Validation

Primary evaluation:

Train
  ↓
Predict next draw
  ↓
Record actual result
  ↓
Move forward
  ↓
Train/update
  ↓
Predict next draw

Example:

Train: historical data → 2023-01
Predict: 2023-02


Train: historical data → 2023-02
Predict: 2023-03


Train: historical data → 2023-03
Predict: 2023-04

This must simulate real production behavior.

20. Statistics Engine

Implement:

Frequency
lifetime frequency
5-draw frequency
10-draw frequency
20-draw frequency
50-draw frequency
100-draw frequency
Gap
current gap
average gap
median gap
maximum gap
gap standard deviation
Pairs

Calculate:

number pair frequency
pair recency
pair strength
Triples

Calculate:

triple frequency
triple recency
triple strength
Distribution

Calculate:

sum
mean
median
range
odd/even
low/high
prime/composite
consecutive runs
last digits
number zones
21. Feature Engineering

Three feature levels.

Number-level

For each number:

frequency_5
frequency_10
frequency_20
frequency_50
frequency_100


gap_current
gap_mean
gap_std
gap_max


recent_frequency


pair_strength
triple_strength
Draw-level
sum
mean
median
range
odd_count
even_count
low_count
high_count
prime_count
consecutive_count
last_digit_distribution
Ticket-level
sum_score
odd_even_score
low_high_score
frequency_score
gap_score
pair_score
triple_score
diversity_score
recent_overlap
ensemble_score

All feature calculations must be deterministic and testable.

22. Prediction Architecture

Do NOT ask an LLM:

"Give me six winning numbers."

Instead:

Historical data
      ↓
Feature engineering
      ↓
Number scoring
      ↓
Candidate generation
      ↓
Ticket scoring
      ↓
Portfolio optimization
      ↓
Final tickets
23. Number Scoring

For LOTO6:

numbers = 1..43

For Mini Loto:

numbers = 1..31

Each number receives a model score.

Example:

Number    Score


03        0.81
08        0.79
15        0.77
24        0.76
...

The score is a ranking value.

It is NOT automatically a probability of winning.

24. Candidate Generation

Do not generate only two tickets.

Generate a large candidate pool:

10,000
50,000
100,000
500,000

depending on performance.

Pipeline:

Candidate pool
      ↓
Validate
      ↓
Remove duplicates
      ↓
Score
      ↓
Rank
      ↓
Optimize

Candidate generation must be reproducible using a stored random seed.

25. Monte Carlo

Use Monte Carlo simulation for:

candidate generation
baseline estimation
distribution analysis
stress testing

Monte Carlo must not be presented as proof that future draws are predictable.

Record:

random_seed
simulation_count
configuration
timestamp
26. Machine Learning Models

Start simple.

First models
Random Forest
Extra Trees
Logistic Regression
XGBoost
LightGBM
CatBoost

Only later:

Neural Network
LSTM
GRU
Transformer

Complexity must be justified by backtesting.

27. Model Registry

Every model needs a version.

Example:

xgboost-loto6-v001
xgboost-loto6-v002


lightgbm-loto6-v001


catboost-mini-loto-v001

Record:

model_id
model_version
lottery
dataset_version
feature_version
hyperparameters
training_date
evaluation_metrics
random_seed
git_commit
28. Ensemble

Multiple models produce scores.

Example:

Random Forest
XGBoost
LightGBM
CatBoost
Neural Network

Then:

Model scores
      ↓
Calibration
      ↓
Weighted ensemble
      ↓
Final number score

Weights must be evaluated using historical validation.

Do not assume equal weights are optimal.

29. Baseline Strategies

The system MUST contain baselines.

Baseline 1 — Pure Random

Uniform random ticket generation.

Baseline 2 — Balanced Random

Random tickets with basic distribution constraints.

Baseline 3 — Frequency

Historical frequency weighted selection.

Baseline 4 — Gap

Gap-based heuristic.

Baseline 5 — Hot/Cold

Hot/cold heuristic.

Baseline 6 — AI Ensemble

Production strategy.

The AI strategy must be compared against these baselines.

30. Backtesting

Backtesting is more important than model complexity.

For every predicted ticket store:

prediction_date
target_draw
lottery
model_version
feature_version
ticket
matched_numbers
bonus_match
prize_category

Metrics:

total tickets
average matches
3-match count
4-match count
5-match count
6-match count
prize count
total cost
total payout
net result
ROI
31. Match-Based Evaluation

Do not only measure money.

Track:

average matched numbers
distribution of match counts
3+ match rate
4+ match rate
5+ match rate
prize rate

Then separately track:

cost
payout
ROI

This prevents prize jackpot fluctuations from completely dominating evaluation.

32. Statistical Evaluation

Compare:

AI
vs
Random

Use:

bootstrap confidence intervals
effect size
distribution comparisons
statistical tests

The goal is to determine whether any observed advantage is likely to be meaningful rather than random variation.

33. Two-Ticket Optimization

The final two tickets should be selected as a portfolio.

Do not simply select:

top #1
top #2

because they may be almost identical.

Example:

Ticket A:
03 08 15 24 31 42


Ticket B:
03 08 15 24 31 39

Very high overlap.

Instead optimize:

individual score
+
portfolio coverage
+
diversity
-
excessive overlap
34. Portfolio Objective

Conceptually:

Portfolio Score =


Ticket A score
+
Ticket B score
+
Coverage bonus
-
Overlap penalty

The exact formula must be experimentally evaluated.

35. Weekly Production

Every week:

1. Get latest official draw data
2. Validate data
3. Update database
4. Calculate features
5. Check model health
6. Generate candidate pool
7. Score candidates
8. Optimize portfolio
9. Generate 2 LOTO6 tickets
10. Generate 2 Mini Loto tickets
11. Save predictions
12. Generate report
36. Weekly Output

Example:

================================
LotoSystem Weekly Prediction
================================


LOTO6


Ticket 1:
03 08 15 24 31 42


Ticket 2:
06 11 17 25 34 39




MINI LOTO


Ticket 1:
04 09 16 22 28


Ticket 2:
07 12 19 25 30

Additional metadata:

Model:
ensemble-v004


Feature version:
features-v007


Candidate pool:
100,000


Random seed:
123456


Generated:
YYYY-MM-DD
37. Prediction Storage

Every prediction must be permanently stored.

Store:

run_id
lottery
target_draw
prediction_date
ticket_number
ticket
model_version
feature_version
candidate_pool_size
random_seed
ensemble_score
configuration_version
git_commit

After the actual draw:

actual_main_numbers
actual_bonus
matched_numbers
bonus_match
prize_category
payout
38. Future Evaluation

The system must automatically evaluate old predictions when new results arrive.

Example:

Prediction:
2026-08-06


Actual:
2026-08-06


      ↓


Match
      ↓
Calculate prize
      ↓
Update metrics
      ↓
AI vs random
39. Experiment System

Every research idea must become an experiment.

Example:

Experiment ID:
EXP-0001


Hypothesis:
20-draw frequency contains useful signal.


Lottery:
LOTO6


Features:
frequency_10
frequency_20
frequency_50


Model:
XGBoost


Baseline:
Random


Evaluation:
Walk-forward


Result:
...

Store:

hypothesis
dataset
features
model
parameters
baseline
evaluation period
metrics
result
conclusion
40. Research Agent

The Research Agent can analyze:

historical statistics
model performance
feature importance
recent model behavior
experiment results

It can propose:

new experiments
new features
new models

But proposed ideas must be tested by the deterministic experiment system.

41. Explanation Agent

The Explanation Agent can produce:

Why did the model rank these numbers highly?


Which features influenced the result?


How does this week's prediction differ from last week's?


What does the backtest show?

It must never invent evidence.

42. Report Agent

Generate weekly reports containing:

latest data status
model status
LOTO6 tickets
Mini Loto tickets
model scores
backtest performance
AI vs random
recent performance
experiment results
warnings
43. Manager Agent

High-level workflow:

Manager
   │
   ├── Data Agent
   ├── Statistics Agent
   ├── Feature Agent
   ├── Model Agent
   ├── Simulation Agent
   ├── Backtest Agent
   ├── Experiment Agent
   ├── Explanation Agent
   └── Report Agent

The Manager coordinates.

It should not contain the actual mathematical implementation of each component.

44. AI Provider Switching

Example:

LOTO_SYSTEM
     │
     ▼
LLMProvider
     │
     ├── Ollama
     │
     └── OpenAI

Ollama:

LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b

OpenAI:

LLM_PROVIDER=openai
LLM_MODEL=<model>
OPENAI_API_KEY=<secret>

The application should continue working with the same agent interfaces.

45. API

Initial API:

GET  /api/health


GET  /api/lotteries


GET  /api/draws/{lottery}


GET  /api/draws/{lottery}/latest


GET  /api/statistics/{lottery}


GET  /api/predictions/latest


GET  /api/predictions/history


POST /api/predictions/generate


GET  /api/backtests


POST /api/backtests/run


GET  /api/models


GET  /api/experiments


POST /api/experiments


GET  /api/reports/latest

Routes must call services.

Do not put ML logic directly inside API routes.

46. Service Layer

Architecture:

API
 ↓
Service
 ↓
Domain/Engine
 ↓
Repository
 ↓
Database

Services:

DrawService
PredictionService
BacktestService
ExperimentService
ReportService
47. Dashboard

Eventually show:

System Health


Latest Draw


Next Prediction


LOTO6
├── Ticket 1
└── Ticket 2


Mini Loto
├── Ticket 1
└── Ticket 2


AI Performance


Random Performance


3-match count
4-match count
5-match count
6-match count


Prize rate


ROI


Model leaderboard


Feature importance


Experiment leaderboard


Data health


Model health
48. Model Health

Before production prediction:

Data health
Feature health
Model health
Backtest health
Drift health

If the current model is degraded:

Do not blindly generate production predictions.


Consider fallback to best validated model.
49. Model Drift

Track model performance over time.

Example:

2024:
AI > Random


2025:
AI > Random


2026:
AI ≈ Random

This should trigger an investigation.

Potential causes:

model drift
feature drift
implementation changes
data problems
random variation
50. Reproducibility

Every experiment and production run must record:

dataset version
feature version
model version
configuration version
random seed
code commit
timestamp

A production prediction should be reproducible.

51. Security

Never commit:

API keys
database passwords
tokens
credentials
private configuration

Use:

.env

and:

.env.example

The .env file must be ignored by Git.

52. Testing
Unit Tests

Test:

LotteryDefinition
Ticket
Draw
Prize calculation
Match calculation
Bonus calculation
Number validation
Feature calculations
Candidate generation
Portfolio optimization
Integration Tests

Test:

collector → database


database → features


features → model


model → candidate generation


candidate generation → optimizer


optimizer → backtest
53. Regression Tests

Create known test cases.

Example:

Winning numbers:
03 08 15 24 31 42


Bonus:
19


Ticket:
03 08 15 24 31 42


Expected:
6 matches
Top prize category

Add edge cases:

0 matches
1 match
2 matches
3 matches
4 matches
5 matches
5 + bonus
6 matches

Use official rules to determine exact prize classifications.

54. Code Quality

Recommended:

pytest
ruff
mypy
pre-commit

Every significant feature must have tests.

Avoid unnecessary abstractions.

Prefer:

small
clear
testable

over:

complex
clever
hard to maintain
55. Technology Stack

Recommended:

Backend
Python 3.12+
FastAPI
Pydantic
SQLAlchemy
Alembic
Data
Polars
Pandas
NumPy
SciPy
Machine Learning
scikit-learn
XGBoost
LightGBM
CatBoost
Deep Learning
PyTorch
Database
PostgreSQL
Cache
Redis

if required.

Scheduler
APScheduler

initially.

LLM
Ollama
OpenAI API
56. Docker

Development can use:

FastAPI
PostgreSQL
Redis
Ollama
Frontend

Production can later separate:

API
Worker
Scheduler
Database
Redis
Ollama
Frontend
57. Development Phases
Phase 1 — Repository Foundation

Create:

AGENTS.md
README.md
fullblueprint.md
pyproject.toml
.env.example
.gitignore
configuration
logging
basic FastAPI application
pytest setup
ruff setup
directory structure

Do NOT implement prediction yet.

Phase 2 — Domain

Implement:

LotteryDefinition
Draw
Ticket
Prize
Prediction
Experiment
Phase 3 — Database

Implement:

PostgreSQL
SQLAlchemy
Alembic
repositories
migrations
Phase 4 — Historical Data

Implement:

LOTO6 collector
Mini Loto collector
source tracking
validation
historical import
Phase 5 — Prize Engine

Implement:

matching
bonus
prize classification

Test heavily.

Phase 6 — Statistics

Implement:

frequency
gap
pairs
triples
distribution
correlation
Phase 7 — Backtesting Foundation

Implement:

random ticket generator
random baseline
walk-forward engine
metrics
leakage tests
Phase 8 — Features

Implement:

number features
draw features
ticket features
feature versioning
Phase 9 — Machine Learning

Implement:

Random Forest
XGBoost
LightGBM
CatBoost
Phase 10 — Ensemble

Implement:

model scoring
model weighting
calibration
ensemble
Phase 11 — Optimization

Implement:

candidate generation
Monte Carlo
ticket scoring
portfolio optimization
2-ticket selection
Phase 12 — Advanced ML

Only if justified:

Neural Network
LSTM
GRU
Transformer
Phase 13 — LLM

Implement:

LLM interface
provider factory
Ollama
OpenAI
Research Agent
Experiment Agent
Explanation Agent
Report Agent
Phase 14 — Automation

Implement:

scheduler
weekly pipeline
post-draw evaluation
prediction persistence
Phase 15 — API/UI

Implement:

FastAPI endpoints
dashboard
reports
model monitoring
Phase 16 — Production

Implement:

Docker
monitoring
backup
security
documentation
58. First Meaningful Milestone

The first meaningful milestone is NOT AI.

It is:

Historical data
       ↓
Database
       ↓
Random ticket generator
       ↓
Prize engine
       ↓
Backtesting engine
       ↓
Random baseline report

The system should answer:

If we had generated two random LOTO6 tickets before every historical draw, how many 3rd, 4th, and 5th prize results would we have achieved?

And similarly for Mini Loto.

59. Second Milestone

Add:

Frequency strategy
Gap strategy
Pair strategy

Compare:

Random
vs
Frequency
vs
Gap
vs
Pair
60. Third Milestone

Add:

Random Forest
XGBoost
LightGBM
CatBoost

Compare all against:

Random
Frequency
Gap
Pair
61. Fourth Milestone

Add:

Ensemble
Candidate generation
Portfolio optimization

Evaluate whether the ensemble provides measurable improvement.

62. Fifth Milestone

Add:

Ollama
Research Agent
Experiment Agent
Explanation Agent
Report Agent
63. Sixth Milestone

Add:

OpenAI provider

Run the same research workflows using OpenAI without changing the core prediction system.

64. Weekly Production Pipeline
Latest official result
        ↓
Data validation
        ↓
Database update
        ↓
Feature update
        ↓
Model health check
        ↓
Candidate generation
        ↓
Model scoring
        ↓
Ensemble
        ↓
Portfolio optimization
        ↓
2 LOTO6 tickets
        ↓
2 Mini Loto tickets
        ↓
Save prediction
        ↓
Generate report

After the actual draw:

Actual result
       ↓
Match predictions
       ↓
Calculate prize
       ↓
Update metrics
       ↓
Compare AI vs random
       ↓
Update research database
65. Weekly Cost Tracking

Current intended strategy:

2 LOTO6
+
2 Mini Loto

At ¥200 per ticket:

4 × ¥200 = ¥800

Track:

weekly cost
total cost
prize amount
net result
ROI

Do not use ROI alone as the model quality metric.

66. No Automatic Purchase

The application may:

generate predictions
analyze data
generate reports
evaluate results

It must NOT automatically purchase lottery tickets.

Any purchase remains a manual user decision.

67. Prediction Score

Never display unsupported claims like:

Winning probability = 87%

Instead:

Model score = 87.0

Explain:

This is a relative model ranking score, not a guaranteed probability of winning.

68. AI vs Random Dashboard

The dashboard should eventually provide:

                 AI       Random


Average matches  1.02      0.98


3+ matches       4.2%      3.9%


4+ matches       0.8%      0.7%


Prize rate       X%        Y%


ROI              X%        Y%

Actual values must come from the backtesting database.

Never hard-code performance claims.

69. Long-Term Research

Possible future experiments:

Bayesian models
Graph neural networks
Number co-occurrence networks
Change-point detection
Regime detection
Temporal models
Genetic algorithms
Evolutionary optimization
Reinforcement learning
Distribution shift detection
Model stacking

These are research directions, not assumptions.

70. Critical Anti-Bias Rules

Never assume:

hot numbers must appear again

Never assume:

cold numbers are due

Never assume:

a number is "due"

Never assume:

a chart pattern means future predictability

Never assume:

AI score = probability

Never tune a model repeatedly against the final test set.

Never select a model solely because it performed well over a short historical period.

71. Model Selection

The best model is not necessarily:

the most complicated model

The best model is:

the model with the strongest
reproducible out-of-sample evidence

Potential outcome:

Random:
baseline


XGBoost:
slightly better


Transformer:
same as random

In this case:

Use XGBoost

not Transformer simply because Transformer is more advanced.

72. Research Logging

Every experiment must have a permanent record.

Example:

EXP-0027


Hypothesis:
Recent pair frequency may improve candidate ranking.


Dataset:
LOTO6 historical draws


Training:
2000–2024


Validation:
2025


Test:
2026


Features:
pair_frequency_10
pair_frequency_20
pair_frequency_50


Model:
LightGBM


Baseline:
Random


Result:
...


Conclusion:
...
73. Feature Importance

For models that support feature importance, record:

feature
importance
model
training period

Example:

frequency_20       0.17
gap_current        0.12
pair_strength      0.09
sum_score          0.07
...

But feature importance does NOT prove causal predictability.

74. Model Explainability

When possible, use:

SHAP
permutation importance
feature importance

to understand why a model ranked candidates.

Do not confuse explainability with prediction validity.

75. Data Versioning

Historical data should be versioned.

Example:

dataset-v001
dataset-v002
dataset-v003

When historical data changes, record:

old version
new version
reason
source
timestamp
76. Feature Versioning

Example:

features-v001
features-v002
features-v003

A prediction must record the feature version used.

77. Configuration Versioning

Production runs should record:

configuration version

This includes:

model weights
candidate pool size
optimizer parameters
baseline settings
feature settings
78. Random Seed

All stochastic algorithms should accept a random seed.

Example:

seed = 123456

Store the seed with the experiment/prediction.

79. Reproducible Prediction

Given:

same dataset
same features
same model
same configuration
same seed
same code version

the system should reproduce the same prediction.

80. Error Handling

If an important dependency fails:

Do not silently continue.

Examples:

database unavailable
historical data invalid
model missing
feature calculation failed
LLM unavailable

LLM failure should not prevent deterministic prediction.

81. LLM Failure Strategy

If Ollama is unavailable:

Prediction engine continues.

If OpenAI is unavailable:

Prediction engine continues.

LLM is optional.

The core system must remain functional without an LLM.

82. Scheduler

Jobs:

update_draw_data
calculate_statistics
update_features
model_health_check
run_prediction
generate_report
evaluate_previous_prediction

Scheduler must avoid duplicate runs.

Each run should have a unique run ID.

83. Run IDs

Example:

RUN-2026-08-20-LOTO6-001

or UUID.

Every major operation should have a traceable run ID.

84. Logging

Logs should include:

timestamp
level
component
run_id
message
error

Example:

2026-08-20 12:00:00 INFO
PredictionService
RUN-001
Candidate generation started
85. Monitoring

Monitor:

database
collector
data freshness
feature generation
model status
prediction generation
scheduler
API
LLM provider
disk usage
86. Backup

Back up:

PostgreSQL
prediction history
experiment history
configuration
model metadata

Do not rely only on raw downloaded data.

87. Documentation

Maintain:

README.md


docs/
├── architecture.md
├── data-model.md
├── prediction-system.md
├── backtesting.md
├── ai-providers.md
├── experiments.md
└── weekly-operation.md

fullblueprint.md remains the high-level master blueprint.

88. Git Strategy

Use incremental commits.

Example:

stage 01: initialize repository


stage 02: add domain models


stage 03: add database


stage 04: add historical data pipeline


stage 05: add data validation


stage 06: add prize engine


stage 07: add statistics


stage 08: add feature engineering


stage 09: add random baseline


stage 10: add backtesting


stage 11: add ML models


stage 12: add ensemble


stage 13: add ticket optimizer


stage 14: add LLM providers


stage 15: add agents


stage 16: add scheduler


stage 17: add API


stage 18: add dashboard

Each stage should be independently testable.

89. Codex Rules

Codex is the primary coding tool for this project.

Codex must:

Read AGENTS.md.
Read relevant documentation.
Inspect existing code.
Implement only the requested task.
Add tests.
Run tests.
Run lint/type checks.
Report changed files.
Report test results.
Stop after the requested scope.

Do not ask Codex to implement the entire blueprint in one prompt.

90. No Unrelated Refactoring

If a task is:

Add ticket domain model.

Codex must not also:

rewrite database
rewrite API
add ML
add frontend
change Docker

unless explicitly requested.

91. Definition of Done

A task is complete only when:

[ ] Implementation complete
[ ] Tests added
[ ] Existing tests still pass
[ ] Lint passes
[ ] Type checking passes where applicable
[ ] Documentation updated where necessary
[ ] No secrets committed
[ ] No unrelated files changed
92. First Coding Stage

The first Codex task must create only the foundation.

Create:

AGENTS.md
README.md
fullblueprint.md
pyproject.toml
.env.example
.gitignore


basic configuration
logging
FastAPI application
pytest setup
ruff configuration
initial folders
architecture documentation

Do NOT implement:

ML
prediction
LLM agents
ticket optimizer
web scraping
scheduler
deep learning
93. First Coding Validation

After Stage 1:

pytest

must run successfully.

The API should start.

Health endpoint should work:

GET /api/health

Expected concept:

{
  "status": "ok"
}
94. Second Coding Stage

Implement domain models:

LotteryDefinition
Draw
Ticket
Prediction
Prize
Experiment

Then write unit tests.

95. Third Coding Stage

Implement database:

PostgreSQL
SQLAlchemy
Alembic
repositories

Create migrations.

96. Fourth Coding Stage

Implement historical collectors.

Priority:

LOTO6
Mini Loto

Collector must be testable without requiring the real website during unit tests.

Use mocked/fixture data.

97. Fifth Coding Stage

Implement the prize/matching engine before ML.

This is mandatory.

If the system cannot correctly determine:

3 matches
4 matches
5 matches
5 + bonus
6 matches

then ML evaluation cannot be trusted.

98. Sixth Coding Stage

Implement random baseline.

For example:

100,000 random tickets

against historical draws.

Generate baseline statistics.

This becomes the benchmark for all future AI strategies.

99. Seventh Coding Stage

Implement walk-forward backtesting.

This is the foundation for determining whether the AI system is actually useful.

100. Eighth Coding Stage

Implement features.

Features must be generated using only historical data available at prediction time.

101. Ninth Coding Stage

Implement ML models.

Start with:

Random Forest
XGBoost
LightGBM
CatBoost

Do not immediately implement Transformers.

102. Tenth Coding Stage

Implement ensemble.

Compare:

individual models
vs
ensemble
103. Eleventh Coding Stage

Implement candidate generation and portfolio optimization.

The optimizer produces:

2 LOTO6 tickets
2 Mini Loto tickets
104. Twelfth Coding Stage

Implement LLM providers.

Start:

Ollama

Then:

OpenAI
105. Thirteenth Coding Stage

Implement agents.

Start:

Research Agent
Experiment Agent
Explanation Agent
Report Agent

Manager Agent comes after the individual agents are stable.

106. Fourteenth Coding Stage

Implement weekly automation.

Data update
↓
Features
↓
Model
↓
Candidates
↓
Optimization
↓
Predictions
↓
Report
107. Final Architecture
                         USER
                           │
                           ▼
                     WEB DASHBOARD
                           │
                           ▼
                      FASTAPI API
                           │
                           ▼
                    SERVICE LAYER
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   DATA SYSTEM       ML SYSTEM          AI SYSTEM
        │                  │                  │
        ▼                  ▼                  ▼
   Collectors          Features          Agents
        │                  │                  │
        ▼                  ▼                  ▼
   PostgreSQL        ML Models            LLM
                           │                  │
                           ▼                  │
                      Ensemble ◄──────────────┘
                           │
                           ▼
                    Candidate Generator
                           │
                           ▼
                    Ticket Optimizer
                           │
                           ▼
                   Portfolio Selection
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  LOTO6       MINI LOTO
                    │             │
                  2 tickets     2 tickets
                    │             │
                    └──────┬──────┘
                           ▼
                     BACKTEST ENGINE
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                   AI          RANDOM
                    │             │
                    └──────┬──────┘
                           ▼
                  STATISTICAL TESTS
                           │
                           ▼
                   WEEKLY REPORT
                           │
                           ▼
                    FUTURE RESULTS
                           │
                           └──────► CONTINUOUS RESEARCH
108. Final Research Principle

The system should never be judged by whether one or two tickets happen to win.

A single winning ticket can be luck.

A single losing week proves nothing.

The important question is:

Over a large number of unseen historical draws,


does the strategy consistently perform differently
from properly constructed random baselines?

That is the central scientific objective of LotoSystem.

109. Final Project Goal

The final system should automatically perform:

EVERY WEEK


             ↓


Collect latest results


             ↓


Validate data


             ↓


Update database


             ↓


Calculate statistics


             ↓


Generate features


             ↓


Evaluate model health


             ↓


Run validated models


             ↓


Generate large candidate pool


             ↓


Score candidates


             ↓


Optimize portfolio


             ↓


Select:


2 × LOTO6


2 × Mini Loto


             ↓


Save prediction


             ↓


Generate weekly report


             ↓


Wait for actual draw


             ↓


Evaluate prediction


             ↓


Compare AI vs Random


             ↓


Update long-term research


             ↓


Improve next experiment

The project must remain evidence-driven.

The objective is not to promise winning numbers.

The objective is to build a serious experimental system capable of discovering whether any measurable predictive signal exists in the historical data.