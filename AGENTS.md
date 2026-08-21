# AGENTS.md

Guidance for Codex and other coding agents working on LotoSystem.

## Source Of Truth

- Read `fullblueprint.md` before making architectural changes.
- Implement only the stage requested by the user.
- Keep changes small, testable, and aligned with the existing structure.

## Current Stage

Stage 04: Authoritative Historical Data Import.

Allowed through Stage 04:

- Repository structure
- Python project configuration
- FastAPI application shell
- Configuration system
- Logging setup
- Health endpoint
- pytest setup
- Ruff setup
- Environment examples
- Basic documentation
- Lottery rule definitions
- Core domain models
- Domain validation tests
- Historical draw loading and validation
- Deterministic statistics
- Deterministic feature generation
- Deterministic research candidate generation
- Transparent research scoring
- Walk-forward backtesting/evaluation
- JSON research result persistence
- CLI for offline research commands
- CSV collector abstraction
- Dataset hashing and provenance support
- Prize/match classification foundation
- Seeded uniform-random baseline replications
- Evaluation date ranges
- Strategy-vs-random comparison
- Authoritative Mizuho historical collector
- Canonical historical CSV exports
- Incremental historical update and conflict detection

Not allowed in this stage:

- Claims that research candidates are predictions
- Machine learning
- Ticket optimization
- Arbitrary third-party historical scraping
- Database integration
- Scheduler
- LLM providers or agents
- Frontend dashboard
- Automatic ticket purchasing

## Development Rules

- Use Python 3.12+ and modern typing.
- Prefer simple modules over premature abstractions.
- Keep API routes thin and move behavior into services when behavior exists.
- Never commit secrets. Use `.env` locally and `.env.example` for documented settings.
- Add or update tests for meaningful behavior.
- Run `pytest` and `ruff check .` before reporting completion when tools are available.

## Scientific Rules

- Treat lottery draws as random unless evidence proves otherwise.
- Never claim guaranteed winning numbers.
- Never present model scores as winning probabilities.
- Protect future work from data leakage: predictions for a draw may only use information available before that draw.
- The deterministic research engine must run without an LLM.
- Stage 04 network access belongs only to the authoritative history import path.
