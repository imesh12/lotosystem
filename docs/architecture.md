# Architecture

`fullblueprint.md` is the master blueprint for LotoSystem. This document records the Stage 01 foundation only.

## Implemented Scope

The codebase provides a minimal FastAPI backend with configuration, logging, and a health endpoint. Stage 03 adds hardened historical-data validation, deterministic dataset hashing, prize matching, seeded random baselines, and stronger backtesting.

```text
backend.app.main
  -> backend.app.core.config
  -> backend.app.core.logging
  -> backend.app.api.routes.health
  -> backend.app.domain
  -> backend.app.research
```

## Application Layers

- `api`: HTTP routes and dependencies.
- `core`: cross-cutting concerns such as configuration, logging, and shared exceptions.
- `domain`: lottery rules and core domain objects.
- `research`: deterministic loading, validation, collectors, dataset hashing, statistics, features, candidates, scoring, prize matching, random baselines, backtesting, persistence, and CLI.
- `services`: future application services.
- `repositories`: future persistence adapters.

## Deferred Work

Databases, official web collectors, machine learning, LLM providers, agents, scheduling, frontend work, and automatic purchasing are intentionally deferred to later stages.
