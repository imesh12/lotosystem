# LotoSystem Version 1 Runbook

Version: 1.0.0

LotoSystem Version 1 is a local Windows paper-trading system. It updates official lottery results, evaluates saved predictions, records paper settlements, generates the next prediction, and serves a local dashboard. It does not purchase tickets, move money, or claim guaranteed winning numbers.

## First-Time Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m playwright install chromium
```

Copy `.env.example` to `.env` only when local overrides are needed. Never commit `.env`.

## Start Manually

Start the local FastAPI/frontend server:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/start-server.ps1
```

Defaults:

- host: `127.0.0.1`
- port: `8000`
- dashboard: `http://127.0.0.1:8000/`
- settings: `http://127.0.0.1:8000/settings`
- health: `http://127.0.0.1:8000/api/health`

The server is local-only by default and writes compact logs under `data/logs/`.

## Automation Behavior

Run one automation pass manually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run-auto-cycle.ps1
```

This calls:

```powershell
python -m backend.app.research.cli --lottery ALL --seed 123456 --tickets-per-draw 3 auto-run
```

Python remains the source of truth. The PowerShell wrapper only sets the working directory, uses `.venv`, and records logs.

## Windows Scheduler

Install scheduled tasks explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/install-scheduled-tasks.ps1
```

Created tasks:

- `LotoSystem-AutoRun`: wakes every 3 hours and runs the Python one-shot automation.
- `LotoSystem-Web`: starts the local API/frontend at user logon.

The 3-hour wake-up is intentionally modest. Stage 13 automation decides whether work is due and returns `NO_ACTION` when nothing should run.

Uninstall only LotoSystem scheduled tasks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/uninstall-scheduled-tasks.ps1
```

Uninstalling tasks does not delete history, predictions, settings, settlements, notifications, automation records, or backups.

## Status Check

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/status.ps1
```

The status script reports repository path, `.venv` availability, API reachability, automation status, notification status, scheduled task presence, and the latest automation run.

## Email Setup

Email is optional and disabled by default. Configure only through environment variables:

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

SMTP passwords are never stored in operational settings or notification JSON. If email is disabled, the lifecycle still works normally.

## Backup

Create a local timestamped archive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/backup-data.ps1
```

The backup includes:

- `config/operational_settings.json`
- `data/processed/`
- `data/predictions/`
- `data/settlements/`
- `data/notifications/`
- `data/automation/`

It excludes `.venv`, caches, logs, and SMTP secrets.

## Restore Concept

Stop the server, unpack a trusted backup into the repository root, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/status.ps1
```

If the PC was offline for several days, the next `auto-run` catches up official results, evaluates only predictions that already existed before their target draw, does not fabricate retroactive predictions, and creates only the next valid future prediction.

## Update From GitHub

Before updating, create a backup. Then pull the latest code, reinstall dependencies if needed, and run:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

## Troubleshooting

- Missing `.venv`: rerun setup.
- API not reachable: run `scripts/windows/start-server.ps1` and check `data/logs/server-*.log`.
- Automation source failure: run `automation-status`; if all automated sources fail, use the validated manual result command from the README.
- Email failure: lifecycle state remains valid; run `notification-status` and `send-pending-notifications` after fixing SMTP settings.
- Active automation lock: wait for the current run. Stale locks are recovered by Stage 13 automation.
