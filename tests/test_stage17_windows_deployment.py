import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SCRIPTS = ROOT / "scripts" / "windows"


def _script(name: str) -> str:
    return (WINDOWS_SCRIPTS / name).read_text(encoding="utf-8")


def test_stage17_required_windows_scripts_exist() -> None:
    expected = {
        "common.ps1",
        "start-server.ps1",
        "run-auto-cycle.ps1",
        "install-scheduled-tasks.ps1",
        "uninstall-scheduled-tasks.ps1",
        "status.ps1",
        "backup-data.ps1",
    }

    assert expected <= {path.name for path in WINDOWS_SCRIPTS.glob("*.ps1")}


def test_scripts_resolve_repo_relative_to_script_location() -> None:
    common = _script("common.ps1")

    assert 'Join-Path $PSScriptRoot "..\\.."' in common
    for path in WINDOWS_SCRIPTS.glob("*.ps1"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users\\imcom\\Downloads\\loto-system" not in text


def test_server_script_is_local_only_and_uses_existing_fastapi_app() -> None:
    text = _script("start-server.ps1")

    assert '[string] $HostName = "127.0.0.1"' in text
    assert "[int] $Port = 8000" in text
    assert "backend.app.main:app" in text
    assert "/api/health" in _script("common.ps1")
    assert "server-*.log" in text


def test_auto_cycle_wrapper_calls_python_source_of_truth() -> None:
    text = _script("run-auto-cycle.ps1")

    assert "backend.app.research.cli" in text
    assert '"--lottery", $Lottery' in text
    assert '"auto-run"' in text
    assert "automation-*.log" in text
    assert "Register-ScheduledTask" not in text


def test_status_script_reports_all_lottery_automation() -> None:
    text = _script("status.ps1")

    assert "backend.app.research.cli --lottery ALL automation-status" in text
    assert "notification-status" in text


def test_task_installer_is_idempotent_and_uses_stable_task_names() -> None:
    text = _script("install-scheduled-tasks.ps1")

    assert '"LotoSystem-AutoRun"' in text
    assert '"LotoSystem-Web"' in text
    assert "Register-ScheduledTask" in text
    assert "-Force" in text
    assert "New-ScheduledTaskTrigger -AtLogOn -User $currentUser" in text
    assert "New-TimeSpan -Hours $AutoRunIntervalHours" in text
    assert "New-ScheduledTaskTrigger `" in text
    assert "-Once" in text
    assert "-RepetitionInterval (New-TimeSpan -Hours $AutoRunIntervalHours)" in text


def test_task_installer_does_not_use_invalid_gigantic_repetition_duration() -> None:
    text = _script("install-scheduled-tasks.ps1")

    assert "[TimeSpan]::MaxValue" not in text
    assert "-RepetitionDuration" not in text
    assert "P99999999DT23H59M59S" not in text


def test_uninstall_script_only_removes_lotosystem_tasks() -> None:
    text = _script("uninstall-scheduled-tasks.ps1")

    assert "Unregister-ScheduledTask" in text
    assert "Remove-Item" not in text
    assert '"LotoSystem-AutoRun"' in text
    assert '"LotoSystem-Web"' in text


def test_backup_script_includes_operational_state_and_excludes_logs() -> None:
    text = _script("backup-data.ps1")

    for item in (
        "config\\operational_settings.json",
        "data\\processed",
        "data\\predictions",
        "data\\settlements",
        "data\\notifications",
        "data\\automation",
    ):
        assert item in text
    assert "data\\logs" not in text
    assert ".venv" not in text
    assert "LOTO_SMTP_PASSWORD" not in text


def test_version_marker_and_fastapi_version_are_v1() -> None:
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "1.0.0"
    assert 'version = "1.0.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version="1.0.0"' in (ROOT / "backend/app/main.py").read_text(encoding="utf-8")


def test_runbook_documents_recovery_and_scheduler_commands() -> None:
    text = (ROOT / "docs" / "version-1-runbook.md").read_text(encoding="utf-8")

    assert "Version: 1.0.0" in text
    assert "install-scheduled-tasks.ps1" in text
    assert "uninstall-scheduled-tasks.ps1" in text
    assert "PC was offline for several days" in text
    assert "does not fabricate retroactive predictions" in text


def test_powershell_scripts_parse_when_powershell_is_available() -> None:
    if not Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").exists():
        return
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$ErrorActionPreference='Stop';"
            "Get-ChildItem scripts/windows/*.ps1 | "
            "ForEach-Object { $null = [scriptblock]::Create((Get-Content $_ -Raw)) }"
        ),
    ]

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
