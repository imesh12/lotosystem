from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"

router = APIRouter(tags=["frontend"])


@router.get("/", response_class=HTMLResponse)
def dashboard_page() -> HTMLResponse:
    return _html_response("dashboard.html")


@router.get("/settings", response_class=HTMLResponse)
def settings_page() -> HTMLResponse:
    return _html_response("settings.html")


def _html_response(filename: str) -> HTMLResponse:
    return HTMLResponse((FRONTEND_ROOT / filename).read_text(encoding="utf-8"))
