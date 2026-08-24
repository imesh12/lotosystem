from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_dashboard_page_returns_200(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "LotoSystem Dashboard" in response.text


def test_settings_page_returns_200(client: TestClient) -> None:
    response = client.get("/settings")

    assert response.status_code == 200
    assert "LotoSystem Settings" in response.text


def test_dashboard_renders_lottery_shells_and_financial_sections(client: TestClient) -> None:
    response = client.get("/")

    assert "LOTO6" in response.text
    assert "Mini Loto" in response.text
    assert "Financial Summary" in response.text
    assert "Paper Trading / Simulated" in response.text


def test_dashboard_javascript_handles_operational_states() -> None:
    script = Path("backend/app/frontend/assets/dashboard.js").read_text(encoding="utf-8")

    assert "Official Result" in script
    assert "Next Prediction" in script
    assert "No saved prediction for this completed draw" in script
    assert "PAYOUT_PENDING" in script
    assert "Notification failures" in script


def test_settings_defaults_and_future_only_note(client: TestClient) -> None:
    response = client.get("/settings")

    assert "Number of sets per draw" in response.text
    assert "Changes affect future predictions only" in response.text
    assert 'min="1" max="20"' in response.text


def test_settings_update_from_ui_api(
    monkeypatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("backend.app.api.routes.operations.API_SETTINGS_PATH", settings_path)

    response = client.put(
        "/api/settings",
        json={
            "LOTO6": {"enabled": True, "tickets_per_draw": 5},
            "MINI_LOTO": {"enabled": False, "tickets_per_draw": 3},
            "email_enabled": True,
        },
    )
    saved = client.get("/api/settings")

    assert response.status_code == 200
    assert saved.json()["LOTO6"]["tickets_per_draw"] == 5
    assert saved.json()["MINI_LOTO"]["enabled"] is False
    assert saved.json()["email_enabled"] is True


def test_invalid_ticket_count_rejected_by_api(client: TestClient) -> None:
    response = client.put("/api/settings", json={"LOTO6": {"tickets_per_draw": 21}})

    assert response.status_code == 400


def test_lottery_disabled_state_supported_in_settings_page(client: TestClient) -> None:
    response = client.get("/settings")

    assert "loto6-enabled" in response.text
    assert "mini-loto-enabled" in response.text
    assert "Email notifications enabled" in response.text


def test_frontend_does_not_render_secrets_or_direct_data_files(client: TestClient) -> None:
    dashboard = client.get("/").text
    settings = client.get("/settings").text
    assets = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "backend/app/frontend/assets/dashboard.js",
            "backend/app/frontend/assets/settings.js",
        )
    )
    combined = dashboard + settings + assets

    assert "LOTO_SMTP_PASSWORD" not in combined
    assert "data/processed" not in combined
    assert ".csv" not in combined


def test_static_assets_return_200(client: TestClient) -> None:
    css = client.get("/assets/app.css")
    dashboard_js = client.get("/assets/dashboard.js")
    settings_js = client.get("/assets/settings.js")

    assert css.status_code == 200
    assert dashboard_js.status_code == 200
    assert settings_js.status_code == 200
