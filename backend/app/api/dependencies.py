from collections.abc import Generator

from backend.app.core.config import Settings, get_settings


def settings_dependency() -> Generator[Settings]:
    yield get_settings()
