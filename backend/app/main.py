from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.frontend import FRONTEND_ROOT
from backend.app.api.routes.frontend import router as frontend_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.operations import router as operations_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )
    app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT / "assets"), name="assets")
    app.include_router(frontend_router)
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(operations_router, prefix=settings.api_prefix)
    return app


app = create_app()
