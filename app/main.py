"""FastAPI entrypoint."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, players, teams, fixtures, predictions, optimizer, manager
from app.config import get_settings
from app.db.session import engine
from app.db.base import Base

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("fpl_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FPL AI Manager API")
    try:
        async with engine.begin() as conn:
            # In production, prefer Alembic migrations over create_all.
            # This is a convenience fallback for first-run / dev environments.
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified/created")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise database: %s", exc)

    yield

    logger.info("Shutting down FPL AI Manager API")


app = FastAPI(
    title="FPL AI Manager",
    description="ML-powered Fantasy Premier League predictions and squad optimisation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(teams.router, prefix="/teams", tags=["teams"])
app.include_router(players.router, prefix="/players", tags=["players"])
app.include_router(fixtures.router, prefix="/fixtures", tags=["fixtures"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(optimizer.router, prefix="/optimizer", tags=["optimizer"])
app.include_router(manager.router, prefix="/manager", tags=["manager"])
