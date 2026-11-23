import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.cache.redis_cache import get_client
from app.ml.model_registry import get_latest_model_info

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    checks = {"database": "unknown", "redis": "unknown", "model": "unknown"}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"

    try:
        await get_client().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    model_info = get_latest_model_info()
    checks["model"] = model_info["version"] if model_info else "no model trained yet"

    overall = "ok" if checks["database"] == "ok" and checks["redis"] == "ok" else "degraded"
    return {"status": overall, "checks": checks}
