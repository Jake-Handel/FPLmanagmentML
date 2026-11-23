from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import Prediction
from app.schemas.prediction import PredictionOut

router = APIRouter()


@router.get("/gameweek/{gameweek}", response_model=list[PredictionOut])
async def get_gameweek_predictions(gameweek: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction).where(Prediction.gameweek == gameweek).order_by(
        Prediction.predicted_points.desc()
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    if not rows:
        raise HTTPException(404, f"No predictions found for gameweek {gameweek}. Has refresh_predictions run?")
    return rows


@router.get("/player/{player_id}", response_model=list[PredictionOut])
async def get_player_predictions(player_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Prediction).where(Prediction.player_id == player_id).order_by(Prediction.gameweek)
    result = await db.execute(stmt)
    return result.scalars().all()
