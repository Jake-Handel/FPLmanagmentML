from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import Fixture

router = APIRouter()


@router.get("")
async def list_fixtures(
    gameweek: int | None = Query(None),
    team_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Fixture)
    if gameweek:
        stmt = stmt.where(Fixture.gameweek == gameweek)
    if team_id:
        stmt = stmt.where((Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id))
    result = await db.execute(stmt.order_by(Fixture.kickoff_time))
    return result.scalars().all()
