from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import Team
from app.schemas.player import TeamOut

router = APIRouter()


@router.get("", response_model=list[TeamOut])
async def list_teams(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Team))
    return result.scalars().all()
