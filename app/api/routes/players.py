from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import Player, PlayerGameweekStat
from app.schemas.player import PlayerOut

router = APIRouter()


@router.get("", response_model=list[PlayerOut])
async def list_players(
    position: str | None = Query(None, description="Filter by GKP/DEF/MID/FWD"),
    team_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Player)
    if position:
        stmt = stmt.where(Player.position == position)
    if team_id:
        stmt = stmt.where(Player.team_id == team_id)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{player_id}", response_model=PlayerOut)
async def get_player(player_id: int, db: AsyncSession = Depends(get_db)):
    player = await db.get(Player, player_id)
    if not player:
        raise HTTPException(404, "Player not found")
    return player


@router.get("/{player_id}/history")
async def get_player_history(player_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(PlayerGameweekStat).where(PlayerGameweekStat.player_id == player_id).order_by(
        PlayerGameweekStat.gameweek
    )
    result = await db.execute(stmt)
    return result.scalars().all()
