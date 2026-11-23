"""Optimiser endpoints. These assemble (player_id, position, team_id,
now_cost, predicted_points) rows from the DB and hand them to the ILP
solvers in app/optimizer/*. The DB-fetching here is intentionally thin —
swap in real queries joining Player + Prediction for the given gameweek.
"""
import asyncio

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.collectors.fpl_client import FPLClient
from app.config import get_settings
from app.db.models import Player, Prediction
from app.optimizer.squad_optimizer import optimize_squad
from app.optimizer.transfer_optimizer import optimize_transfers
from app.schemas.optimizer import (
    SquadOptimizeRequest, SquadOptimizeResponse,
    TransferOptimizeRequest, TransferOptimizeResponse,
)

router = APIRouter()


async def _players_with_predictions(db: AsyncSession, gameweek: int) -> list[dict]:
    stmt = (
        select(Player, Prediction.predicted_points)
        .join(Prediction, (Prediction.player_id == Player.id) & (Prediction.gameweek == gameweek))
    )
    result = await db.execute(stmt)
    rows = result.all()
    if not rows:
        raise HTTPException(404, f"No predictions available for gameweek {gameweek}")
    return [
        {
            "id": player.id,
            "position": player.position,
            "team_id": player.team_id,
            "now_cost": player.now_cost,
            "predicted_points": predicted_points,
        }
        for player, predicted_points in rows
    ]


@router.post("/squad", response_model=SquadOptimizeResponse)
async def optimize_squad_endpoint(req: SquadOptimizeRequest, db: AsyncSession = Depends(get_db)):
    """Builds the single best 15-player squad from scratch (e.g. for a wildcard)."""
    players = await _players_with_predictions(db, req.gameweek)
    return optimize_squad(players, budget=req.budget)


@router.post("/transfers", response_model=TransferOptimizeResponse)
async def optimize_transfers_endpoint(req: TransferOptimizeRequest, db: AsyncSession = Depends(get_db)):
    """Recommends transfers for a manager's EXISTING squad.

    NOTE: this stub needs the manager's current squad + bank balance, which
    come from the FPL API (`/entry/{id}/event/{gw}/picks/`), not the local
    DB. Wire in FPLClient().get_manager_picks(...) here before using this
    endpoint for real.
    """
    players = await _players_with_predictions(db, req.gameweek)

    manager_id = req.manager_id or get_settings().default_manager_id
    if manager_id is None:
        raise HTTPException(400, "manager_id required (or set DEFAULT_MANAGER_ID in .env)")

    # Squad comes from /entry/{id}/event/{gw}/picks/ — element = fpl_player_id.
    # Bank is NOT in the picks response (FPL removed it); it lives on the
    # entry endpoint as last_deadline_bank (in millions of pounds).
    client = FPLClient()
    try:
        picks_data, entry_data = await asyncio.gather(
            client.get_manager_picks(manager_id, req.gameweek),
            client.get_manager_entry(manager_id),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"FPL API request failed: {exc}")

    picks = picks_data.get("picks") or []
    if not picks:
        raise HTTPException(404, f"No picks found for manager {manager_id} gameweek {req.gameweek}")

    # fpl_player_id -> internal player.id
    fpl_rows = (await db.execute(select(Player.id, Player.fpl_player_id))).all()
    fpl_to_internal = {fpl_id: internal_id for internal_id, fpl_id in fpl_rows}

    current_squad_ids: list[int] = []
    for pick in picks:
        internal_id = fpl_to_internal.get(pick["element"])
        if internal_id is not None:
            current_squad_ids.append(internal_id)

    # NOTE: FPL no longer returns selling_price/purchase_price, so we can't
    # compute the real sell-price rule (half of price rises since purchase).
    # optimize_transfers falls back to now_cost as the sell-price proxy —
    # see the caveat in transfer_optimizer.py. To do better you'd need to
    # track purchase prices from transfer history over time.

    # Only players we have predictions for can be evaluated; drop any owned
    # player missing from the prediction set (shouldn't happen, but guard).
    predicted_ids = {p["id"] for p in players}
    current_squad_ids = [pid for pid in current_squad_ids if pid in predicted_ids]
    if not current_squad_ids:
        raise HTTPException(404, "Manager's squad has no players with predictions for this gameweek")

    return optimize_transfers(
        current_squad_ids, players, free_transfers=req.free_transfers,
        bank=float(entry_data.get("last_deadline_bank") or 0),
        max_transfers=req.max_transfers,
    )
