"""Optimiser endpoints. These assemble (player_id, position, team_id,
now_cost, predicted_points) rows from the DB and hand them to the ILP
solvers in app/optimizer/*. The DB-fetching here is intentionally thin —
swap in real queries joining Player + Prediction for the given gameweek.
"""
import asyncio
import math

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.collectors.fpl_client import FPLClient
from app.config import get_settings
from app.db.models import Player, PlayerGameweekStat, Prediction
from app.optimizer.squad_optimizer import optimize_squad
from app.optimizer.transfer_optimizer import optimize_transfers
from app.schemas.optimizer import (
    PlayerRef, SquadOptimizeRequest, SquadOptimizeResponse,
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
            "fpl_player_id": player.fpl_player_id,
            "web_name": player.web_name,
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
    result = optimize_squad(players, budget=req.budget)

    ref_by_id = {
        p["id"]: PlayerRef(id=p["id"], fpl_player_id=p["fpl_player_id"], web_name=p["web_name"])
        for p in players
    }
    return SquadOptimizeResponse(
        squad=[ref_by_id[i] for i in result["squad"]],
        starting_xi=[ref_by_id[i] for i in result["starting_xi"]],
        captain=ref_by_id[result["captain_id"]],
        total_predicted_points=result["total_predicted_points"],
    )


async def _get_latest_picks(client: FPLClient, manager_id: int, gameweek: int) -> tuple[int, dict]:
    """Fetch picks for `gameweek`, walking backwards on 404s (future events
    don't exist on the FPL API until they start). Returns (gameweek, picks
    payload) for the most recent available gameweek."""
    for gw in range(gameweek, max(gameweek - 10, 0), -1):
        try:
            data = await client.get_manager_picks(manager_id, gw)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                continue
            raise
        if data.get("picks"):
            return gw, data
    raise httpx.HTTPError(f"No picks found for manager {manager_id} at or before gameweek {gameweek}")


def _apply_pending_transfers(
    picks: list[dict],
    transfers: list[dict],
    picks_gameweek: int,
    target_gameweek: int,
) -> tuple[set[int], float]:
    """Reconstruct the squad for `target_gameweek` from the most recent
    picks snapshot plus transfers made afterwards.

    Picks for a gameweek only exist once that event starts, so during the
    team-selection window the latest picks are last gameweek's LOCKED squad
    and don't reflect transfers already made for the upcoming one. FPL's
    /transfers/ endpoint fills that gap: each transfer's `event` is the
    gameweek it applies to, so everything with picks_gameweek < event <=
    target_gameweek is pending relative to the snapshot.

    Returns (set of FPL element ids, bank delta in millions, number of
    pending transfers applied). Costs arrive x10 (FPL convention), hence
    the /10.
    """
    elements = {p["element"] for p in picks}
    bank_delta = 0.0
    n_pending = 0
    for t in sorted(transfers, key=lambda t: t.get("time") or ""):
        event = t.get("event")
        if event is None or not picks_gameweek < event <= target_gameweek:
            continue
        n_pending += 1
        if t.get("element_out"):
            elements.discard(t["element_out"])
        if t.get("element_in"):
            elements.add(t["element_in"])
        bank_delta += (t.get("element_out_cost", 0) - t.get("element_in_cost", 0)) / 10.0
    return elements, bank_delta, n_pending


def _fpl_sell_price(purchase_price: float, price_series: list[float], now_cost: float) -> float:
    """Approximate FPL's sell-price rule.

    FPL pays purchase_price + half of all price RISES (in 0.1 steps) since
    purchase; drops earn nothing back. We only have weekly price snapshots,
    so intra-week moves are missed — the result is approximate. Capped at
    now_cost so we never overestimate proceeds (conservative direction:
    may miss borderline-affordable transfers, never suggests unaffordable
    ones). Floored to 0.1 like real FPL prices.
    """
    obs = [purchase_price, *price_series, now_cost]
    rises = sum(max(0.0, b - a) for a, b in zip(obs, obs[1:]))
    raw = min(now_cost, purchase_price + rises / 2.0)
    return max(0.1, math.floor(raw * 10 + 1e-9) / 10)


async def _sell_price_map(
    db: AsyncSession,
    current_squad_ids: list[int],
    players_by_id: dict[int, dict],
    transfers: list[dict],
) -> dict[int, float]:
    """Real-ish sell prices for owned players.

    Purchase price comes from the transfer that bought them
    (element_in_cost, x10) — players never transferred were bought at the
    start of the season, approximated by their earliest recorded price.
    Price rises since purchase are summed from PlayerGameweekStat.value
    snapshots (weekly granularity).
    """
    if not current_squad_ids:
        return {}

    stats_rows = (await db.execute(
        select(PlayerGameweekStat.player_id, PlayerGameweekStat.gameweek, PlayerGameweekStat.value)
        .where(PlayerGameweekStat.player_id.in_(current_squad_ids))
        .order_by(PlayerGameweekStat.player_id, PlayerGameweekStat.gameweek)
    )).all()
    price_history: dict[int, list[tuple[int, float]]] = {}
    for pid, gw, value in stats_rows:
        price_history.setdefault(pid, []).append((gw, value))

    # Latest purchase per FPL element id: element -> (price £m, gameweek)
    purchase_by_element: dict[int, tuple[float, int]] = {}
    for t in sorted(transfers, key=lambda t: t.get("time") or ""):
        if t.get("element_in") and t.get("event") is not None:
            purchase_by_element[t["element_in"]] = (
                (t.get("element_in_cost") or 0) / 10.0, t["event"]
            )

    sell_prices: dict[int, float] = {}
    for pid in current_squad_ids:
        player = players_by_id.get(pid)
        if player is None:
            continue
        now_cost = player["now_cost"]
        fpl_id = player["fpl_player_id"]
        history = price_history.get(pid, [])

        purchase = purchase_by_element.get(fpl_id)
        if purchase is not None:
            purchase_price, purchase_gw = purchase
            series = [v for gw, v in history if gw > purchase_gw]
        elif history:
            # Never transferred: first recorded price ~ what they cost at
            # the start of the season.
            purchase_gw, purchase_price = history[0]
            series = [v for gw, v in history if gw > purchase_gw]
        else:
            sell_prices[pid] = now_cost
            continue

        sell_prices[pid] = _fpl_sell_price(purchase_price, series, now_cost)
    return sell_prices


@router.post("/transfers", response_model=TransferOptimizeResponse)
async def optimize_transfers_endpoint(req: TransferOptimizeRequest, db: AsyncSession = Depends(get_db)):
    """Recommends transfers for a manager's EXISTING squad.

    Squad + bank come from the FPL API: the latest available picks snapshot
    (the upcoming gameweek's picks 404 until the event starts) plus pending
    transfers from /entry/{id}/transfers/ to reconstruct the true current
    squad. Money fields arrive x10 and are converted to millions; sell
    prices are approximated from purchase price + price-rise history.
    """
    players = await _players_with_predictions(db, req.gameweek)

    manager_id = req.manager_id or get_settings().default_manager_id
    if manager_id is None:
        raise HTTPException(400, "manager_id required (or set DEFAULT_MANAGER_ID in .env)")

    client = FPLClient()
    try:
        entry_data, (picks_gw, picks_data), transfers = await asyncio.gather(
            client.get_manager_entry(manager_id),
            _get_latest_picks(client, manager_id, req.gameweek),
            client.get_manager_transfers(manager_id),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"FPL API request failed: {exc}")

    picks = picks_data.get("picks") or []
    if not picks:
        raise HTTPException(404, f"No picks found for manager {manager_id}")

    squad_elements, bank_delta, n_pending = _apply_pending_transfers(
        picks, transfers, picks_gw, req.gameweek
    )

    # fpl_player_id -> internal player.id
    fpl_rows = (await db.execute(select(Player.id, Player.fpl_player_id))).all()
    fpl_to_internal = {fpl_id: internal_id for internal_id, fpl_id in fpl_rows}

    current_squad_ids: list[int] = []
    for element in squad_elements:
        internal_id = fpl_to_internal.get(element)
        if internal_id is not None:
            current_squad_ids.append(internal_id)

    # Only players we have predictions for can be evaluated; drop any owned
    # player missing from the prediction set (shouldn't happen, but guard).
    predicted_ids = {p["id"] for p in players}
    current_squad_ids = [pid for pid in current_squad_ids if pid in predicted_ids]
    if not current_squad_ids:
        raise HTTPException(404, "Manager's squad has no players with predictions for this gameweek")

    # FPL money fields (now_cost, last_deadline_bank, transfer costs) all
    # arrive x10 — convert to millions. last_deadline_bank is the bank at
    # the most recent deadline; pending transfers adjust it by their cost
    # deltas (bank_delta is already in millions).
    bank = float(entry_data.get("last_deadline_bank") or 0) / 10.0 + bank_delta

    # Real-ish sell prices for owned players (purchase price + half of
    # price rises since purchase, capped at now_cost). optimize_transfers
    # uses the "sell_price" key when present, now_cost otherwise.
    players_by_id = {p["id"]: p for p in players}
    sell_prices = await _sell_price_map(db, current_squad_ids, players_by_id, transfers)
    for pid, sell_price in sell_prices.items():
        players_by_id[pid]["sell_price"] = sell_price

    # Transfers already made for this gameweek consumed free transfers —
    # only the remainder are available for NEW recommendations.
    free_transfers = max(0, req.free_transfers - n_pending)

    result = optimize_transfers(
        current_squad_ids, players, free_transfers=free_transfers,
        bank=max(0.0, bank),
        max_transfers=req.max_transfers,
    )

    ref_by_id = {
        p["id"]: PlayerRef(id=p["id"], fpl_player_id=p["fpl_player_id"], web_name=p["web_name"])
        for p in players
    }
    return TransferOptimizeResponse(
        squad=[ref_by_id[i] for i in result["squad"]],
        starting_xi=[ref_by_id[i] for i in result["starting_xi"]],
        captain=ref_by_id[result["captain_id"]],
        transfers_out=[ref_by_id[i] for i in result["transfers_out"]],
        transfers_in=[ref_by_id[i] for i in result["transfers_in"]],
        hits_taken=result["hits_taken"],
        net_predicted_points=result["net_predicted_points"],
    )
