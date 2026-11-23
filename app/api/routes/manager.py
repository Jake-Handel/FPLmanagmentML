"""Pulls a manager's own FPL team directly from the official API (no DB
storage needed for this — it's cheap to fetch live and always current)."""
from fastapi import APIRouter, HTTPException

from app.collectors.fpl_client import FPLClient

router = APIRouter()
client = FPLClient()


@router.get("/{manager_id}")
async def get_manager(manager_id: int):
    try:
        return await client.get_manager_entry(manager_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Failed to fetch manager {manager_id} from FPL API: {exc}")


@router.get("/{manager_id}/picks/{gameweek}")
async def get_manager_picks(manager_id: int, gameweek: int):
    try:
        return await client.get_manager_picks(manager_id, gameweek)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Failed to fetch picks for manager {manager_id}, GW{gameweek}: {exc}")
