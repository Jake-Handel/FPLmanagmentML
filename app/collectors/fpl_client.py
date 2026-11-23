"""Async client for the official (unauthenticated, undocumented-but-stable) FPL API.

Key endpoints used:
  GET /bootstrap-static/          -> all players, teams, gameweeks (events) in one call
  GET /fixtures/                  -> full fixture list with difficulty ratings
  GET /element-summary/{id}/      -> one player's full match-by-match history
  GET /entry/{manager_id}/        -> a manager's basic info (team name, overall rank...)
  GET /entry/{manager_id}/event/{gw}/picks/  -> a manager's squad for a given gameweek
"""
import asyncio
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FPLClient:
    def __init__(self, base_url: str | None = None, max_concurrent: int = 5):
        self.base_url = base_url or settings.fpl_base_url
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _get(self, client: httpx.AsyncClient, path: str) -> dict:
        async with self._semaphore:
            resp = await client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()

    async def get_bootstrap_static(self) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._get(client, "/bootstrap-static/")

    async def get_fixtures(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._get(client, "/fixtures/")

    async def get_player_summary(self, fpl_player_id: int) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._get(client, f"/element-summary/{fpl_player_id}/")

    async def get_player_summaries(self, fpl_player_ids: list[int]) -> dict[int, dict]:
        """Fetch many players concurrently, respecting the semaphore limit."""
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = {
                pid: asyncio.create_task(self._get(client, f"/element-summary/{pid}/"))
                for pid in fpl_player_ids
            }
            results = {}
            for pid, task in tasks.items():
                try:
                    results[pid] = await task
                except httpx.HTTPError as exc:
                    logger.warning("Failed to fetch player %s: %s", pid, exc)
            return results

    async def get_manager_entry(self, manager_id: int) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._get(client, f"/entry/{manager_id}/")

    async def get_manager_picks(self, manager_id: int, gameweek: int) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            return await self._get(client, f"/entry/{manager_id}/event/{gameweek}/picks/")
