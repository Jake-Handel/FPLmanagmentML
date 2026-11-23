"""Scraper for Understat.com — a free source of xG / xA / shot-quality data
that the official FPL API doesn't provide.

CAVEAT: this is an unofficial scrape (Understat has no public API). Understat
embeds its data as a JSON blob inside a <script> tag on each page, obfuscated
with a trivial string-escape. This client extracts and decodes it. Because
it depends on Understat's HTML structure, it WILL break if they change their
site — treat it as the most fragile part of the pipeline, wrap calls in the
try/except patterns already shown, and check their terms of use /
rate-limit yourself (small delays between requests) before relying on this
in anything beyond personal, low-volume use.
"""
import json
import logging
import re

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SCRIPT_VAR_RE = re.compile(r"JSON\.parse\('(.+?)'\)")


def _decode_understat_json(raw: str) -> dict | list:
    decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    return json.loads(decoded)


class UnderstatClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.understat_base_url

    async def _fetch_page(self, client: httpx.AsyncClient, path: str) -> str:
        resp = await client.get(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.text

    def _extract_json_blocks(self, html: str) -> list[dict | list]:
        blocks = []
        for match in _SCRIPT_VAR_RE.finditer(html):
            try:
                blocks.append(_decode_understat_json(match.group(1)))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.debug("Skipping unparseable Understat block: %s", exc)
        return blocks

    async def get_league_players(self, league: str = "EPL", season: str = "2025") -> list[dict]:
        """Season-level aggregated stats for every player in the league."""
        async with httpx.AsyncClient(timeout=30) as client:
            html = await self._fetch_page(client, f"/league/{league}/{season}")
        blocks = self._extract_json_blocks(html)
        # Understat's `playersData` block is typically the richest list of dicts
        for block in blocks:
            if isinstance(block, list) and block and "player_name" in block[0]:
                return block
        logger.warning("No playersData block found for %s %s", league, season)
        return []

    async def get_player_matches(self, understat_player_id: str) -> list[dict]:
        """Match-by-match xG/xA for one player."""
        async with httpx.AsyncClient(timeout=30) as client:
            html = await self._fetch_page(client, f"/player/{understat_player_id}")
        blocks = self._extract_json_blocks(html)
        for block in blocks:
            if isinstance(block, list) and block and "xG" in block[0]:
                return block
        return []
