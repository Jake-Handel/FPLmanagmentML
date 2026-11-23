"""One-off script: pulls bootstrap-static from the FPL API and populates the
teams/players tables. Run this first, before anything else.

    python -m scripts.seed_initial_data
"""
import asyncio
import logging

from sqlalchemy import select

from app.collectors.fpl_client import FPLClient
from app.db.models import Player, Team
from app.db.session import AsyncSessionLocal

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


async def main():
    client = FPLClient()
    data = await client.get_bootstrap_static()

    async with AsyncSessionLocal() as session:
        team_id_map: dict[int, int] = {}  # fpl_team_id -> internal id

        for t in data["teams"]:
            existing = (
                await session.execute(select(Team).where(Team.fpl_team_id == t["id"]))
            ).scalar_one_or_none()
            if existing:
                team_id_map[t["id"]] = existing.id
                continue
            team = Team(fpl_team_id=t["id"], name=t["name"], short_name=t["short_name"])
            session.add(team)
            await session.flush()
            team_id_map[t["id"]] = team.id

        await session.commit()
        logger.info("Seeded %d teams", len(team_id_map))

        count = 0
        for p in data["elements"]:
            existing = (
                await session.execute(select(Player).where(Player.fpl_player_id == p["id"]))
            ).scalar_one_or_none()
            if existing:
                continue
            player = Player(
                fpl_player_id=p["id"],
                fpl_code=p["code"],
                web_name=p["web_name"],
                full_name=f"{p['first_name']} {p['second_name']}",
                position=POSITION_MAP.get(p["element_type"], "UNK"),
                team_id=team_id_map[p["team"]],
                now_cost=p["now_cost"] / 10.0,  # FPL stores price *10
                status=p["status"],
            )
            session.add(player)
            count += 1

        await session.commit()
        logger.info("Seeded %d new players", count)


if __name__ == "__main__":
    asyncio.run(main())
