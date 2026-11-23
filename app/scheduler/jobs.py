"""Scheduled background jobs, run in the `worker` container.

- ingest_bootstrap_static: pulls current players/teams/prices (frequent —
  prices and injury flags change throughout the week).
- ingest_latest_gameweek_results: once a gameweek finishes, pulls final
  stats + Understat xG and writes them to the DB.
- refresh_predictions: regenerates predictions for the upcoming gameweek,
  e.g. nightly and again a few hours before each deadline (team news lands
  late, so a pre-deadline refresh matters more than raw frequency).
- retrain_model: periodic full retrain (e.g. weekly) as more gameweeks of
  data accumulate.

This uses APScheduler for simplicity (single-process, no extra broker to
run). If you outgrow a single worker process — e.g. you want retries,
distributed workers, or a dashboard of job runs — swap this for Celery +
Redis/RabbitMQ; the job *functions* below don't need to change, only how
they're invoked.
"""
import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select

import polars as pl

from app.collectors.fpl_client import FPLClient
from app.collectors.player_matcher import PlayerMatcher
from app.collectors.understat_client import UnderstatClient
from app.db.models import (
    Fixture,
    Player,
    PlayerGameweekStat,
    PlayerMapping,
    Prediction,
    Team,
    TeamEloRating,
    UnderstatPlayerStat,
)
from app.db.session import AsyncSessionLocal
from app.features.elo_rating import TeamEloRatingSystem
from app.features.feature_pipeline import build_feature_matrix
from app.ml.predict import predict_gameweek

logger = logging.getLogger(__name__)

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


async def ingest_bootstrap_static():
    logger.info("Running job: ingest_bootstrap_static")
    client = FPLClient()
    data = await client.get_bootstrap_static()

    async with AsyncSessionLocal() as session:
        team_id_map: dict[int, int] = {}  # fpl_team_id -> internal id

        for t in data["teams"]:
            existing = (
                await session.execute(select(Team).where(Team.fpl_team_id == t["id"]))
            ).scalar_one_or_none()
            if existing:
                existing.name = t["name"]
                existing.short_name = t["short_name"]
                team_id_map[t["id"]] = existing.id
                continue
            team = Team(fpl_team_id=t["id"], name=t["name"], short_name=t["short_name"])
            session.add(team)
            await session.flush()
            team_id_map[t["id"]] = team.id

        await session.commit()
        logger.info("Synced %d teams", len(team_id_map))

        count = 0
        updated = 0
        for p in data["elements"]:
            existing = (
                await session.execute(select(Player).where(Player.fpl_player_id == p["id"]))
            ).scalar_one_or_none()
            if existing:
                existing.web_name = p["web_name"]
                existing.full_name = f"{p['first_name']} {p['second_name']}"
                existing.now_cost = p["now_cost"] / 10.0
                existing.status = p["status"]
                updated += 1
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
        logger.info("Updated %d players, inserted %d new", updated, count)


async def load_team_map(session) -> dict[int, int]:
    result = await session.execute(select(Team.id,Team.fpl_team_id))
    return {fpl_team_id: team_id for team_id, fpl_team_id in result.all()}


async def load_player_map(session) -> dict[int, int]:
    result = await session.execute(select(Player.id,Player.fpl_player_id))
    return {fpl_player_id: player_id for player_id, fpl_player_id in result.all()}

def latest_finished_gameweek(fixtures: list[Fixture]) -> int | None:
    finished = [f.gameweek for f in fixtures if f.finished]
    return max(finished) if finished else None


async def sync_fixtures(session, client: FPLClient) -> int:
    """Pull GET /fixtures/ and upsert every fixture into the `fixtures` table.

    FPL fixture fields -> Fixture columns:
      id            -> fpl_fixture_id  (unique key for upsert)
      event         -> gameweek
      kickoff_time  -> kickoff_time (parse ISO string to datetime)
      team_h/team_a -> home_team_id/away_team_id (FPL team ids -> internal ids
                      via load_team_map; FPL stores them as ints)
      team_h/difficulty, team_a/difficulty -> home_difficulty/away_difficulty
      finished      -> finished
      team_h_score/team_a_score -> home_score/away_score (None until played)

    Upsert on fpl_fixture_id: if a row exists UPDATE all mutable fields
    (scores/finished change after matches); else INSERT.

    Returns the number of fixtures upserted.
    """
    raw_fixtures = await client.get_fixtures()
    team_map = await load_team_map(session)
    existing = {f.fpl_fixture_id: f for f in (await session.execute(select(Fixture))).scalars()}
    count = 0
    for fixture in raw_fixtures:
        fpl_fixture_id = fixture["id"]
        gameweek = fixture["event"]
        kickoff_time = datetime.fromisoformat(
            fixture["kickoff_time"].replace("Z", "+00:00")
        ).replace(tzinfo=None)
        home_team_id = team_map[fixture["team_h"]]
        away_team_id = team_map[fixture["team_a"]]
        home_difficulty = None
        away_difficulty = None
        finished = fixture["finished"]
        home_score = fixture["team_h_score"]
        away_score = fixture["team_a_score"]
        
        if fpl_fixture_id in existing:
            fx = existing[fpl_fixture_id]
            fx.gameweek = gameweek
            fx.kickoff_time = kickoff_time
            fx.home_team_id = home_team_id
            fx.away_team_id = away_team_id
            fx.home_difficulty = home_difficulty
            fx.away_difficulty = away_difficulty
            fx.finished = finished
            fx.home_score = home_score
            fx.away_score = away_score
        else:
            session.add(Fixture(
                fpl_fixture_id=fpl_fixture_id,
                gameweek=gameweek,
                kickoff_time=kickoff_time,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_difficulty=home_difficulty,
                away_difficulty=away_difficulty,
                finished=finished,
                home_score=home_score,
                away_score=away_score
            ))
        count += 1
            
    await session.commit()
    return count


# ---------------------------------------------------------------------------
# E2 — player gameweek stats
# ---------------------------------------------------------------------------

async def sync_player_gameweek_stats(session, client: FPLClient) -> int:
    """Upsert PlayerGameweekStat rows for every finished gameweek.

    Scope decision: we sync ALL finished gameweeks, not just the latest.
    The job runs nightly and upserts are idempotent, so catching up after
    downtime costs nothing extra.
    """
    fixtures = (await session.execute(select(Fixture))).scalars().all()
    finished_gws = {f.gameweek for f in fixtures if f.finished}
    if not finished_gws:
        logger.info("No finished fixtures yet, skipping player stats")
        return 0

    player_map = await load_player_map(session)   # fpl_player_id -> internal id

    # History entries carry "fixture" = the FPL fixture id, which maps
    # directly to Fixture.fpl_fixture_id — simpler and immune to the
    # team-transfer edge cases a (gameweek, team) lookup would have.
    fixture_lookup = {f.fpl_fixture_id: f.id for f in fixtures}

    summaries = await client.get_player_summaries(list(player_map.keys()))

    # Existing rows keyed for O(1) upsert lookup
    existing = {
        (s.player_id, s.gameweek): s
        for s in (await session.execute(select(PlayerGameweekStat))).scalars().all()
    }

    count = 0
    for fpl_pid, summary in summaries.items():
        internal_id = player_map.get(int(fpl_pid))
        if internal_id is None:
            continue
        for entry in summary.get("history", []):
            gw = entry.get("round")
            if gw not in finished_gws:
                continue

            fixture_id = fixture_lookup.get(entry.get("fixture"))

            row = existing.get((internal_id, gw))
            if row is None:
                row = PlayerGameweekStat(player_id=internal_id, gameweek=gw)
                session.add(row)
                existing[(internal_id, gw)] = row

            row.fixture_id = fixture_id
            row.minutes = int(entry.get("minutes") or 0)
            row.goals_scored = int(entry.get("goals_scored") or 0)
            row.assists = int(entry.get("assists") or 0)
            row.clean_sheets = int(entry.get("clean_sheets") or 0)
            row.goals_conceded = int(entry.get("goals_conceded") or 0)
            row.own_goals = int(entry.get("own_goals") or 0)
            row.penalties_saved = int(entry.get("penalties_saved") or 0)
            row.penalties_missed = int(entry.get("penalties_missed") or 0)
            row.yellow_cards = int(entry.get("yellow_cards") or 0)
            row.red_cards = int(entry.get("red_cards") or 0)
            row.saves = int(entry.get("saves") or 0)
            row.bonus = int(entry.get("bonus") or 0)
            row.bps = int(entry.get("bps") or 0)
            # FPL returns some numeric fields as strings — cast explicitly.
            row.influence = float(entry.get("influence") or 0)
            row.creativity = float(entry.get("creativity") or 0)
            row.threat = float(entry.get("threat") or 0)
            row.ict_index = float(entry.get("ict_index") or 0)
            row.total_points = int(entry.get("total_points") or 0)
            # FPL stores price in tenths (55 = £5.5m) — same convention as
            # Player.now_cost, so divide by 10 for consistency.
            row.value = float(entry.get("value") or 0) / 10.0
            row.selected_by_percent = float(entry.get("selected_by_percent") or 0)
            row.was_home = entry.get("was_home", True)
            count += 1

    await session.commit()
    logger.info("Upserted %d player-gameweek stat rows", count)
    return count


# ---------------------------------------------------------------------------
# E3 — Understat xG data
# ---------------------------------------------------------------------------

async def sync_player_mappings(session) -> int:
    """E3a — one-off-ish: match FPL players to Understat identities.

    Steps:
      1. understat_players = await UnderstatClient().get_league_players()
         # list of {player_name, team_title, id, ...} — inspect actual keys
      2. fpl_players = [{id: internal player.id, full_name, team_id} ...]
         # NOTE: matcher expects 'id' = INTERNAL player id (it writes
         # player_mappings.player_id which FKs to players.id)
      3. team_name_to_id: map Understat team_title -> internal team id.
         Team names differ between sources ("Man United" vs "Man Utd") —
         you may need a small alias dict. Inspect both sides first!
      4. matches = PlayerMatcher().match_all(fpl_players, understat_players,
                                             team_name_to_id)
      5. Upsert into PlayerMapping on player_id: set understat_player_id,
         match_confidence, match_method.
      6. Log a summary: counts by match_method (exact/fuzzy/unmatched).
         UNMATCHED players are expected (~a few) — log them by name so a
         human can review; don't fail the job.

    Returns number of mappings written.
    """
    # TODO: implement.
    raise NotImplementedError


async def sync_understat_match_stats(session) -> int:
    """E3b — pull match-by-match xG/xA for every matched player.

    Steps:
      1. mappings = SELECT PlayerMapping WHERE understat_player_id IS NOT NULL
      2. For each mapping (consider limiting concurrency — Understat is a
         scrape, be polite: sequential or small batches with a delay):
           matches = await UnderstatClient().get_player_matches(
               mapping.understat_player_id)
           # each match dict has ~{'xG','xA','npxG','shots','key_passes',
           #   'time','date','h_team','a_team','h_score','a_score',...}
           # INSPECT the real keys first — print one match dict during dev.
      3. Upsert UnderstatPlayerStat rows keyed on
         (player_id, understat_match_id):
           player_id          = mapping.player_id
           understat_match_id = match identifier (find the stable id field)
           gameweek           = None for now (resolved in next step)
           xg/xa/npxg/shots/key_passes/time = from match dict
      4. Commit once at end.

    Returns number of rows upserted.
    """
    # TODO: implement.
    raise NotImplementedError


async def resolve_understat_gameweeks(session) -> int:
    """E3c — THE HARD PART: assign FPL gameweek numbers to Understat matches.

    Understat doesn't know FPL gameweeks. Each UnderstatPlayerStat row needs
    gameweek set so the feature pipeline can join on (player_id, gameweek).

    Matching strategy (date + teams):
      1. Load all Fixture rows that are finished: kickoff_time,
         home_team_id, away_team_id, gameweek.
      2. For each UnderstatPlayerStat with gameweek NULL, find its match's
         date + the player's team that day. PROBLEM: you need to know WHICH
         team the player played for in that Understat match — the match
         dict's h_team/a_team plus the player's name tells you, but you
         stored only understat_match_id... You may need to ALSO store the
         match date and team on UnderstatPlayerStat (schema has no date
         column — you could parse date during E3b and stash it, or match
         during E3b while the match dict is in hand).

    SIMPLER ALTERNATIVE worth considering: do gameweek resolution INSIDE
    sync_understat_match_stats while you still have the match dict — match
    date -> fixture kickoff_time (same calendar day, or within a few hours
    for timezone edges), and team -> fixture home/away. If exactly one
    fixture matches, set gameweek immediately. Only fall back to a separate
    pass for unmatched rows.

    Whatever you choose: ambiguous matches (two fixtures same day involving
    same team — rare but happens with reschedules) should stay NULL and be
    logged, not guessed.

    Returns number of rows resolved.
    """
    # TODO: implement (or fold into sync_understat_match_stats — decide).
    raise NotImplementedError


# ---------------------------------------------------------------------------
# E4 — Elo ratings
# ---------------------------------------------------------------------------

async def update_elo_ratings(session) -> int:
    """Replay all finished fixtures through TeamEloRatingSystem and persist
    per-gameweek attack/defence ratings.

    Steps:
      1. Load finished Fixture rows ORDER BY gameweek, kickoff_time —
         order matters, Elo is path-dependent.
      2. For each fixture, compute team xG:
           - players' UnderstatPlayerStat.xg for that gameweek, grouped by
             the player's team. Join PlayerGameweekStat (or players table)
             to know which team each player belonged to THAT gameweek —
             current team may differ (transfers). If PlayerGameweekStat
             lacks team_id, approximate with players.team_id and comment
             the limitation.
           - home_xg = sum of home team's players' xg; away_xg likewise.
             NOTE: UnderstatPlayerStat.gameweek must be resolved (E3c)
             before this works.
      3. elo = TeamEloRatingSystem()  # fresh state = replay from scratch;
         # deterministic and immune to missed job runs. ~380 matches/season
         # is trivial compute.
      4. for each fixture in order: elo.update_after_match(
             home_fpl_team_id->? , ...)
         CAREFUL: update_after_match takes TEAM IDs — use internal team ids
         consistently (home_team_id/away_team_id from Fixture are already
         internal). The ratings dict keys are whatever you pass — keep it
         internal ids end to end.
      5. After each gameweek's matches, snapshot: for every team that
         played, upsert TeamEloRating(team_id, gameweek,
         attack_rating=state.attack, defence_rating=state.defence).
         Teams that didn't play keep their previous rating — carry it
         forward or leave the row absent (feature pipeline joins on
         gameweek; decide and comment).
      6. Commit once at end.

    Returns number of TeamEloRating rows written.
    """
    # TODO: implement.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The job itself — orchestrates E1 -> E3. E4 can be its own job or called
# here; it's cheap enough to run nightly after stats land.
# ---------------------------------------------------------------------------

async def ingest_latest_gameweek_results():
    logger.info("Running job: ingest_latest_gameweek_results")
    client = FPLClient()

    async with AsyncSessionLocal() as session:
        # E1: fixtures must be fresh before stats can reference them
        n_fixtures = await sync_fixtures(session, client)

        # E2: player stats for finished gameweeks
        n_stats = await sync_player_gameweek_stats(session, client)

        # E3a-d (Understat xG + Elo) are intentionally disabled while
        # understat.com is offline. Re-enable when a data source is wired:
        # n_maps = await sync_player_mappings(session)
        # n_xg = await sync_understat_match_stats(session)
        # n_resolved = await resolve_understat_gameweeks(session)
        # n_elo = await update_elo_ratings(session)

        logger.info(
            "ingest_latest_gameweek_results done: %d fixtures, %d stat rows",
            n_fixtures, n_stats,
        )


async def refresh_predictions():
    logger.info("Running job: refresh_predictions")
    async with AsyncSessionLocal() as session:
        target_gw = (
            await session.execute(
                select(func.min(Fixture.gameweek)).where(Fixture.finished == False)
            )
        ).scalar_one_or_none()

        if target_gw is None:
            logger.info("No upcoming fixtures")
            return

        gw_fixtures = (
            await session.execute(
                select(Fixture).where(
                    Fixture.gameweek == target_gw,
                    Fixture.finished == False,
                )
            )
        ).scalars().all()

        stats_rows = (await session.execute(select(PlayerGameweekStat))).scalars().all()
        stats = pl.DataFrame([
            {
                "player_id": s.player_id,
                "gameweek": s.gameweek,
                "total_points": s.total_points,
                "minutes": s.minutes,
                "goals_scored": s.goals_scored,
                "assists": s.assists,
                "bonus": s.bonus,
                "bps": s.bps,
                "ict_index": s.ict_index,
            }
            for s in stats_rows
        ])
        
        playing_team_ids = set()
        for fx in gw_fixtures:
            playing_team_ids.add(fx.home_team_id)
            playing_team_ids.add(fx.away_team_id)
            
        players = (await session.execute(
            select(Player).where(Player.team_id.in_(playing_team_ids))
        )).scalars().all()
        
        future_rows = pl.DataFrame([
            {"player_id": p.id, "gameweek": target_gw}
            for p in players
        ])
        stats = pl.concat([stats, future_rows], how="diagonal_relaxed")
        
        fixture_lookup = {}
        for fx in gw_fixtures:
            fixture_lookup[(fx.home_team_id,target_gw)] = (fx.away_team_id, True)
            fixture_lookup[(fx.away_team_id,target_gw)] = (fx.home_team_id, False)
            
        fixture_rows = []
        for p in players:
            opponent, was_home = fixture_lookup[(p.team_id, target_gw)]
            fixture_rows.append({
                "player_id": p.id,
                "gameweek": target_gw,
                "opponent_team_id": opponent,
                "was_home": was_home,
                "fixture_difficulty": None,
            })
        fixtures_df = pl.DataFrame(fixture_rows).with_columns(
            pl.col("fixture_difficulty").cast(pl.Float64)
        )
        
        empty_stats = pl.DataFrame(schema={"player_id": pl.Int64, "gameweek": pl.Int64})
        empty_elo = pl.DataFrame(schema={
            "team_id": pl.Int64,
            "gameweek": pl.Int64,
            "attack_rating": pl.Float64,
            "defence_rating": pl.Float64,
        })
        feature_matrix = build_feature_matrix(
            stats,
            understat_stats=empty_stats,
            elo_ratings=empty_elo,
            fixtures=fixtures_df,
        )

        future_matrix = feature_matrix.filter(pl.col("gameweek") == target_gw)
        preds = predict_gameweek(future_matrix)
        
        existing = {
            (p.player_id, p.gameweek, p.model_version) : p
            for p in (await session.execute(
                select(Prediction).where(
                    Prediction.gameweek == target_gw,
                    Prediction.model_version == preds["model_version"][0],
                )
            )).scalars().all()
        }
        
        for row in preds.iter_rows(named=True):
            key = (row["player_id"], row["gameweek"], row["model_version"])
            if key in existing:
                existing[key].predicted_points = row["predicted_points"]
            else:
                session.add(Prediction(
                    player_id=row["player_id"],
                    gameweek=row["gameweek"],
                    predicted_points = row["predicted_points"],
                    model_version=row["model_version"],
                ))
                
        await session.commit()
        logger.info("Stored %d predictions for gameweek %d", preds.height, target_gw)

async def retrain_model():
    logger.info("Running job: retrain_model")
    # Reuses the exact same flow as `python -m scripts.train_model` so the
    # scheduled retrain and manual retrains can never diverge.
    from scripts.train_model import main as run_training
    await run_training()


def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(ingest_bootstrap_static, CronTrigger(hour="*/4"))
    scheduler.add_job(ingest_latest_gameweek_results, CronTrigger(hour=3, minute=0))
    scheduler.add_job(refresh_predictions, CronTrigger(hour="6,18"))
    scheduler.add_job(retrain_model, CronTrigger(day_of_week="mon", hour=4))
    return scheduler


async def main():
    logging.basicConfig(level="INFO")
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("Scheduler started. Waiting for jobs...")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
