"""CLI entry point to train (or retrain) the points-prediction model end to
end: pull historical stats from the DB, build the feature matrix, train,
and register the new model version.

    python -m scripts.train_model
"""
import asyncio
import logging

import polars as pl
from sqlalchemy import select

from app.db.models import Fixture, PlayerGameweekStat
from app.db.session import AsyncSessionLocal
from app.features.feature_pipeline import DEFAULT_FEATURE_COLUMNS, build_feature_matrix
from app.ml.train import train_points_model

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


async def load_training_data() -> pl.DataFrame:
    async with AsyncSessionLocal() as session:
        stats = (await session.execute(select(PlayerGameweekStat))).scalars().all()

    rows = [
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
        for s in stats
    ]
    return pl.DataFrame(rows)


async def load_fixture_frame() -> pl.DataFrame:
    """One row per (player, gameweek) describing that player's fixture:
    opponent team, home/away, and difficulty.

    opponent_team_id is expressed in INTERNAL team ids (matching
    TeamEloRating.team_id when Elo data exists). fixture_difficulty is NULL
    for now — FPL doesn't expose FDR via its public API; revisit once Elo
    ratings provide a strength-based proxy.
    """
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(PlayerGameweekStat, Fixture)
                .join(Fixture, PlayerGameweekStat.fixture_id == Fixture.id)
            )
        ).all()

    data = [
        {
            "player_id": stat.player_id,
            "gameweek": stat.gameweek,
            "opponent_team_id": (
                fx.away_team_id if stat.was_home else fx.home_team_id
            ),
            "was_home": stat.was_home,
            "fixture_difficulty": None,
        }
        for stat, fx in rows
    ]
    return pl.DataFrame(data).with_columns(
        pl.col("fixture_difficulty").cast(pl.Float64)
    )


async def main():
    player_gw_stats = await load_training_data()
    if player_gw_stats.is_empty():
        logger.error("No historical data found. Run data ingestion first.")
        return

    # TODO: pass real understat_stats / elo_ratings once Understat is back
    # online (or an alternative xG source is wired). Fixtures are real now.
    # NOTE: empty frames must carry the join/rename columns the pipeline
    # expects — elo_ratings needs team_id/attack_rating/defence_rating or
    # the .rename() in build_feature_matrix fails.
    fixtures_df = await load_fixture_frame()
    empty_stats = pl.DataFrame(schema={"player_id": pl.Int64, "gameweek": pl.Int64})
    empty_elo = pl.DataFrame(schema={
        "team_id": pl.Int64,
        "gameweek": pl.Int64,
        "attack_rating": pl.Float64,
        "defence_rating": pl.Float64,
    })
    feature_matrix = build_feature_matrix(
        player_gw_stats,
        understat_stats=empty_stats,
        elo_ratings=empty_elo,
        fixtures=fixtures_df,
    )

    available_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in feature_matrix.columns]
    result = train_points_model(feature_matrix, feature_cols=available_cols)
    logger.info("Trained model %s -> %s", result["version"], result["metrics"])


if __name__ == "__main__":
    asyncio.run(main())
