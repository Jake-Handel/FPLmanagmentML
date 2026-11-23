"""Orchestrates feature building: joins FPL stats + Understat stats + Elo
ratings + fixture difficulty into one feature matrix, ready for training
or prediction.

Design note: this function is deliberately pure (DataFrame in, DataFrame
out) with no DB or network calls, so it's trivial to unit test and to reuse
identically for both training (historical data) and inference (next
gameweek's fixtures, where the target columns don't exist yet).
"""
import polars as pl

from app.features.rolling_features import add_composite_features, add_rolling_features

TARGET_COL = "total_points"


def build_feature_matrix(
    player_gw_stats: pl.DataFrame,      # from player_gameweek_stats table
    understat_stats: pl.DataFrame,      # from understat_player_stats, pre-aggregated to 1 row/player/gw
    elo_ratings: pl.DataFrame,          # one row per team per gameweek: team_id, gameweek, attack, defence
    fixtures: pl.DataFrame,             # gameweek, player's team_id, opponent_team_id, was_home, fixture_difficulty
) -> pl.DataFrame:
    df = player_gw_stats.join(
        understat_stats, on=["player_id", "gameweek"], how="left"
    )

    df = df.join(
        fixtures.select(["player_id", "gameweek", "opponent_team_id", "was_home", "fixture_difficulty"]),
        on=["player_id", "gameweek"],
        how="left",
    )

    # Bring in the OPPONENT's elo rating (attacking players care about the
    # opponent's defence rating and vice versa; keep both, let the model decide)
    df = df.join(
        elo_ratings.rename({
            "team_id": "opponent_team_id",
            "attack_rating": "opponent_attack_rating",
            "defence_rating": "opponent_defence_rating",
        }),
        on=["opponent_team_id", "gameweek"],
        how="left",
    )

    df = add_rolling_features(df)
    df = add_composite_features(df)

    return df


def split_features_and_target(
    feature_matrix: pl.DataFrame, feature_cols: list[str]
) -> tuple[pl.DataFrame, pl.Series | None]:
    """For training: returns (X, y). For inference on a future, unplayed
    gameweek the target column won't exist, so y is None.
    """
    X = feature_matrix.select(feature_cols)
    if TARGET_COL in feature_matrix.columns:
        y = feature_matrix[TARGET_COL]
    else:
        y = None
    return X, y


DEFAULT_FEATURE_COLUMNS = [
    "total_points_roll3_mean", "total_points_roll5_mean", "total_points_roll10_mean",
    "total_points_trend",
    "minutes_roll3_mean", "minutes_roll5_mean",
    "ict_index_roll3_mean", "ict_index_roll5_mean",
    "xg_roll3_mean", "xg_roll5_mean", "xa_roll3_mean", "xa_roll5_mean",
    "bonus_roll3_mean", "bps_roll5_mean",
    "finishing_overperformance_roll10",
    "opponent_attack_rating", "opponent_defence_rating",
    "was_home", "fixture_difficulty",
]
