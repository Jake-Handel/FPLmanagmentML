"""Guards the single most important property of the whole pipeline: no
rolling feature may 'see' the gameweek it's trying to predict."""
import polars as pl

from app.features.rolling_features import add_rolling_features, assert_no_leakage


def _sample_df():
    return pl.DataFrame({
        "player_id": [1, 1, 1, 1, 2, 2],
        "gameweek": [1, 2, 3, 4, 1, 2],
        "total_points": [10, 2, 8, 1, 5, 5],
        "minutes": [90, 90, 90, 90, 90, 90],
        "goals_scored": [1, 0, 1, 0, 0, 0],
        "assists": [0, 1, 0, 0, 1, 0],
        "bonus": [3, 0, 1, 0, 0, 1],
        "bps": [30, 10, 25, 5, 15, 15],
        "ict_index": [15.0, 5.0, 12.0, 4.0, 8.0, 8.0],
    })


def test_first_gameweek_has_no_rolling_history():
    df = add_rolling_features(_sample_df())
    first_gw = df.filter(pl.col("gameweek") == 1)
    # On a player's very first tracked gameweek there is no prior history,
    # so the rolling mean must be 0 (not the current game's own score).
    assert (first_gw["total_points_roll3_mean"] == 0).all()


def test_rolling_mean_excludes_current_gameweek():
    df = add_rolling_features(_sample_df())
    player1 = df.filter(pl.col("player_id") == 1).sort("gameweek")
    # At gameweek 2, the roll3 mean should equal gameweek 1's score (10),
    # NOT include gameweek 2's own score (2).
    gw2_roll3 = player1.filter(pl.col("gameweek") == 2)["total_points_roll3_mean"].item()
    assert gw2_roll3 == 10.0


def test_assert_no_leakage_passes_on_clean_data():
    df = add_rolling_features(_sample_df())
    # Should not raise
    assert_no_leakage(df, ["total_points_roll3_mean", "total_points_roll5_mean"])
