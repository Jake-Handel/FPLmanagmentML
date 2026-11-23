"""Rolling-window features with NO DATA LEAKAGE.

The single most important rule in this whole module: every rolling
aggregate must be computed with `.shift(1)` BEFORE the rolling window, so
that gameweek N's features only ever see data from gameweeks < N. Without
the shift, a player's "rolling average points" for gameweek N would include
gameweek N's own result — which is exactly what we're trying to predict.
"""
import polars as pl

ROLLING_WINDOWS = [3, 5, 10]

# Base stats we build rolling means over (extend as needed)
ROLLING_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "bonus", "bps",
    "ict_index", "xg", "xa",
]


def add_rolling_features(df: pl.DataFrame, group_col: str = "player_id") -> pl.DataFrame:
    """Expects df sorted by (player_id, gameweek) and containing ROLLING_STATS columns."""
    df = df.sort([group_col, "gameweek"])

    expressions = []
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        for window in ROLLING_WINDOWS:
            expressions.append(
                pl.col(stat)
                .fill_null(0)
                .shift(1)                      # <-- excludes the current gameweek
                .rolling_mean(window_size=window, min_periods=1)
                .over(group_col)
                .alias(f"{stat}_roll{window}_mean")
            )

        # Short-vs-long-term trend: positive = player is heating up
        expressions.append(
            (
                pl.col(stat).fill_null(0).shift(1).rolling_mean(window_size=3, min_periods=1)
                - pl.col(stat).fill_null(0).shift(1).rolling_mean(window_size=10, min_periods=1)
            ).over(group_col).alias(f"{stat}_trend")
        )

    return df.with_columns(expressions)


def add_composite_features(df: pl.DataFrame, group_col: str = "player_id") -> pl.DataFrame:
    """Derived ratios that are often more predictive than raw counting stats.
    All inputs are already shifted (built on top of add_rolling_features output
    is fine too, but we recompute directly here for clarity/standalone use).
    """
    df = df.sort([group_col, "gameweek"])
    expressions = []

    if {"shots", "shots_on_target"}.issubset(df.columns):
        expressions.append(
            (
                pl.col("shots_on_target").fill_null(0).shift(1).rolling_sum(window_size=5, min_periods=1)
                / (pl.col("shots").fill_null(0).shift(1).rolling_sum(window_size=5, min_periods=1) + 1e-6)
            ).over(group_col).alias("shot_accuracy_roll5")
        )

    if {"xg", "goals_scored"}.issubset(df.columns):
        expressions.append(
            (
                pl.col("goals_scored").fill_null(0).shift(1).rolling_sum(window_size=10, min_periods=1)
                - pl.col("xg").fill_null(0).shift(1).rolling_sum(window_size=10, min_periods=1)
            ).over(group_col).alias("finishing_overperformance_roll10")
        )

    if expressions:
        df = df.with_columns(expressions)
    return df


def assert_no_leakage(df: pl.DataFrame, feature_cols: list[str], target_gameweek_col: str = "gameweek") -> None:
    """Lightweight sanity check used in tests/CI: for a single player, the
    feature value at gameweek N must be computable from rows < N only. This
    doesn't prove correctness but catches the common bug of forgetting a
    shift() on a newly added feature (row count mismatch / first row non-null
    when it should be null/0 before any history exists is a red flag).
    """
    first_rows = df.sort("gameweek").group_by("player_id").head(1)
    for col in feature_cols:
        if col not in df.columns:
            continue
        non_zero_first = first_rows.filter(pl.col(col) != 0).height
        if non_zero_first > 0:
            raise AssertionError(
                f"Possible leakage in '{col}': {non_zero_first} players have a "
                "non-zero value on their very first tracked gameweek, before any "
                "history exists to compute it from."
            )
