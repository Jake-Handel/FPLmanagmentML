"""Generates predictions for a given gameweek using the latest trained model."""
import logging

import joblib
import polars as pl

from app.ml.model_registry import get_latest_model_info

logger = logging.getLogger(__name__)


def predict_gameweek(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """feature_matrix must contain a 'player_id' and 'gameweek' column plus
    whatever feature columns the current model expects (loaded alongside it).
    Returns a DataFrame: player_id, gameweek, predicted_points, model_version.
    """
    info = get_latest_model_info()
    if info is None:
        raise RuntimeError("No trained model found. Run `python scripts/train_model.py` first.")

    bundle = joblib.load(info["path"])
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    X = feature_matrix.select(feature_cols).fill_null(0).to_pandas()
    preds = model.predict(X)

    return feature_matrix.select(["player_id", "gameweek"]).with_columns([
        pl.Series("predicted_points", preds),
        pl.lit(info["version"]).alias("model_version"),
    ])
