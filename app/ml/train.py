"""Model training.

Default: LightGBM regressor predicting next-gameweek total_points directly.
This is deliberately simpler than a full AutoML stack (AutoGluon etc.) —
it's fast to train, easy to debug, ships with sane defaults, and is a
drop-in interface you can later swap for AutoGluon/XGBoost/an ensemble
without touching the rest of the pipeline (see model_registry.py).

Validation strategy: TIME-BASED split, not random k-fold. Predicting
gameweek 20 using a model that trained partly on gameweek 25 data would
be leakage at the model-selection level, even if individual features are
leak-free. We hold out the most recent N gameweeks as validation.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score

from app.config import get_settings
from app.features.feature_pipeline import DEFAULT_FEATURE_COLUMNS, TARGET_COL

logger = logging.getLogger(__name__)
settings = get_settings()

VALIDATION_HOLDOUT_GAMEWEEKS = 4


def time_based_split(df: pl.DataFrame, holdout_gws: int = VALIDATION_HOLDOUT_GAMEWEEKS):
    max_gw = df["gameweek"].max()
    cutoff = max_gw - holdout_gws
    train_df = df.filter(pl.col("gameweek") <= cutoff)
    val_df = df.filter(pl.col("gameweek") > cutoff)
    return train_df, val_df


def train_points_model(
    feature_matrix: pl.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURE_COLUMNS,
    model_dir: str | None = None,
) -> dict:
    model_dir = Path(model_dir or settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = time_based_split(feature_matrix)

    X_train = train_df.select(feature_cols).fill_null(0).to_pandas()
    y_train = train_df[TARGET_COL].to_pandas()
    X_val = val_df.select(feature_cols).fill_null(0).to_pandas()
    y_val = val_df[TARGET_COL].to_pandas()

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    preds = model.predict(X_val)
    metrics = {
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }
    logger.info("Validation metrics: %s", metrics)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"points_model_{version}.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    registry_path = model_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "version": version,
        "path": str(model_path),
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    registry_path.write_text(json.dumps(registry, indent=2))

    return {"version": version, "model_path": str(model_path), "metrics": metrics}
