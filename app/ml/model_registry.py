"""Tracks trained model versions so the API always knows which model is
'current' without hardcoding a filename. Deliberately file-based (a JSON
manifest next to the .joblib files) rather than a DB table — it keeps
model artefacts and their metadata co-located and trivial to inspect/copy
between environments; swap for a DB table if you need multi-instance
consistency guarantees.
"""
import json
from pathlib import Path

from app.config import get_settings

settings = get_settings()


def get_latest_model_info(model_dir: str | None = None) -> dict | None:
    model_dir = Path(model_dir or settings.model_dir)
    registry_path = model_dir / "registry.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text())
    if not registry:
        return None
    return sorted(registry, key=lambda r: r["created_at"])[-1]
