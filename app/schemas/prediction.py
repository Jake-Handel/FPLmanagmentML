from pydantic import BaseModel


class PredictionOut(BaseModel):
    player_id: int
    gameweek: int
    predicted_points: float
    model_version: str

    model_config = {"from_attributes": True}
