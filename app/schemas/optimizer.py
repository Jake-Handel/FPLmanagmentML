from pydantic import BaseModel


class SquadOptimizeRequest(BaseModel):
    gameweek: int
    budget: float = 100.0


class SquadOptimizeResponse(BaseModel):
    squad: list[int]
    starting_xi: list[int]
    captain_id: int
    total_predicted_points: float


class TransferOptimizeRequest(BaseModel):
    manager_id: int | None = None  # falls back to DEFAULT_MANAGER_ID env var
    gameweek: int
    free_transfers: int = 1
    max_transfers: int = 5


class TransferOptimizeResponse(BaseModel):
    squad: list[int]
    starting_xi: list[int]
    captain_id: int
    transfers_out: list[int]
    transfers_in: list[int]
    hits_taken: int
    net_predicted_points: float
