from pydantic import BaseModel


class PlayerRef(BaseModel):
    """A player reference for optimiser responses — internal DB id plus the
    FPL-facing identifiers, so callers don't have to guess which id space
    a number belongs to."""
    id: int
    fpl_player_id: int
    web_name: str


class SquadOptimizeRequest(BaseModel):
    gameweek: int
    budget: float = 100.0


class SquadOptimizeResponse(BaseModel):
    squad: list[PlayerRef]
    starting_xi: list[PlayerRef]
    captain: PlayerRef
    total_predicted_points: float


class TransferOptimizeRequest(BaseModel):
    manager_id: int | None = None  # falls back to DEFAULT_MANAGER_ID env var
    gameweek: int
    free_transfers: int = 1
    max_transfers: int = 5


class TransferOptimizeResponse(BaseModel):
    squad: list[PlayerRef]
    starting_xi: list[PlayerRef]
    captain: PlayerRef
    transfers_out: list[PlayerRef]
    transfers_in: list[PlayerRef]
    hits_taken: int
    net_predicted_points: float
