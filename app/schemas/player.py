from pydantic import BaseModel


class PlayerOut(BaseModel):
    id: int
    web_name: str
    full_name: str
    position: str
    team_id: int
    now_cost: float
    status: str

    model_config = {"from_attributes": True}


class TeamOut(BaseModel):
    id: int
    name: str
    short_name: str

    model_config = {"from_attributes": True}
