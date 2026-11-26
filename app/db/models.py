"""SQLAlchemy ORM models.

Naming: fpl_* ids/codes come straight from the official FPL API.
understat_* ids come from Understat. PlayerMapping reconciles the two.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    fpl_team_id: Mapped[int] = mapped_column(unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    short_name: Mapped[str] = mapped_column(String(10))
    understat_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    players: Mapped[list["Player"]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    fpl_player_id: Mapped[int] = mapped_column(unique=True, index=True)
    fpl_code: Mapped[int] = mapped_column(index=True)  # stable across seasons, unlike fpl_player_id
    web_name: Mapped[str] = mapped_column(String(100))
    full_name: Mapped[str] = mapped_column(String(150))
    position: Mapped[str] = mapped_column(String(3))  # GKP / DEF / MID / FWD
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    now_cost: Mapped[float] = mapped_column(Float)  # in millions, e.g. 5.5
    status: Mapped[str] = mapped_column(String(1), default="a")  # a/d/i/s/u (available/doubtful/injured/...)
    team: Mapped["Team"] = relationship(back_populates="players")

    understat_mapping: Mapped["PlayerMapping | None"] = relationship(
        back_populates="player", uselist=False
    )


class PlayerMapping(Base):
    """Reconciles an FPL player with their Understat identity."""
    __tablename__ = "player_mappings"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), unique=True)
    understat_player_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    match_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    match_method: Mapped[str] = mapped_column(String(30), default="unmatched")  # exact / fuzzy / manual / unmatched

    player: Mapped["Player"] = relationship(back_populates="understat_mapping")


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True)
    fpl_fixture_id: Mapped[int] = mapped_column(unique=True, index=True)
    gameweek: Mapped[int] = mapped_column(index=True)
    kickoff_time: Mapped[datetime | None]
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_difficulty: Mapped[int | None]
    away_difficulty: Mapped[int | None]
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]


class PlayerGameweekStat(Base):
    """One row per player per gameweek, from the official FPL API."""
    __tablename__ = "player_gameweek_stats"
    __table_args__ = (UniqueConstraint("player_id", "gameweek", name="uq_player_gw"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixtures.id"))
    gameweek: Mapped[int] = mapped_column(index=True)

    minutes: Mapped[int] = mapped_column(default=0)
    goals_scored: Mapped[int] = mapped_column(default=0)
    assists: Mapped[int] = mapped_column(default=0)
    clean_sheets: Mapped[int] = mapped_column(default=0)
    goals_conceded: Mapped[int] = mapped_column(default=0)
    own_goals: Mapped[int] = mapped_column(default=0)
    penalties_saved: Mapped[int] = mapped_column(default=0)
    penalties_missed: Mapped[int] = mapped_column(default=0)
    yellow_cards: Mapped[int] = mapped_column(default=0)
    red_cards: Mapped[int] = mapped_column(default=0)
    saves: Mapped[int] = mapped_column(default=0)
    bonus: Mapped[int] = mapped_column(default=0)
    bps: Mapped[int] = mapped_column(default=0)
    influence: Mapped[float] = mapped_column(default=0.0)
    creativity: Mapped[float] = mapped_column(default=0.0)
    threat: Mapped[float] = mapped_column(default=0.0)
    ict_index: Mapped[float] = mapped_column(default=0.0)
    total_points: Mapped[int] = mapped_column(default=0)
    value: Mapped[float] = mapped_column(default=0.0)
    selected_by_percent: Mapped[float] = mapped_column(default=0.0)
    was_home: Mapped[bool] = mapped_column(Boolean, default=True)


class UnderstatPlayerStat(Base):
    """One row per player per match, from Understat (xG / xA / shot data)."""
    __tablename__ = "understat_player_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    understat_match_id: Mapped[str] = mapped_column(String(20))
    gameweek: Mapped[int | None] = mapped_column(index=True)  # resolved via fixture date matching

    xg: Mapped[float] = mapped_column(default=0.0)
    xa: Mapped[float] = mapped_column(default=0.0)
    npxg: Mapped[float] = mapped_column(default=0.0)
    shots: Mapped[int] = mapped_column(default=0)
    key_passes: Mapped[int] = mapped_column(default=0)
    time: Mapped[int] = mapped_column(default=0)


class TeamEloRating(Base):
    """Rolling attack/defence Elo-style rating per team, updated after each match."""
    __tablename__ = "team_elo_ratings"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    gameweek: Mapped[int] = mapped_column(index=True)
    attack_rating: Mapped[float] = mapped_column(default=1500.0)
    defence_rating: Mapped[float] = mapped_column(default=1500.0)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("player_id", "gameweek", "model_version", name="uq_pred"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    gameweek: Mapped[int] = mapped_column(index=True)
    predicted_points: Mapped[float]
    predicted_minutes: Mapped[float | None]
    model_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
