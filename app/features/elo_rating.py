"""A custom Elo-style rating system for team strength, driven by xG rather
than results. A team that creates lots of chances but loses to a fluke
scoreline still sees its rating improve — over a season this tracks
underlying quality better than results alone.

Separate attack/defence ratings let the model express "strong defence, weak
attack" teams correctly, rather than collapsing a team into one number.
"""
from dataclasses import dataclass, field

K_FACTOR = 20.0
MIN_RATING = 1000.0
MAX_RATING = 2200.0
START_RATING = 1500.0


@dataclass
class TeamRatingState:
    attack: float = START_RATING
    defence: float = START_RATING


class TeamEloRatingSystem:
    def __init__(self, k_factor: float = K_FACTOR):
        self.k_factor = k_factor
        self.ratings: dict[int, TeamRatingState] = {}

    def get(self, team_id: int) -> TeamRatingState:
        return self.ratings.setdefault(team_id, TeamRatingState())

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Standard Elo expectation curve, scaled to this xG rating range."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _bounded(self, rating: float) -> float:
        return max(MIN_RATING, min(MAX_RATING, rating))

    def update_after_match(
        self,
        home_team_id: int,
        away_team_id: int,
        home_xg: float,
        away_xg: float,
    ) -> None:
        """Update attack/defence ratings for both teams based on this match's xG.

        We treat each team's xG output as an "attack vs opposing defence"
        contest, and each team's xG *conceded* as a "defence vs opposing
        attack" contest, so four one-sided Elo updates happen per match.
        """
        home = self.get(home_team_id)
        away = self.get(away_team_id)

        # Normalise xG into a 0-1 "match score" for the Elo formula via a simple
        # soft-max of (xg_for - xg_against); teams that heavily out-xG their
        # opponent get a score near 1, even scraps sit near 0.5.
        def xg_to_score(xg_for: float, xg_against: float) -> float:
            diff = xg_for - xg_against
            return 1.0 / (1.0 + pow(2.718281828, -diff))  # logistic squashing

        home_attack_score = xg_to_score(home_xg, away_xg)
        away_attack_score = xg_to_score(away_xg, home_xg)

        # Attack ratings move based on attack performance vs opponent's defence
        expected_home_attack = self._expected(home.attack, away.defence)
        expected_away_attack = self._expected(away.attack, home.defence)

        home.attack = self._bounded(
            home.attack + self.k_factor * (home_attack_score - expected_home_attack)
        )
        away.attack = self._bounded(
            away.attack + self.k_factor * (away_attack_score - expected_away_attack)
        )

        # Defence ratings move inversely: conceding more xG than expected hurts
        expected_home_defence = self._expected(home.defence, away.attack)
        expected_away_defence = self._expected(away.defence, home.attack)

        home.defence = self._bounded(
            home.defence + self.k_factor * ((1 - away_attack_score) - expected_home_defence)
        )
        away.defence = self._bounded(
            away.defence + self.k_factor * ((1 - home_attack_score) - expected_away_defence)
        )
