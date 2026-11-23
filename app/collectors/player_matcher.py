"""Reconciles FPL players with their Understat identity.

FPL and Understat use different name formats (e.g. FPL "Fernandes B." vs
Understat "Bruno Fernandes"). Strategy, cheapest-to-most-expensive:
  1. Exact match on normalised full name within the same team.
  2. Fuzzy match (rapidfuzz token_sort_ratio) constrained to the same team,
     to avoid matching two unrelated players who happen to share a surname.
  3. Leave unmatched for manual review (logged, not silently dropped).

Constraining fuzzy search by team first is the key trick: it turns an
O(n_fpl * n_understat) global fuzzy search into a much smaller, much more
accurate per-team search.
"""
import logging
import unicodedata

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 85.0  # 0-100; below this we leave the player unmatched


def _normalise(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return name.lower().strip()


class PlayerMatcher:
    def match_all(
        self,
        fpl_players: list[dict],          # each: {id, full_name, team_id}
        understat_players: list[dict],    # each: {id, player_name, team_title}
        team_name_to_id: dict[str, int],  # understat team name -> internal team id
    ) -> list[dict]:
        """Returns a list of {player_id, understat_player_id, confidence, method}."""
        # Group understat players by resolved internal team id for O(1) lookup
        by_team: dict[int, list[dict]] = {}
        for up in understat_players:
            team_id = team_name_to_id.get(up["team_title"])
            if team_id is not None:
                by_team.setdefault(team_id, []).append(up)

        results = []
        for fp in fpl_players:
            candidates = by_team.get(fp["team_id"], [])
            match = self._find_best_match(fp, candidates)
            if match:
                understat_id, confidence, method = match
                results.append({
                    "player_id": fp["id"],
                    "understat_player_id": understat_id,
                    "match_confidence": confidence,
                    "match_method": method,
                })
            else:
                logger.info("No Understat match for FPL player %s (%s)", fp["id"], fp["full_name"])
                results.append({
                    "player_id": fp["id"],
                    "understat_player_id": None,
                    "match_confidence": 0.0,
                    "match_method": "unmatched",
                })
        return results

    def _find_best_match(self, fpl_player: dict, candidates: list[dict]):
        target = _normalise(fpl_player["full_name"])

        # 1. exact match
        for cand in candidates:
            if _normalise(cand["player_name"]) == target:
                return cand["id"], 100.0, "exact"

        # 2. fuzzy match, constrained to this team's candidates only
        best_score, best_cand = 0.0, None
        for cand in candidates:
            score = fuzz.token_sort_ratio(target, _normalise(cand["player_name"]))
            if score > best_score:
                best_score, best_cand = score, cand

        if best_cand and best_score >= FUZZY_MATCH_THRESHOLD:
            return best_cand["id"], float(best_score), "fuzzy"

        return None
