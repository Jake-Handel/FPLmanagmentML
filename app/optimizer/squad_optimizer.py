"""Builds the single best valid 15-player squad from predicted points, using
Mixed-Integer Linear Programming (PuLP / CBC solver).

WHY ILP over a heuristic (e.g. greedy "pick highest predicted points per
position until budget runs out"): FPL squad selection has several
simultaneous hard constraints — exact position counts (2/5/5/3), a budget
cap, a max-3-players-per-real-club rule, and (for the starting XI) formation
rules. A greedy heuristic can easily paint itself into a corner (e.g. blow
the budget on forwards and be forced into cheap, low-scoring defenders) and
has no way to prove it found the best *feasible* combination. An ILP solver
explores the whole feasible space and returns a certifiably optimal (or
near-optimal, if time-limited) answer, and it's fast at this scale: ~700
players, 15 picks, solves in well under a second with CBC.
"""
import pulp

SQUAD_BUDGET = 100.0
POSITION_REQUIREMENTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PLAYERS_PER_TEAM = 3

# valid starting-XI formations: (GKP, DEF, MID, FWD)
VALID_FORMATIONS = [
    (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3), (1, 4, 4, 2),
    (1, 4, 5, 1), (1, 5, 2, 3), (1, 5, 3, 2), (1, 5, 4, 1),
]


def optimize_squad(players: list[dict], budget: float = SQUAD_BUDGET) -> dict:
    """players: list of {id, position, team_id, now_cost, predicted_points}.
    Returns {squad: [...15 player ids...], starting_xi: [...11...], captain_id, total_predicted_points}.
    """
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    pick = {p["id"]: pulp.LpVariable(f"pick_{p['id']}", cat="Binary") for p in players}
    starts = {p["id"]: pulp.LpVariable(f"start_{p['id']}", cat="Binary") for p in players}
    is_captain = {p["id"]: pulp.LpVariable(f"cap_{p['id']}", cat="Binary") for p in players}

    by_id = {p["id"]: p for p in players}

    # Objective: maximise predicted points of the starting XI, with the
    # captain's points counted twice (FPL doubles the captain's score).
    prob += pulp.lpSum(
        starts[pid] * by_id[pid]["predicted_points"]
        + is_captain[pid] * by_id[pid]["predicted_points"]
        for pid in by_id
    )

    # --- Squad-level constraints ---
    prob += pulp.lpSum(pick.values()) == 15
    prob += pulp.lpSum(pick[pid] * by_id[pid]["now_cost"] for pid in by_id) <= budget

    for position, count in POSITION_REQUIREMENTS.items():
        prob += pulp.lpSum(pick[pid] for pid in by_id if by_id[pid]["position"] == position) == count

    team_ids = {p["team_id"] for p in players}
    for team_id in team_ids:
        prob += pulp.lpSum(pick[pid] for pid in by_id if by_id[pid]["team_id"] == team_id) <= MAX_PLAYERS_PER_TEAM

    # --- Starting XI constraints ---
    for pid in by_id:
        prob += starts[pid] <= pick[pid]          # can only start players who are in the squad
        prob += is_captain[pid] <= starts[pid]     # captain must be starting

    prob += pulp.lpSum(starts.values()) == 11
    prob += pulp.lpSum(is_captain.values()) == 1

    # Formation: exactly one of the valid (GKP,DEF,MID,FWD) combinations.
    # Modelled as: pick a formation variable per valid formation, constrain
    # position counts to match whichever formation is chosen.
    formation_vars = [pulp.LpVariable(f"formation_{i}", cat="Binary") for i in range(len(VALID_FORMATIONS))]
    prob += pulp.lpSum(formation_vars) == 1

    for position, idx in zip(("GKP", "DEF", "MID", "FWD"), range(4)):
        prob += pulp.lpSum(
            starts[pid] for pid in by_id if by_id[pid]["position"] == position
        ) == pulp.lpSum(
            formation_vars[i] * VALID_FORMATIONS[i][idx] for i in range(len(VALID_FORMATIONS))
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {pulp.LpStatus[prob.status]}")

    squad = [pid for pid in by_id if pick[pid].value() == 1]
    starting_xi = [pid for pid in by_id if starts[pid].value() == 1]
    captain_id = next(pid for pid in by_id if is_captain[pid].value() == 1)

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "captain_id": captain_id,
        "total_predicted_points": pulp.value(prob.objective),
    }
