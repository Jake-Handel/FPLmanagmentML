"""Given an EXISTING squad, recommends which transfers (if any) to make
ahead of the next gameweek(s).

Also ILP-based, for the same reason as squad_optimizer: transfers interact
with the budget, position and per-club constraints exactly like a from-
scratch squad build, just anchored to players you already own. The extra
wrinkle is the transfer-cost trade-off: each transfer beyond your free
transfers costs -4 points, so "should I take a hit" is itself an
optimisation decision, not a rule of thumb.

Horizon: by default this looks one gameweek ahead, which is simplest and
safest. You can pass predicted_points for a multi-gameweek horizon (summed
or discounted) to make it look further ahead — useful for giving a
player's reduced price-pending-injury recovery or a good fixture run more
weight — but every gameweek you add is a guess about the future, so keep
the discount for weeks N+2, N+3... below 1.0.
"""
import pulp

from app.optimizer.squad_optimizer import (
    MAX_PLAYERS_PER_TEAM, POSITION_REQUIREMENTS, SQUAD_BUDGET, VALID_FORMATIONS
)

POINTS_PER_HIT = 4.0


def optimize_transfers(
    current_squad_ids: list[int],
    all_players: list[dict],      # {id, position, team_id, now_cost, predicted_points, sell_price}
    free_transfers: int = 1,
    bank: float = 0.0,
    max_transfers: int = 5,
) -> dict:
    by_id = {p["id"]: p for p in all_players}
    current_set = set(current_squad_ids)

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)

    pick = {pid: pulp.LpVariable(f"pick_{pid}", cat="Binary") for pid in by_id}
    starts = {pid: pulp.LpVariable(f"start_{pid}", cat="Binary") for pid in by_id}
    is_captain = {pid: pulp.LpVariable(f"cap_{pid}", cat="Binary") for pid in by_id}
    # transferred_out[pid] = 1 if a currently-owned player is dropped
    transferred_out = {
        pid: pulp.LpVariable(f"out_{pid}", cat="Binary") for pid in current_set
    }
    n_hits = pulp.LpVariable("n_hits", lowBound=0, cat="Integer")

    # Objective: maximise starting-XI points (captain doubled) minus the
    # point cost of any transfers beyond the free allowance.
    prob += (
        pulp.lpSum(
            starts[pid] * by_id[pid]["predicted_points"] + is_captain[pid] * by_id[pid]["predicted_points"]
            for pid in by_id
        )
        - POINTS_PER_HIT * n_hits
    )

    # Budget: cost of newly bought players must fit within bank + sell value
    # of anyone transferred out. Using each owned player's current `now_cost`
    # as a simple proxy for sell price (ignores FPL's sell-price-on-profit
    # rule — refine with real sell prices from the manager's entry data for
    # full accuracy).
    money_in = pulp.lpSum(
        transferred_out[pid] * by_id[pid].get("sell_price", by_id[pid]["now_cost"])
        for pid in current_set
    )
    money_out = pulp.lpSum(
        pick[pid] * by_id[pid]["now_cost"] for pid in by_id if pid not in current_set
    )
    prob += money_out <= bank + money_in

    for position, count in POSITION_REQUIREMENTS.items():
        prob += pulp.lpSum(pick[pid] for pid in by_id if by_id[pid]["position"] == position) == count
    prob += pulp.lpSum(pick.values()) == 15

    team_ids = {p["team_id"] for p in all_players}
    for team_id in team_ids:
        prob += pulp.lpSum(pick[pid] for pid in by_id if by_id[pid]["team_id"] == team_id) <= MAX_PLAYERS_PER_TEAM

    # Link "transferred out" to the pick variable: a current player is
    # transferred out iff they're no longer picked.
    for pid in current_set:
        prob += transferred_out[pid] == 1 - pick[pid]

    n_transfers = pulp.lpSum(transferred_out.values())
    prob += n_transfers <= max_transfers
    prob += n_hits >= n_transfers - free_transfers

    for pid in by_id:
        prob += starts[pid] <= pick[pid]
        prob += is_captain[pid] <= starts[pid]
    prob += pulp.lpSum(starts.values()) == 11
    prob += pulp.lpSum(is_captain.values()) == 1

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

    new_squad = [pid for pid in by_id if pick[pid].value() == 1]
    transfers_out = [pid for pid in current_set if transferred_out[pid].value() == 1]
    transfers_in = [pid for pid in new_squad if pid not in current_set]

    return {
        "squad": new_squad,
        "starting_xi": [pid for pid in by_id if starts[pid].value() == 1],
        "captain_id": next(pid for pid in by_id if is_captain[pid].value() == 1),
        "transfers_out": transfers_out,
        "transfers_in": transfers_in,
        "hits_taken": int(n_hits.value()),
        "net_predicted_points": pulp.value(prob.objective),
    }
