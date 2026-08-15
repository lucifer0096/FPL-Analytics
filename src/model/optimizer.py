"""FPL squad optimizer: given per-player expected points and constraints, solve
for the optimal 15-man squad and starting XI under FPL's real rules.

FPL squad rules encoded here:
- 15 players total: exactly 2 GK, 5 DEF, 5 MID, 3 FWD.
- Total spend <= budget (default GBP100.0m, FPL's standard starting budget).
- At most 3 players from any single real-life club.
- Starting XI: exactly 1 GK, and at least 3 DEF / 2 MID / 1 FWD among the outfield
  10, matching FPL's minimum-per-position rule for a valid formation.

This is a pure optimization layer -- it takes a DataFrame of (player_id, position,
team, cost, predicted_points) and returns a squad. It doesn't care whether
predicted_points came from the trained xP model, FPL's own xP, or a manual
watchlist -- keeping the solver decoupled from the model makes it usable (and
testable) even before the model's live-gameweek predictions exist.
"""

import pandas as pd
import pulp

SQUAD_SIZE = 15
POSITION_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
DEFAULT_BUDGET = 100.0

STARTING_XI_SIZE = 11
STARTING_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}


def optimize_squad(
    players: pd.DataFrame,
    budget: float = DEFAULT_BUDGET,
    id_col: str = "player_id",
    position_col: str = "position",
    team_col: str = "team",
    cost_col: str = "cost",
    points_col: str = "predicted_points",
) -> pd.DataFrame:
    """Solve for the 15-man squad that maximizes total predicted points, subject
    to FPL's squad-composition, budget, and per-team constraints.

    Returns the input DataFrame filtered to the selected 15 players, with an
    added `in_starting_xi` column marking the optimal starting XI within that
    squad (a separate, nested optimization -- see _select_starting_xi)."""
    required_cols = {id_col, position_col, team_col, cost_col, points_col}
    missing = required_cols - set(players.columns)
    if missing:
        raise ValueError(f"players DataFrame is missing required columns: {missing}")

    unknown_positions = set(players[position_col].unique()) - set(POSITION_REQUIREMENTS)
    if unknown_positions:
        raise ValueError(f"Unexpected position values (expected GK/DEF/MID/FWD): {unknown_positions}")

    prob = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)

    player_ids = players[id_col].tolist()
    pick = pulp.LpVariable.dicts("pick", player_ids, cat="Binary")

    points = players.set_index(id_col)[points_col].to_dict()
    cost = players.set_index(id_col)[cost_col].to_dict()
    position = players.set_index(id_col)[position_col].to_dict()
    team = players.set_index(id_col)[team_col].to_dict()

    # Objective: maximize total predicted points across the 15-man squad.
    prob += pulp.lpSum(pick[pid] * points[pid] for pid in player_ids)

    # Exactly 15 players.
    prob += pulp.lpSum(pick[pid] for pid in player_ids) == SQUAD_SIZE

    # Budget.
    prob += pulp.lpSum(pick[pid] * cost[pid] for pid in player_ids) <= budget

    # Exact position quotas (2 GK, 5 DEF, 5 MID, 3 FWD).
    for pos, quota in POSITION_REQUIREMENTS.items():
        pos_ids = [pid for pid in player_ids if position[pid] == pos]
        prob += pulp.lpSum(pick[pid] for pid in pos_ids) == quota

    # At most 3 players per real club.
    for club in set(team.values()):
        club_ids = [pid for pid in player_ids if team[pid] == club]
        prob += pulp.lpSum(pick[pid] for pid in club_ids) <= MAX_PER_TEAM

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"Solver did not find an optimal solution (status: {pulp.LpStatus[status]}). "
            f"Check that enough players exist per position/budget/team constraints."
        )

    selected_ids = [pid for pid in player_ids if pick[pid].value() == 1]
    squad = players[players[id_col].isin(selected_ids)].copy()

    starting_ids = _select_starting_xi(squad, id_col, position_col, points_col)
    squad["in_starting_xi"] = squad[id_col].isin(starting_ids)

    return squad.sort_values([position_col, points_col], ascending=[True, False]).reset_index(drop=True)


def _select_starting_xi(
    squad: pd.DataFrame, id_col: str, position_col: str, points_col: str
) -> list:
    """Given a fixed 15-man squad, pick the 11 starters that maximize predicted
    points subject to FPL's formation rule (1 GK, >=3 DEF, >=2 MID, >=1 FWD)."""
    prob = pulp.LpProblem("fpl_starting_xi", pulp.LpMaximize)

    player_ids = squad[id_col].tolist()
    start = pulp.LpVariable.dicts("start", player_ids, cat="Binary")

    points = squad.set_index(id_col)[points_col].to_dict()
    position = squad.set_index(id_col)[position_col].to_dict()

    prob += pulp.lpSum(start[pid] * points[pid] for pid in player_ids)
    prob += pulp.lpSum(start[pid] for pid in player_ids) == STARTING_XI_SIZE

    gk_ids = [pid for pid in player_ids if position[pid] == "GK"]
    prob += pulp.lpSum(start[pid] for pid in gk_ids) == STARTING_XI_MIN["GK"]

    for pos in ["DEF", "MID", "FWD"]:
        pos_ids = [pid for pid in player_ids if position[pid] == pos]
        prob += pulp.lpSum(start[pid] for pid in pos_ids) >= STARTING_XI_MIN[pos]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Starting XI solver failed (status: {pulp.LpStatus[status]})")

    return [pid for pid in player_ids if start[pid].value() == 1]
