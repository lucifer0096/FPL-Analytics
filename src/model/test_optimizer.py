"""Ad-hoc test of the squad optimizer against a real player pool.

Not a unit test suite -- a script to sanity-check optimizer.py against realistic
data (587 real players, real prices/positions/teams from the live bootstrap
snapshot) before trusting it. Uses `form` as a stand-in for predicted_points,
since no live per-gameweek model predictions exist yet (the season hasn't
started) -- this only tests that the SOLVER is correct, not the model.
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import optimize_squad, POSITION_REQUIREMENTS, MAX_PER_TEAM, DEFAULT_BUDGET

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_player_pool() -> pd.DataFrame:
    pattern = os.path.join("data", "raw", "*", "bootstrap", "bootstrap_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No bootstrap snapshot found matching {pattern}")
    with open(paths[-1], encoding="utf-8") as f:
        data = json.load(f)

    teams_by_id = {t["id"]: t["name"] for t in data["teams"]}
    rows = []
    for p in data["elements"]:
        rows.append({
            "player_id": p["id"],
            "name": f"{p['first_name']} {p['second_name']}",
            "position": POSITION_MAP[p["element_type"]],
            "team": teams_by_id[p["team"]],
            "cost": p["now_cost"] / 10.0,  # FPL's API stores price in tenths
            "predicted_points": float(p["form"]) if p["form"] else 0.0,
        })
    return pd.DataFrame(rows)


def verify_squad(squad: pd.DataFrame) -> None:
    print(f"Squad size: {len(squad)} (expect {sum(POSITION_REQUIREMENTS.values())})")
    assert len(squad) == sum(POSITION_REQUIREMENTS.values())

    pos_counts = squad["position"].value_counts().to_dict()
    print(f"Position breakdown: {pos_counts}")
    for pos, quota in POSITION_REQUIREMENTS.items():
        assert pos_counts.get(pos, 0) == quota, f"{pos}: expected {quota}, got {pos_counts.get(pos, 0)}"

    total_cost = squad["cost"].sum()
    print(f"Total cost: £{total_cost:.1f}m (budget: £{DEFAULT_BUDGET}m)")
    assert total_cost <= DEFAULT_BUDGET + 1e-6

    team_counts = squad["team"].value_counts()
    print(f"Max players from one team: {team_counts.max()} (limit: {MAX_PER_TEAM})")
    assert team_counts.max() <= MAX_PER_TEAM

    starters = squad[squad["in_starting_xi"]]
    print(f"Starting XI size: {len(starters)} (expect 11)")
    assert len(starters) == 11

    starter_pos_counts = starters["position"].value_counts().to_dict()
    print(f"Starting XI formation: {starter_pos_counts}")
    assert starter_pos_counts.get("GK", 0) == 1
    assert starter_pos_counts.get("DEF", 0) >= 3
    assert starter_pos_counts.get("MID", 0) >= 2
    assert starter_pos_counts.get("FWD", 0) >= 1

    print("\nAll constraints verified OK.")


if __name__ == "__main__":
    pool = load_player_pool()
    print(f"Loaded player pool: {len(pool)} players\n")

    squad = optimize_squad(pool)
    print("=== Optimal squad ===")
    print(squad[["name", "position", "team", "cost", "predicted_points", "in_starting_xi"]].to_string(index=False))
    print()

    verify_squad(squad)
