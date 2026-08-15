"""Ad-hoc test of the squad optimizer against real player pools.

Not a unit test suite -- a script to sanity-check optimizer.py against two kinds
of realistic data before trusting it:

1. The live bootstrap pool (587 real players, real prices/positions/teams).
   Pre-season `form` is 0 for every player right now (no gameweeks played yet),
   so this only proves the SOLVER logic is correct -- constraints hold, valid
   formation -- not that it picks good players.
2. Historical data (2025-26 GW20, each player's rolling-5 average as a stand-in
   for predicted_points) -- 2025-26 specifically, not the 2024-25 season this
   project's model was validated against, for two reasons: it's the untouched
   final-holdout season, and its player pool is far closer to who's actually in
   the Premier League now (the 2024-25 pool includes players like Salah who have
   since left). This checks that, given real varied scores, the optimizer
   actually selects good players -- not just a constraint-satisfying squad.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import (
    optimize_squad,
    optimize_transfers,
    load_latest_prices,
    POSITION_REQUIREMENTS,
    MAX_PER_TEAM,
    DEFAULT_BUDGET,
    POINTS_PER_HIT,
    SQUAD_SIZE,
)

HISTORICAL_TEST_SEASON = "2025-26"
HISTORICAL_TEST_GW = 20


def load_historical_pool(season: str = HISTORICAL_TEST_SEASON, gw: int = HISTORICAL_TEST_GW) -> pd.DataFrame:
    features_path = os.path.join("data", "processed", "features.parquet")
    df = pd.read_parquet(features_path)
    df = df[(df["season"] == season) & (df["GW"] == gw)].copy()

    df["player_id"] = df["player_code"]
    df["cost"] = df["value"] / 10.0  # FPL's API stores price in tenths
    df["predicted_points"] = df["total_points_avg_last_5"].fillna(0).clip(lower=0)
    df = df.drop_duplicates(subset="player_id")

    return df[["player_id", "name", "position", "team", "cost", "predicted_points"]]


def load_player_pool() -> pd.DataFrame:
    """Live bootstrap pool with `form` as a predicted_points stand-in (form is 0
    for everyone pre-season -- see this module's docstring). Uses the same
    latest-snapshot file load_latest_prices() would pick, so both stay in sync."""
    import glob
    import json

    pool = load_latest_prices()

    pattern = os.path.join("data", "raw", "*", "bootstrap", "bootstrap_*.json")
    latest_path = sorted(glob.glob(pattern))[-1]
    with open(latest_path, encoding="utf-8") as f:
        data = json.load(f)
    form_by_id = {p["id"]: (float(p["form"]) if p["form"] else 0.0) for p in data["elements"]}
    pool["predicted_points"] = pool["player_id"].map(form_by_id)

    return pool


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
    print("### Test 1: live bootstrap pool (solver correctness only -- form is 0 pre-season) ###\n")
    pool = load_player_pool()
    print(f"Loaded player pool: {len(pool)} players\n")
    squad = optimize_squad(pool)
    print("=== Optimal squad ===")
    print(squad[["name", "position", "team", "cost", "predicted_points", "in_starting_xi"]].to_string(index=False))
    print()
    verify_squad(squad)

    print(f"\n\n### Test 2: historical data ({HISTORICAL_TEST_SEASON} GW{HISTORICAL_TEST_GW}) -- real varied scores ###\n")
    hist_pool = load_historical_pool()
    print(f"Loaded player pool: {len(hist_pool)} players\n")
    hist_squad = optimize_squad(hist_pool)
    print("=== Optimal squad ===")
    print(hist_squad.sort_values("predicted_points", ascending=False)[
        ["name", "position", "team", "cost", "predicted_points", "in_starting_xi"]
    ].to_string(index=False))
    print()
    verify_squad(hist_squad)
    print(f"\nTotal predicted points: {hist_squad['predicted_points'].sum():.1f}")
    print(f"Total cost: £{hist_squad['cost'].sum():.1f}m")

    print(f"\n\n### Test 3: transfer optimizer (GW{HISTORICAL_TEST_GW} squad -> GW{HISTORICAL_TEST_GW + 1} pool) ###\n")
    next_gw_pool = load_historical_pool(gw=HISTORICAL_TEST_GW + 1)
    # Only players present in both gameweeks' pools are valid transfer targets/
    # holdovers -- a player who left the pool (rare, e.g. deregistered) can't be
    # priced or projected for the next gameweek.
    common_ids = set(hist_squad["player_id"]) & set(next_gw_pool["player_id"])
    squad_ids_for_transfer = [pid for pid in hist_squad["player_id"] if pid in common_ids]
    if len(squad_ids_for_transfer) < len(hist_squad):
        print(f"Note: {len(hist_squad) - len(squad_ids_for_transfer)} squad player(s) "
              f"not in the GW{HISTORICAL_TEST_GW + 1} pool, excluded from this test.")

    if len(squad_ids_for_transfer) == SQUAD_SIZE:
        result = optimize_transfers(
            current_squad_ids=squad_ids_for_transfer,
            players=next_gw_pool,
            free_transfers=1,
            bank=0.0,
        )
        print(f"Transfers out: {result['transfers_out']}")
        print(f"Transfers in: {result['transfers_in']}")
        print(f"Paid transfers: {result['num_paid_transfers']} (hit cost: -{result['hit_cost']} pts)")
        print(f"Net points gain: {result['net_points_gain']:.1f}")
        verify_squad(result["new_squad"])
    else:
        print("Skipping transfer test: squad player set changed too much between gameweeks.")
