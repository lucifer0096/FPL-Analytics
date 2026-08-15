"""Ad-hoc test of the chip advisor against real historical data.

Not a unit test suite -- builds a squad at 2025-26 GW10, then projects (using
each player's rolling-5 average as a predicted_points stand-in, same as
test_optimizer.py) across GW10-14 to check the suggestions are sensible: real,
plausible players surfacing as Triple Captain candidates, differentiated
Bench Boost values across gameweeks, and a Free Hit gap that correctly reflects
how far the existing squad has drifted from what's freshly optimal that week.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import optimize_squad
from chips import suggest_bench_boost, suggest_triple_captain, suggest_free_hit_or_wildcard

TEST_SEASON = "2025-26"
SQUAD_BUILD_GW = 10
GW_RANGE = range(10, 15)


def pool_for_gw(df: pd.DataFrame, gw: int) -> pd.DataFrame:
    sub = df[(df["season"] == TEST_SEASON) & (df["GW"] == gw)].copy()
    sub["player_id"] = sub["player_code"]
    sub["cost"] = sub["value"] / 10.0
    sub["predicted_points"] = sub["total_points_avg_last_5"].fillna(0).clip(lower=0)
    return sub.drop_duplicates(subset="player_id")[
        ["player_id", "name", "position", "team", "cost", "predicted_points"]
    ]


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join("data", "processed", "features.parquet"))

    squad = optimize_squad(pool_for_gw(df, SQUAD_BUILD_GW))
    print(f"Built squad at GW{SQUAD_BUILD_GW}\n")

    future_points_by_gw = {}
    optimal_points_by_gw = {}
    for gw in GW_RANGE:
        pool = pool_for_gw(df, gw)
        future_points_by_gw[gw] = dict(zip(pool["player_id"], pool["predicted_points"]))
        opt_squad = optimize_squad(pool)
        optimal_points_by_gw[gw] = opt_squad["predicted_points"].sum()

    print("=== Bench Boost suggestions (best 3) ===")
    for s in suggest_bench_boost(squad, future_points_by_gw, GW_RANGE)[:3]:
        print(f"  GW{s.gameweek}: {s.detail}")

    print("\n=== Triple Captain suggestions (best 3) ===")
    tc = suggest_triple_captain(squad, future_points_by_gw, GW_RANGE)
    for s in tc[:3]:
        print(f"  GW{s.gameweek}: {s.detail}")
    assert all(s.score >= 0 for s in tc), "Triple Captain extra value should never be negative"

    print("\n=== Free Hit suggestions (best 3) ===")
    fh = suggest_free_hit_or_wildcard(squad, future_points_by_gw, optimal_points_by_gw, GW_RANGE, chip="free_hit")
    for s in fh[:3]:
        print(f"  GW{s.gameweek}: {s.detail}")
    assert fh[0].score >= fh[-1].score, "Suggestions should be sorted descending by score"

    print("\nAll checks passed.")
