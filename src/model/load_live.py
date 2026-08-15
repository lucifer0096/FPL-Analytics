"""Load this project's OWN collector snapshots (data/raw/{season}/gw_history.csv)
for 2026-27 onward -- as opposed to load_historical.py, which loads vaastav's
archived dataset for 2016-17 through 2025-26.

This split exists because vaastav's merged_gw.csv schema structurally can't
carry several fields the live FPL API actually returns per gameweek:
in_dreamteam, defensive_contribution (part of FPL's 2025-26 scoring overhaul),
starts, and real expected_goals/expected_assists/expected_goal_involvements/
expected_goals_conceded. Those columns don't exist for ANY historical season in
vaastav's data -- not a coverage gap to backfill, a genuine "FPL didn't publish
this shape of data when that CSV was built" gap. This loader preserves them for
whichever seasons this project's own collector actually captured, rather than
dropping them to match vaastav's older column set.

snapshot_gameweek_stats() in src/collector/snapshot.py already writes every key
from the live API's element-summary `history` rows verbatim (plus player_id,
web_name, element_type) -- so no changes were needed there. This loader just
reads that file back and normalizes it to the same core shape load_historical.py
produces (GW/element/opponent_team/team/position/player_code/season/...), so the
two can be concatenated for feature engineering, while keeping the extra
FPL-only columns intact (NaN for historical rows once concatenated, since they
genuinely don't exist there)."""

import glob
import json
import os

import pandas as pd

RAW_DIR = os.environ.get("FPL_RAW_DATA_ROOT", os.path.join("data", "raw"))

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Columns this loader adds/renames to match load_historical.py's shape --
# everything else from gw_history.csv (in_dreamteam, defensive_contribution,
# starts, expected_goals, etc.) passes through untouched.
_RENAME = {"element": "element", "player_id": "element"}


def live_seasons_available(raw_dir: str = None) -> list:
    """Which seasons have a real gw_history.csv (i.e. at least one finished
    gameweek has actually been collected) under raw_dir."""
    raw_dir = raw_dir or RAW_DIR
    pattern = os.path.join(raw_dir, "*", "gw_history.csv")
    return sorted(
        os.path.basename(os.path.dirname(p)) for p in glob.glob(pattern)
    )


def _latest_bootstrap(season_dir: str) -> dict:
    pattern = os.path.join(season_dir, "bootstrap", "bootstrap_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No bootstrap snapshot found under {season_dir}")
    with open(paths[-1], encoding="utf-8") as f:
        return json.load(f)


def load_live_season(season: str, raw_dir: str = None) -> pd.DataFrame:
    """Load one season's worth of this project's own collected gameweek data.
    Returns an empty-schema DataFrame (not an error) if nothing's been
    collected yet for this season -- a season with zero finished gameweeks is
    an expected state early on, not a bug."""
    raw_dir = raw_dir or RAW_DIR
    season_dir = os.path.join(raw_dir, season)
    gw_path = os.path.join(season_dir, "gw_history.csv")

    if not os.path.exists(gw_path):
        return pd.DataFrame()

    df = pd.read_csv(gw_path, low_memory=False)
    if df.empty:
        return df

    df["element"] = df["player_id"]
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["name"] = df["web_name"]

    # opponent_team/team come through as numeric ids, same as vaastav's data --
    # resolved to name strings via this season's own collected bootstrap, same
    # pattern load_historical.py uses via teams.csv.
    bootstrap = _latest_bootstrap(season_dir)
    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df["opponent_team"] = df["opponent_team"].map(team_names)

    element_to_team = {p["id"]: p["team"] for p in bootstrap["elements"]}
    element_to_code = {p["id"]: p["code"] for p in bootstrap["elements"]}
    df["team"] = df["element"].map(element_to_team).map(team_names)
    df["player_code"] = df["element"].map(element_to_code)

    unmatched = df["player_code"].isna().sum()
    if unmatched:
        print(f"  WARNING [{season}, live]: {unmatched} rows have no player_code "
              f"match (player left the bootstrap between snapshot and this one?)")

    df["season"] = season
    df["value"] = df["value"]  # already 10x-scaled cost, same convention as vaastav's data

    return df


def load_all_live_seasons(raw_dir: str = None) -> pd.DataFrame:
    """Load and concatenate every season this project's own collector has data
    for. Returns an empty DataFrame if nothing's been collected yet -- callers
    should treat that as "no live data yet", not an error, since it's the
    expected state before a season's first gameweek finishes."""
    raw_dir = raw_dir or RAW_DIR
    seasons = live_seasons_available(raw_dir)
    if not seasons:
        return pd.DataFrame()
    frames = [load_live_season(s, raw_dir) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_all_live_seasons()
    if df.empty:
        print("No live-collected gameweek data yet (expected before any 2026-27+ "
              "gameweek has finished and been collected).")
    else:
        print(f"Loaded {len(df):,} rows across seasons: {sorted(df['season'].unique())}")
        extra_cols = [c for c in df.columns if c in (
            "in_dreamteam", "defensive_contribution", "starts",
            "expected_goals", "expected_assists", "expected_goal_involvements",
            "expected_goals_conceded",
        )]
        print(f"FPL-only columns present (not in vaastav's data): {extra_cols}")
