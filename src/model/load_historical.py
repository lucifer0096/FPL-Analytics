"""Load and unify historical FPL gameweek data (2016-17 through 2024-25) from the
vaastav/Fantasy-Premier-League dataset for xP model training.

Source data lives outside this repo, at a local clone of
https://github.com/vaastav/Fantasy-Premier-League (that fork is archived after
2024-25 and won't be updated — see this repo's README for why FPL-Analytics exists).

Feature availability varies by season:
- 2016-17 to 2019-20: no `position`/`team` in merged_gw.csv. `position` is joined in
  from players_raw.csv (element_type); `team` is a numeric, season-scoped ID in
  players_raw.csv rather than a name, so it's further resolved to a team name via
  master_team_list.csv, matching the string names every other season already uses
  natively. No xG/xA fields in this range.
- 2020-21 to 2024-25: `position`/`team` present directly in merged_gw.csv, with
  `team` already a name string (e.g. "Brighton") consistent with the resolved
  names for earlier seasons. 2022-23 onward additionally has full xG/xA fields.
  2024-25 in this local clone only goes up to gameweek 14 (the fork's data
  collection appears to have stopped mid-season).

This loader targets the 33 columns common to every season (see COMMON_COLUMNS)
plus position/team, normalized to consistent types (team as a name string
everywhere), so the core model can train consistently across all 9 seasons.
xG/xA fields are NOT included here — they're only available for the 3 most recent
seasons and would need a separate, narrower training set if used.
"""

import os
import pandas as pd

VAASTAV_ROOT = os.environ.get("VAASTAV_DATA_ROOT", r"E:\Fantasy-Premier-League\data")

ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

# Present in merged_gw.csv for every season 2016-17 through 2024-25.
COMMON_COLUMNS = [
    "GW", "assists", "bonus", "bps", "clean_sheets", "creativity", "element",
    "fixture", "goals_conceded", "goals_scored", "ict_index", "influence",
    "kickoff_time", "minutes", "name", "opponent_team", "own_goals",
    "penalties_missed", "penalties_saved", "red_cards", "round", "saves",
    "selected", "team_a_score", "team_h_score", "threat", "total_points",
    "transfers_balance", "transfers_in", "transfers_out", "value", "was_home",
    "yellow_cards",
]

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# master_team_list.csv only covers 2016-17 through 2023-24. It's only needed here
# for the seasons where merged_gw.csv lacks a team column at all (2016-17 to
# 2019-20), so the missing 2024-25 row is never actually looked up.
_MASTER_TEAM_LIST_PATH = os.path.join(VAASTAV_ROOT, "master_team_list.csv")


def _load_position_team(season: str) -> pd.DataFrame:
    """End-of-season position + numeric team-id snapshot per player id, from
    players_raw.csv. Used to backfill position/team for seasons where
    merged_gw.csv lacks both columns (2016-17 to 2019-20)."""
    path = os.path.join(VAASTAV_ROOT, season, "players_raw.csv")
    df = pd.read_csv(path, encoding="latin1")
    return df[["id", "element_type", "team"]].rename(
        columns={"id": "element", "team": "team_id"}
    )


def _resolve_team_names(season: str, team_ids: pd.Series) -> pd.Series:
    """Map a season's numeric team ids to team name strings, so `team` is a
    consistent name (e.g. "Brighton") across every season, matching what
    merged_gw.csv already provides natively from 2020-21 onward."""
    master = pd.read_csv(_MASTER_TEAM_LIST_PATH)
    season_map = master[master["season"] == season].set_index("team")["team_name"]
    return team_ids.map(season_map)


def load_season(season: str) -> pd.DataFrame:
    """Load one season's gameweek data, unified to COMMON_COLUMNS + position/team,
    with team normalized to a name string in every season."""
    path = os.path.join(VAASTAV_ROOT, season, "gws", "merged_gw.csv")
    df = pd.read_csv(path, encoding="latin1", low_memory=False)

    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{season}: merged_gw.csv is missing expected columns {missing}")

    df = df[COMMON_COLUMNS].copy()

    if "position" in pd.read_csv(path, encoding="latin1", nrows=1).columns:
        # team is already a name string in these seasons (verified 2020-21 to
        # 2024-25) — no resolution needed.
        pos_team = pd.read_csv(path, encoding="latin1", low_memory=False)[
            ["element", "position", "team"]
        ].drop_duplicates(subset="element")
        df = df.merge(pos_team, on="element", how="left")
    else:
        pos_team = _load_position_team(season)
        df = df.merge(pos_team, on="element", how="left")
        df["position"] = df["element_type"].map(POSITION_MAP)
        df["team"] = _resolve_team_names(season, df["team_id"])
        df = df.drop(columns=["element_type", "team_id"])

    df["season"] = season
    return df


def load_all_seasons(seasons: list = None) -> pd.DataFrame:
    """Load and concatenate all requested seasons into one unified table.
    Defaults to every season in ALL_SEASONS."""
    seasons = seasons or ALL_SEASONS
    frames = []
    for s in seasons:
        print(f"Loading {s}...")
        frames.append(load_season(s))
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} player-gameweek rows across {len(seasons)} seasons.")
    return combined


if __name__ == "__main__":
    df = load_all_seasons()
    print(df.shape)
    print(df["season"].value_counts().sort_index())
    print(df["position"].value_counts(dropna=False))
