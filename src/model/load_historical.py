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

IMPORTANT: `element` (the player id used within a season's files) is reassigned
every season — id 1 is a different real player in every one of the 9 seasons here.
Grouping by `element` across seasons silently blends unrelated players' stats
together. `players_raw.csv`'s `code` field is FPL's actual stable, cross-season
player identifier (verified: Salah's code stayed 118748 across three seasons
while his `id` changed each year) — joined in here as `player_code`. Any rolling/
lagged feature spanning gameweeks or seasons MUST group by `player_code`, not
`element`.

IMPORTANT: `opponent_team` is a numeric, season-scoped team id in EVERY season
(verified: id 4 is Chelsea in 2016-17, Burnley in 2020-21, Brentford in 2022-23)
— same class of bug as `team`, but present even in seasons where `team` is
already a string. Resolved to a name string here via teams.csv (2019-20 onward,
including 2024-25) or master_team_list.csv (2016-17 to 2018-19 fallback).
"""

import os
import pandas as pd

VAASTAV_ROOT = os.environ.get("VAASTAV_DATA_ROOT", r"E:\Fantasy-Premier-League\data")

ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
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

# merged_gw.csv's encoding is NOT consistent across the dataset -- verified by
# inspecting raw bytes directly (not assumed): 2016-17/2017-18/2018-19 are
# genuinely Latin-1 (accented names appear as single Latin-1 bytes, e.g. \xe9 for
# "e"), while 2019-20 onward are genuine UTF-8 (accented names appear as the
# 2-byte UTF-8 sequence, e.g. \xc3\xa9 for "e"). A single fixed encoding for every
# season is wrong either way -- using latin1 universally (an earlier version of
# this loader's bug) decodes UTF-8's 2-byte sequences as two separate wrong
# characters, producing "Ã©" instead of "é" for every UTF-8-era season.
# players_raw.csv, by contrast, IS UTF-8 in all 10 seasons -- verified the same
# way -- so it always uses "utf-8" directly, independent of this cutoff.
_MERGED_GW_LATIN1_SEASONS = {"2016-17", "2017-18", "2018-19"}


def _merged_gw_encoding(season: str) -> str:
    return "latin1" if season in _MERGED_GW_LATIN1_SEASONS else "utf-8"


def _load_players_raw(season: str) -> pd.DataFrame:
    """Full end-of-season players_raw.csv for one season, with `id` renamed to
    `element` to match merged_gw.csv's join key. Always UTF-8 -- see the
    encoding note above _MERGED_GW_LATIN1_SEASONS."""
    path = os.path.join(VAASTAV_ROOT, season, "players_raw.csv")
    return pd.read_csv(path, encoding="utf-8").rename(columns={"id": "element"})


def _load_position_team(season: str) -> pd.DataFrame:
    """End-of-season position + numeric team-id snapshot per player id, from
    players_raw.csv. Used to backfill position/team for seasons where
    merged_gw.csv lacks both columns (2016-17 to 2019-20)."""
    df = _load_players_raw(season)
    return df[["element", "element_type", "team"]].rename(columns={"team": "team_id"})


def _season_team_id_to_name(season: str) -> pd.Series:
    """A season's numeric team id -> team name mapping. Prefers teams.csv (exists
    for 2019-20 onward, including 2024-25 which master_team_list.csv lacks);
    falls back to master_team_list.csv for 2016-17 to 2018-19, the only seasons
    where teams.csv isn't present in this dataset."""
    teams_csv_path = os.path.join(VAASTAV_ROOT, season, "teams.csv")
    if os.path.exists(teams_csv_path):
        teams = pd.read_csv(teams_csv_path)
        return teams.set_index("id")["name"]

    master = pd.read_csv(_MASTER_TEAM_LIST_PATH)
    season_rows = master[master["season"] == season]
    if season_rows.empty:
        raise ValueError(
            f"{season}: no teams.csv and no master_team_list.csv row — "
            f"can't resolve team ids to names for this season."
        )
    return season_rows.set_index("team")["team_name"]


def _resolve_team_names(season: str, team_ids: pd.Series) -> pd.Series:
    """Map a season's numeric team ids to team name strings, so `team` is a
    consistent name (e.g. "Brighton") across every season, matching what
    merged_gw.csv already provides natively from 2020-21 onward."""
    season_map = _season_team_id_to_name(season)
    return team_ids.map(season_map)


def load_season(season: str) -> pd.DataFrame:
    """Load one season's gameweek data, unified to COMMON_COLUMNS + position/team,
    with team normalized to a name string in every season."""
    path = os.path.join(VAASTAV_ROOT, season, "gws", "merged_gw.csv")
    enc = _merged_gw_encoding(season)
    df = pd.read_csv(path, encoding=enc, low_memory=False)

    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{season}: merged_gw.csv is missing expected columns {missing}")

    header = pd.read_csv(path, encoding=enc, nrows=1).columns
    df = df[COMMON_COLUMNS].copy()

    # FPL's own pre-match expected-points prediction, used later as a baseline to
    # compare the trained model against. Only published from ~2020-21 onward --
    # absent (not just this dataset's coverage, but never existed) for earlier
    # seasons, so this stays NaN there rather than being backfilled or estimated.
    if "xP" in header:
        df["fpl_xP"] = pd.read_csv(path, encoding=enc, low_memory=False)["xP"]
    else:
        df["fpl_xP"] = pd.NA

    if "position" in header:
        # team is already a name string in these seasons (verified 2020-21 to
        # 2024-25) — no resolution needed.
        pos_team = pd.read_csv(path, encoding=enc, low_memory=False)[
            ["element", "position", "team"]
        ].drop_duplicates(subset="element")
        df = df.merge(pos_team, on="element", how="left")
    else:
        pos_team = _load_position_team(season)
        df = df.merge(pos_team, on="element", how="left")
        df["position"] = df["element_type"].map(POSITION_MAP)
        df["team"] = _resolve_team_names(season, df["team_id"])
        df = df.drop(columns=["element_type", "team_id"])

    # opponent_team is a numeric, season-scoped id in EVERY season (unlike `team`,
    # which is a string from 2020-21 onward) -- resolved to a name unconditionally.
    df["opponent_team"] = _resolve_team_names(season, df["opponent_team"])

    # player_code: the stable cross-season player id (element resets every season).
    codes = _load_players_raw(season)[["element", "code"]].drop_duplicates(subset="element")
    df = df.merge(codes.rename(columns={"code": "player_code"}), on="element", how="left")

    unmatched = df["player_code"].isna().sum()
    if unmatched:
        print(f"  WARNING [{season}]: {unmatched} rows have no player_code match "
              f"(player left the game mid-season and dropped from players_raw.csv?)")

    # FPL introduced a "pick a Manager" feature in 2024-25 (position == "AM"):
    # a real-life manager selectable alongside your 15 players, scored on
    # completely different rules. Not a player -- excluded from this player-
    # performance dataset entirely, not just at training time, so nothing
    # downstream (feature engineering included) has to special-case it.
    n_managers = (df["position"] == "AM").sum()
    if n_managers:
        print(f"  [{season}]: excluding {n_managers} manager rows (position == 'AM')")
        df = df[df["position"] != "AM"].reset_index(drop=True)

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


def load_all_seasons_with_live(seasons: list = None) -> pd.DataFrame:
    """load_all_seasons() (vaastav's archive, 2016-17 through 2025-26 -- the
    only source of gameweek-level data for those seasons, since FPL's public
    API only ever exposes SEASON-TOTAL aggregates for a finished season, never
    gameweek-by-gameweek history; verified directly against history_past) PLUS
    this project's own collected 2026-27+ data (load_live.py), concatenated.

    Live rows carry several real FPL fields vaastav's data structurally never
    has for any season (in_dreamteam, defensive_contribution, starts,
    expected_goals/assists/goal_involvements/conceded) -- those stay NaN on
    historical rows after concat, which is honest (they don't exist there),
    not a bug to fix. Returns just the vaastav data, unchanged, if the
    collector hasn't captured any live gameweeks yet."""
    import load_live

    historical = load_all_seasons(seasons)
    live = load_live.load_all_live_seasons()
    if live.empty:
        return historical
    combined = pd.concat([historical, live], ignore_index=True)
    print(f"Added {len(live):,} live-collected rows ({sorted(live['season'].unique())}) "
          f"on top of {len(historical):,} historical rows.")
    return combined


if __name__ == "__main__":
    import os

    df = load_all_seasons_with_live()
    print(df.shape)
    print(df["season"].value_counts().sort_index())
    print(df["position"].value_counts(dropna=False))
    print("team nulls:", df["team"].isna().sum())
    print("opponent_team nulls:", df["opponent_team"].isna().sum())
    print("player_code nulls:", df["player_code"].isna().sum())

    out_path = os.path.join("data", "processed", "historical_gw.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
