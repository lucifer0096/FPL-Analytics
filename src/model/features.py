"""Feature engineering for the FPL expected-points (xP) model.

Raw per-gameweek stats (this week's goals, minutes, etc.) aren't useful as model
INPUTS on their own — they're outcomes, not predictors, and using this week's
total_points to predict this week's total_points is a leak, not a model. The
actual signal for predicting a player's NEXT gameweek score is their recent form,
whether they're getting consistent minutes, and their upcoming fixture.

All rolling/lagged features group by `player_code` (see load_historical.py) and
are ordered by (season, GW) within each player, so a rolling window never mixes
one player's history with another's, and never leaks a gameweek's own outcome
into its own feature row.
"""

import os
import pandas as pd

VAASTAV_ROOT = os.environ.get("VAASTAV_DATA_ROOT", r"E:\Fantasy-Premier-League\data")

# Season order matters for rolling windows that span a season boundary (a player's
# form entering GW1 of a new season should still reflect their last games of the
# previous one, not reset to NaN).
SEASON_ORDER = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]


def _season_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    season_rank = {s: i for i, s in enumerate(SEASON_ORDER)}
    df = df.copy()
    df["_season_rank"] = df["season"].map(season_rank)
    if df["_season_rank"].isna().any():
        unknown = df.loc[df["_season_rank"].isna(), "season"].unique()
        raise ValueError(f"Unknown season(s) not in SEASON_ORDER: {unknown}")
    return df


def add_rolling_form_features(df: pd.DataFrame, windows: list = (3, 5)) -> pd.DataFrame:
    """Add rolling-average features per player, computed ONLY from that player's
    own past gameweeks (shifted by 1, so the current row's own outcome is never
    included — avoids target leakage)."""
    df = _season_sort_key(df)
    df = df.sort_values(["player_code", "_season_rank", "GW"]).reset_index(drop=True)

    grouped = df.groupby("player_code", sort=False)

    for w in windows:
        for col in ["total_points", "minutes", "bps", "ict_index"]:
            df[f"{col}_avg_last_{w}"] = (
                grouped[col]
                .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )

    # Games played so far this season (resets at each season boundary), and
    # career gameweeks played overall — both computed from prior rows only.
    df["career_gw_count"] = grouped.cumcount()

    def _season_gw_count(g: pd.DataFrame) -> pd.Series:
        return g.groupby("season").cumcount()

    df["season_gw_count"] = (
        df.groupby("player_code", sort=False)
        .apply(_season_gw_count, include_groups=False)
        .reset_index(level=0, drop=True)
    )

    return df.drop(columns=["_season_rank"])


PRICE_BAND_EDGES = [0, 4.5, 5.5, 6.5, 8.0, 100]
PRICE_BAND_LABELS = ["budget", "low-mid", "mid", "mid-high", "premium"]


def add_new_player_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """A position x price-band baseline for players with no rolling form yet --
    newly promoted teams' players and new signings have career_gw_count == 0 and
    every total_points_avg_last_N feature is null (see build_feature_table), which
    otherwise leaves the model with no signal for a meaningful chunk of players
    every season (roughly 70-130 a season, excluding this dataset's own first
    season where the whole league is naturally "new"). No Championship/lower-
    league data source is used here -- this is a simple, honest fallback: what did
    similarly priced players in the same position score on average, using ONLY
    gameweeks already played before this one across the whole league.

    Leakage-safe by construction: computed as an expanding mean per (season,
    position, price_band) ordered by GW, shifted by one GW boundary so a
    gameweek's own results are never included in its own baseline -- this is a
    league-wide statistic, not a per-player one, so the shift happens at the
    (season, position, price_band, GW) group level, then the resulting one-row-
    per-group baseline is merged back onto every player row in that group."""
    df = df.copy()
    df["price_band"] = pd.cut(
        df["value"] / 10.0, bins=PRICE_BAND_EDGES, labels=PRICE_BAND_LABELS
    )

    group_cols = ["season", "position", "price_band", "GW"]
    gw_group_avg = (
        df.groupby(group_cols, observed=True)["total_points"]
        .mean()
        .reset_index()
        .rename(columns={"total_points": "gw_avg"})
    )

    gw_group_avg = _season_sort_key(gw_group_avg).sort_values(
        ["position", "price_band", "_season_rank", "GW"]
    )
    gw_group_avg["new_player_baseline"] = (
        gw_group_avg.groupby(["position", "price_band"], observed=True)["gw_avg"]
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    baseline_lookup = gw_group_avg[group_cols + ["new_player_baseline"]]
    return df.merge(baseline_lookup, on=group_cols, how="left").drop(columns=["price_band"])


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flags a player's recent minutes trend — a player who's barely played the
    last few gameweeks is a materially different bet than one on a run of full
    90s, independent of their season-long average."""
    df = _season_sort_key(df)
    df = df.sort_values(["player_code", "_season_rank", "GW"]).reset_index(drop=True)
    grouped = df.groupby("player_code", sort=False)

    df["started_last_gw"] = grouped["minutes"].shift(1).fillna(0).gt(0).astype(int)
    df["minutes_last_gw"] = grouped["minutes"].shift(1)

    return df.drop(columns=["_season_rank"])


def _build_team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, GW, team) with that match's goals for/against, derived
    from team_a_score/team_h_score + was_home. These are POST-match results, so
    they're only used here to build a lagged team-form feature — never joined in
    directly as a same-match feature (that would leak the match outcome).

    KNOWN LIMITATION: for 2016-17 to 2019-20, `team` is backfilled from
    players_raw.csv's END-OF-SEASON snapshot (see load_historical.py), so a player
    who transferred mid-season shows their final club for every gameweek,
    including games actually played for a different club earlier that season.
    This can produce two DIFFERENT score rows for the same (season, GW, team) --
    e.g. a player who ended the season at Club A but played their GW1 game for
    Club B shows Club B's GW1 opponent/score under Club A's row. A team genuinely
    can only play one match per gameweek, so any (season, GW, team) with more than
    one distinct score is a transfer-driven artifact, not a real double-fixture --
    dropped here rather than silently corrupting the rolling average with
    contradictory results. This affects a small minority of rows (players who
    moved clubs mid-season) and only in the 4 seasons without a native `team`
    column; 2020-21 onward is unaffected."""
    matches = df[
        ["season", "GW", "team", "opponent_team", "was_home", "team_a_score", "team_h_score"]
    ].drop_duplicates(subset=["season", "GW", "team", "opponent_team", "was_home",
                               "team_a_score", "team_h_score"])

    ambiguous = matches.duplicated(subset=["season", "GW", "team"], keep=False)
    if ambiguous.any():
        n_dropped = ambiguous.sum()
        print(f"  Dropping {n_dropped} team-match rows with contradictory scores "
              f"for the same (season, GW, team) — see _build_team_match_table docstring.")
        matches = matches[~ambiguous]

    matches = matches.copy()
    matches["goals_for"] = matches["team_h_score"].where(
        matches["was_home"], matches["team_a_score"]
    )
    matches["goals_against"] = matches["team_a_score"].where(
        matches["was_home"], matches["team_h_score"]
    )
    return matches[["season", "GW", "team", "goals_for", "goals_against"]]


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling team-level goals-for/goals-against form, as a fixture-difficulty
    proxy that works uniformly across all 9 seasons (teams.csv's own strength
    ratings only exist from 2019-20 onward). Computed once per team-match, shifted
    by 1 so a match's own result never leaks into its own row, then joined back
    onto the player table twice: once for the player's own team ("team_form_*")
    and once for their opponent ("opponent_form_*")."""
    df = _season_sort_key(df)
    team_matches = _build_team_match_table(df)

    team_season_rank = {s: i for i, s in enumerate(SEASON_ORDER)}
    team_matches = team_matches.copy()
    team_matches["_season_rank"] = team_matches["season"].map(team_season_rank)
    team_matches = team_matches.sort_values(["team", "_season_rank", "GW"]).reset_index(drop=True)

    grouped = team_matches.groupby("team", sort=False)
    for col in ["goals_for", "goals_against"]:
        team_matches[f"{col}_avg_last_5"] = (
            grouped[col]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    team_matches = team_matches.drop(columns=["_season_rank", "goals_for", "goals_against"])

    df = df.merge(
        team_matches.rename(columns={
            "team": "team",
            "goals_for_avg_last_5": "team_form_goals_for",
            "goals_against_avg_last_5": "team_form_goals_against",
        }),
        on=["season", "GW", "team"],
        how="left",
    )
    df = df.merge(
        team_matches.rename(columns={
            "team": "opponent_team",
            "goals_for_avg_last_5": "opponent_form_goals_for",
            "goals_against_avg_last_5": "opponent_form_goals_against",
        }),
        on=["season", "GW", "opponent_team"],
        how="left",
    )

    return df.drop(columns=["_season_rank"])


FIXTURES_AVAILABLE_SEASONS = {
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
}


def _load_fixture_difficulty(season: str) -> pd.DataFrame:
    """FPL's own published fixture-difficulty rating (1-5) per match, from
    fixtures.csv. Not available for 2016-17/2017-18 in this dataset. Unlike the
    team-form proxy below, this is known ahead of the match (FPL publishes these
    before kickoff), so no shift/lag is needed — joining it in isn't a leak."""
    path = os.path.join(VAASTAV_ROOT, season, "fixtures.csv")
    fx = pd.read_csv(path)[["id", "team_h_difficulty", "team_a_difficulty"]]
    return fx.rename(columns={"id": "fixture"})


def add_fixture_difficulty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `fixture_difficulty`: the difficulty of the CURRENT player's team in
    this fixture (their home rating if was_home, else their away rating). Falls
    back to the team-form proxy (opponent_form_goals_for/against, already built by
    add_team_form_features) for 2016-17/2017-18, where FPL's own rating doesn't
    exist in this dataset. Must run after add_team_form_features."""
    frames = []
    for season in df["season"].unique():
        sub = df[df["season"] == season].copy()
        if season in FIXTURES_AVAILABLE_SEASONS:
            fx = _load_fixture_difficulty(season)
            sub = sub.merge(fx, on="fixture", how="left")
            sub["fixture_difficulty"] = sub["team_h_difficulty"].where(
                sub["was_home"], sub["team_a_difficulty"]
            )
            sub = sub.drop(columns=["team_h_difficulty", "team_a_difficulty"])
        else:
            # Fallback for 2016-17/2017-18, where FPL's own rating doesn't exist
            # in this dataset: approximate difficulty from the opponent's recent
            # attacking form (opponent_form_goals_for). Rougher than FPL's own
            # rating (which also weighs defense, home advantage, and non-form
            # factors) but keeps the full 9-season window intact rather than
            # leaving these two seasons with no difficulty signal at all.
            #
            # Rescaled to the same 1-5 range as FPL's own rating via equal-
            # frequency quantile bins — the raw proxy has a different scale
            # entirely (mean ~1.4 on a roughly 0-5 range vs FPL's own ratings
            # averaging ~2.9), so leaving it unscaled would mean the same numeric
            # value represents a very different fixture difficulty depending on
            # which era of data a training row came from.
            sub["fixture_difficulty"] = pd.qcut(
                sub["opponent_form_goals_for"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            ).astype("Float64")
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline. `total_points` (this row's own outcome) is
    kept as the training target — every added feature column is a lagged/rolling
    stat that only uses information available before this gameweek was played."""
    df = add_rolling_form_features(df)
    df = add_availability_features(df)
    df = add_team_form_features(df)
    df = add_fixture_difficulty_features(df)
    df = add_new_player_baseline(df)
    return df


if __name__ == "__main__":
    import os

    in_path = os.path.join("data", "processed", "historical_gw.parquet")
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows")

    featured = build_feature_table(df)
    print(f"Built features, {featured.shape[1]} columns")

    # Sanity check: no leakage — a feature for GW n must never equal that same
    # row's own outcome for a player who only has one gameweek of history.
    first_gw = featured[featured["career_gw_count"] == 0]
    still_has_signal = first_gw["total_points_avg_last_3"].notna().sum()
    print(f"Rows with career_gw_count==0 that still have a rolling average "
          f"(should be 0 — would indicate leakage): {still_has_signal}")

    out_path = os.path.join("data", "processed", "features.parquet")
    featured.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
