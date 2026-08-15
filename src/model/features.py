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

import pandas as pd

# Season order matters for rolling windows that span a season boundary (a player's
# form entering GW1 of a new season should still reflect their last games of the
# previous one, not reset to NaN).
SEASON_ORDER = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
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


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline. `total_points` (this row's own outcome) is
    kept as the training target — every added feature column is a lagged/rolling
    stat that only uses information available before this gameweek was played."""
    df = add_rolling_form_features(df)
    df = add_availability_features(df)
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
