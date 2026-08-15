"""Train the FPL expected-points (xP) model.

Uses a chronological train/validation split (never random) -- this is time-series
data, and a random split would let the model "see the future" (train on gameweek
20 while validating on gameweek 10 of the same season). The validation set is
2024-25 (the most recent complete season at model-build time); everything before
it trains the model. 2025-26 is held out entirely, untouched by any training or
tuning decision, as a true final check once the model is otherwise finalized.

Only a fixed, explicit allowlist of pre-match-known columns is used as model
input (FEATURE_COLUMNS below) -- deliberately an allowlist, not "every column
except a few known-bad ones", so a new leaky column added to the feature table
later doesn't silently become a model input.

CAVEAT on the FPL xP baseline: the upstream data source's own maintainer warns
that `xP` (scraped from FPL's `ep_this` field) may contain post-match information
for some gameweeks, since the scraper runs after each gameweek ends and FPL's
update cadence for that field isn't documented. This makes it an uncertain, not
fully leak-free, baseline -- report it as "FPL's published xP, own leakage
caveat noted by the data source" rather than a guaranteed clean pre-match target.
A genuinely leak-free naive baseline (the player's own rolling average, already
a model feature) is reported alongside it for a trustworthy comparison point.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

VALIDATION_SEASON = "2024-25"
FINAL_HOLDOUT_SEASON = "2025-26"  # never touched during training/tuning
EXCLUDED_SEASONS = [FINAL_HOLDOUT_SEASON]

TARGET_COLUMN = "total_points"

# Every one of these is either a rolling/lagged stat (shifted by 1 gameweek in
# features.py, so it only reflects information available BEFORE this gameweek was
# played) or a pre-match-known fact (fixture_difficulty, was_home, position,
# season/career gameweek counts). None of them can see this row's own outcome.
FEATURE_COLUMNS = [
    "was_home",
    "position",
    "season_gw_count",
    "career_gw_count",
    "started_last_gw",
    "minutes_last_gw",
    "total_points_avg_last_3",
    "minutes_avg_last_3",
    "bps_avg_last_3",
    "ict_index_avg_last_3",
    "total_points_avg_last_5",
    "minutes_avg_last_5",
    "bps_avg_last_5",
    "ict_index_avg_last_5",
    "team_form_goals_for",
    "team_form_goals_against",
    "opponent_form_goals_for",
    "opponent_form_goals_against",
    "fixture_difficulty",
]

CATEGORICAL_FEATURES = ["position"]


def load_training_data(path: str = None) -> pd.DataFrame:
    path = path or os.path.join("data", "processed", "features.parquet")
    df = pd.read_parquet(path)
    df = df[~df["season"].isin(EXCLUDED_SEASONS)].copy()
    return df


def chronological_split(df: pd.DataFrame) -> tuple:
    """Everything before VALIDATION_SEASON trains the model; VALIDATION_SEASON
    itself is held out entirely for evaluation."""
    train = df[df["season"] != VALIDATION_SEASON].copy()
    val = df[df["season"] == VALIDATION_SEASON].copy()
    return train, val


def prepare_xy(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df[TARGET_COLUMN].astype(float)
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    return model


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}")
    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    df = load_training_data()
    print(f"Loaded {len(df):,} rows (after excluding {EXCLUDED_SEASONS})")

    train_df, val_df = chronological_split(df)
    print(f"Train: {len(train_df):,} rows (seasons before {VALIDATION_SEASON})")
    print(f"Validation: {len(val_df):,} rows (season {VALIDATION_SEASON})")

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    print("\nTraining LightGBM model...")
    model = train_model(X_train, y_train)

    val_pred = model.predict(X_val)
    val_pred_clipped = np.clip(val_pred, 0, None)  # points can't be negative

    print("\n=== Validation results (season", VALIDATION_SEASON, ") ===")
    model_metrics = evaluate("Trained model", y_val, val_pred_clipped)

    # Clean baseline: the player's own rolling 5-gameweek average, already a model
    # feature. Genuinely leak-free (shifted by 1 in features.py), so this is a
    # trustworthy floor the model should be expected to beat.
    naive_pred = val_df["total_points_avg_last_5"].fillna(0).clip(lower=0)
    evaluate("Naive baseline (player's own rolling-5 average)", y_val, naive_pred)

    # Uncertain baseline: FPL's own published xP. See the CAVEAT in this module's
    # docstring -- the data source's maintainer warns this column may contain
    # post-match information for some gameweeks, so treat this comparison as
    # informative but not a guaranteed-clean pre-match target.
    has_baseline = val_df["fpl_xP"].notna()
    n_baseline = has_baseline.sum()
    if n_baseline > 0:
        evaluate(
            f"FPL's own xP, own leakage caveat noted by data source "
            f"(n={n_baseline:,}/{len(val_df):,})",
            y_val[has_baseline], val_df.loc[has_baseline, "fpl_xP"].astype(float)
        )
        # Compare the model against the baseline on the SAME rows, not the full
        # validation set -- an apples-to-apples comparison.
        evaluate(
            f"Trained model (same {n_baseline:,} rows, for direct comparison)",
            y_val[has_baseline], val_pred_clipped[has_baseline.values]
        )
    else:
        print(f"No fpl_xP available for {VALIDATION_SEASON} — can't compare to baseline.")

    print("\n=== Feature importance ===")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importance.to_string())

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "xp_model.txt")
    model.booster_.save_model(model_path)
    print(f"\nSaved model to {model_path}")
