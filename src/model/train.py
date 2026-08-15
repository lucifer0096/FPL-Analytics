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

TWO-STAGE MODEL: ~64% of validation rows are players who didn't play at all that
gameweek (minutes == 0) -- tried splitting into a play classifier (stage 1) and a
points-conditional-on-playing regressor (stage 2), on the hypothesis that a single
regressor was spending its error budget distinguishing "benched" from "played".
RESULT: this hypothesis didn't hold. The two-stage model (MAE 1.001) barely beat
the single-stage one (MAE 1.003) -- the single model was already implicitly
learning the play/didn't-play distinction well via minutes_avg_last_5/
minutes_last_gw/started_last_gw, so splitting it out explicitly added no new
signal. Diagnosis (see README's Model Training section): restricting the
comparison to rows where the player actually played shows both models land much
closer to FPL's xP (single-stage MAE 1.84 vs FPL 1.76, played-only) than the
full-dataset numbers suggest -- most of the overall gap to FPL's baseline is
concentrated in the non-playing rows, where FPL's xP likely has access to
real injury/team-news signals this project's historical-stats-only feature set
can't replicate. Kept in the pipeline as a documented negative result, not
removed -- both models are trained and reported on every run for comparison.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

VALIDATION_SEASON = "2024-25"
FINAL_HOLDOUT_SEASON = "2025-26"  # never touched during training/tuning
EXCLUDED_SEASONS = [FINAL_HOLDOUT_SEASON]

TARGET_COLUMN = "total_points"
PLAYED_COLUMN = "played"  # derived: minutes > 0

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
    "new_player_baseline",
]

CATEGORICAL_FEATURES = ["position"]


def load_training_data(path: str = None) -> pd.DataFrame:
    path = path or os.path.join("data", "processed", "features.parquet")
    df = pd.read_parquet(path)
    df = df[~df["season"].isin(EXCLUDED_SEASONS)].copy()
    df[PLAYED_COLUMN] = (df["minutes"] > 0).astype(int)
    return df


def chronological_split(df: pd.DataFrame) -> tuple:
    """Everything before VALIDATION_SEASON trains the model; VALIDATION_SEASON
    itself is held out entirely for evaluation."""
    train = df[df["season"] != VALIDATION_SEASON].copy()
    val = df[df["season"] == VALIDATION_SEASON].copy()
    return train, val


def prepare_x(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}")
    return {"mae": mae, "rmse": rmse}


METRICS_PATH = os.path.join("models", "metrics.json")


# ============================================================================
# Single-stage model (predicts total_points directly for every row)
# ============================================================================

def train_single_stage(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
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


# ============================================================================
# Two-stage model: P(plays) classifier + E[points | plays] regressor
# ============================================================================

def train_play_classifier(X_train: pd.DataFrame, played_train: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, played_train, categorical_feature=CATEGORICAL_FEATURES)
    return model


def train_points_given_played(X_train_played: pd.DataFrame, y_train_played: pd.Series) -> lgb.LGBMRegressor:
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
    model.fit(X_train_played, y_train_played, categorical_feature=CATEGORICAL_FEATURES)
    return model


if __name__ == "__main__":
    import json

    df = load_training_data()
    print(f"Loaded {len(df):,} rows (after excluding {EXCLUDED_SEASONS})")

    train_df, val_df = chronological_split(df)
    print(f"Train: {len(train_df):,} rows (seasons before {VALIDATION_SEASON})")
    print(f"Validation: {len(val_df):,} rows (season {VALIDATION_SEASON})")

    X_train = prepare_x(train_df)
    X_val = prepare_x(val_df)
    y_train = train_df[TARGET_COLUMN].astype(float)
    y_val = val_df[TARGET_COLUMN].astype(float)

    metrics = {"validation_season": VALIDATION_SEASON, "final_holdout_season": FINAL_HOLDOUT_SEASON}

    # ---- Single-stage model ----
    print("\nTraining single-stage model...")
    single_model = train_single_stage(X_train, y_train)
    single_pred = np.clip(single_model.predict(X_val), 0, None)

    print("\n=== Validation results (season", VALIDATION_SEASON, ") ===")
    metrics["single_stage"] = evaluate("Single-stage model", y_val, single_pred)

    # ---- Two-stage model ----
    print("\nTraining two-stage model (play classifier + conditional points)...")
    played_train = train_df[PLAYED_COLUMN]
    play_clf = train_play_classifier(X_train, played_train)

    played_mask_train = played_train == 1
    points_model = train_points_given_played(
        X_train[played_mask_train], y_train[played_mask_train]
    )

    play_proba_val = play_clf.predict_proba(X_val)[:, 1]
    points_given_played_val = np.clip(points_model.predict(X_val), 0, None)
    two_stage_pred = play_proba_val * points_given_played_val

    play_auc = roc_auc_score(val_df[PLAYED_COLUMN], play_proba_val)
    print(f"Play classifier AUC: {play_auc:.3f}")
    metrics["two_stage"] = evaluate("Two-stage model (P(plays) x E[points|plays])", y_val, two_stage_pred)
    metrics["two_stage"]["play_classifier_auc"] = play_auc

    # ---- Baselines ----
    naive_pred = val_df["total_points_avg_last_5"].fillna(0).clip(lower=0)
    metrics["naive_baseline"] = evaluate("Naive baseline (player's own rolling-5 average)", y_val, naive_pred)

    has_baseline = val_df["fpl_xP"].notna()
    n_baseline = has_baseline.sum()
    if n_baseline > 0:
        metrics["fpl_xp_baseline"] = evaluate(
            f"FPL's own xP, own leakage caveat noted by data source "
            f"(n={n_baseline:,}/{len(val_df):,})",
            y_val[has_baseline], val_df.loc[has_baseline, "fpl_xP"].astype(float)
        )
        metrics["fpl_xp_baseline"]["n_rows"] = int(n_baseline)
        metrics["fpl_xp_baseline"]["n_rows_total"] = int(len(val_df))
        evaluate(
            f"Single-stage model (same {n_baseline:,} rows, for direct comparison)",
            y_val[has_baseline], single_pred[has_baseline.values]
        )
        evaluate(
            f"Two-stage model (same {n_baseline:,} rows, for direct comparison)",
            y_val[has_baseline], two_stage_pred[has_baseline.values]
        )
    else:
        print(f"No fpl_xP available for {VALIDATION_SEASON} — can't compare to baseline.")

    # Diagnostic: how much of the gap to FPL's xP is concentrated in rows where
    # the player didn't play at all (minutes == 0), vs. rows where they did?
    played_mask = val_df[PLAYED_COLUMN] == 1
    print(f"\n=== Diagnostic: error on PLAYED rows only (n={played_mask.sum():,}/{len(val_df):,}) ===")
    metrics["single_stage_played_only"] = evaluate(
        "Single-stage model (played only)", y_val[played_mask], single_pred[played_mask.values]
    )
    metrics["naive_baseline_played_only"] = evaluate(
        "Naive baseline (played only)", y_val[played_mask], naive_pred[played_mask]
    )
    fx_played = has_baseline & played_mask
    if fx_played.sum() > 0:
        metrics["fpl_xp_baseline_played_only"] = evaluate(
            "FPL's own xP (played only)", y_val[fx_played], val_df.loc[fx_played, "fpl_xP"].astype(float)
        )

    print("\n=== Feature importance (single-stage model) ===")
    importance = pd.Series(single_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importance.to_string())

    print("\n=== Feature importance (play classifier) ===")
    clf_importance = pd.Series(play_clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(clf_importance.to_string())

    os.makedirs("models", exist_ok=True)
    single_model.booster_.save_model(os.path.join("models", "xp_model_single_stage.txt"))
    play_clf.booster_.save_model(os.path.join("models", "xp_model_play_classifier.txt"))
    points_model.booster_.save_model(os.path.join("models", "xp_model_points_given_played.txt"))
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved 3 model files and metrics.json to models/")
