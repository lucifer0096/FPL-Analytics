"""Load the trained single-stage xP model and score a features table.

The single-stage model was chosen over the two-stage one -- see train.py's
module docstring: the two-stage split didn't beat it (MAE 1.001 vs 1.003),
so there's no accuracy reason to carry the extra complexity of loading and
combining two models here.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from train import FEATURE_COLUMNS, CATEGORICAL_FEATURES, prepare_x

MODEL_PATH = os.path.join("models", "xp_model_single_stage.txt")


def load_model(path: str = None) -> lgb.Booster:
    path = path or MODEL_PATH
    return lgb.Booster(model_file=path)


def predict_points(df: pd.DataFrame, model: lgb.Booster = None) -> np.ndarray:
    """Predict expected points for each row of `df`, which must already carry
    every column in FEATURE_COLUMNS (i.e. has been through
    features.build_feature_table). Missing feature values (e.g. a player with
    no rolling history yet) are left as NaN -- LightGBM handles NaN natively
    by learning a default split direction, same as during training, so no
    imputation is needed here."""
    model = model or load_model()
    X = prepare_x(df)
    return np.clip(model.predict(X), 0, None)
