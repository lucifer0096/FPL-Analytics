"""Deterministic tests for model season and validation boundaries."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import features
import train


def test_season_sort_accepts_new_live_season():
    result = features._season_sort_key(pd.DataFrame({"season": ["2026-27", "2025-26"]}))
    assert result.sort_values("_season_rank")["season"].tolist() == ["2025-26", "2026-27"]


def test_season_sort_rejects_malformed_season():
    try:
        features._season_sort_key(pd.DataFrame({"season": ["next-season"]}))
    except ValueError as error:
        assert "Invalid season label" in str(error)
    else:
        raise AssertionError("malformed season label should be rejected")


def test_chronological_split_excludes_future_seasons():
    df = pd.DataFrame({"season": ["2023-24", "2024-25", "2025-26", "2026-27"]})
    train_df, validation_df = train.chronological_split(df)
    assert train_df["season"].tolist() == ["2023-24"]
    assert validation_df["season"].tolist() == ["2024-25"]