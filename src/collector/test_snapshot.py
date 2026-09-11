"""Deterministic tests for collector state decisions."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import snapshot


def test_latest_live_gameweek_includes_current_event():
    bootstrap = {
        "events": [
            {"id": 1, "finished": True, "data_checked": True, "is_current": False},
            {"id": 2, "finished": False, "data_checked": False, "is_current": True},
        ]
    }
    assert snapshot._latest_live_gw(bootstrap) == 2


def test_latest_data_checked_gameweek_excludes_unchecked_current_event():
    bootstrap = {
        "events": [
            {"id": 1, "finished": True, "data_checked": True},
            {"id": 2, "finished": True, "data_checked": False},
        ]
    }
    assert snapshot._latest_data_checked_gw(bootstrap) == 1


def test_season_label_uses_first_deadline_year():
    bootstrap = {"events": [{"deadline_time": "2026-08-21T17:30:00Z"}]}
    assert snapshot._season_label(bootstrap) == "2026-27"
