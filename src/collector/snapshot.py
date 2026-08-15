"""Snapshot the current FPL season's data to disk.

Run this periodically (e.g. weekly, after each gameweek's matches finish) to build
up a local history of player performance, since the live API only exposes the
current state, not a queryable history of past API pulls.

Usage:
    python snapshot.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpl_api

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")

# Set FPL_ENTRY_ID to snapshot a specific manager's team history/picks alongside the
# league-wide data. Leave unset to skip that step.
ENTRY_ID = os.environ.get("FPL_ENTRY_ID")


def _season_label(bootstrap: dict) -> str:
    """Infer a season label like '2025-26' from the bootstrap gameweek deadlines."""
    events = bootstrap["events"]
    first_deadline = events[0]["deadline_time"]
    year = int(first_deadline[:4])
    return f"{year}-{str(year + 1)[2:]}"


def snapshot_bootstrap(bootstrap: dict, season: str) -> str:
    """Save the players/teams/gameweek snapshot as-is (JSON), timestamped."""
    out_dir = os.path.join(RAW_DIR, season, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"bootstrap_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap, f)
    return out_path


def snapshot_fixtures(fixtures: list, season: str) -> str:
    out_dir = os.path.join(RAW_DIR, season)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fixtures.csv")
    if not fixtures:
        return out_path
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fixtures[0].keys())
        writer.writeheader()
        writer.writerows(fixtures)
    return out_path


def snapshot_gameweek_stats(bootstrap: dict, season: str) -> str:
    """Write one row per player per current finished gameweek, using each player's
    element-summary history. This is the core file the xP model trains on."""
    players = bootstrap["elements"]
    player_ids = [p["id"] for p in players]

    print(f"Fetching element-summary for {len(player_ids)} players...")
    summaries = fpl_api.get_all_player_summaries(player_ids)

    rows = []
    for player in players:
        pid = player["id"]
        history = summaries[pid].get("history", [])
        for gw_row in history:
            row = dict(gw_row)
            row["player_id"] = pid
            row["web_name"] = player["web_name"]
            row["element_type"] = player["element_type"]
            rows.append(row)

    out_dir = os.path.join(RAW_DIR, season)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gw_history.csv")

    if not rows:
        print("No finished gameweeks yet — nothing to write.")
        return out_path

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} player-gameweek rows to {out_path}")
    return out_path


def snapshot_entry(entry_id: int, season: str, current_gw: int) -> None:
    """Snapshot one manager's team history (season totals for past seasons, GW-by-GW
    for the current one) and their picks for every finished gameweek this season."""
    out_dir = os.path.join(RAW_DIR, season, "entry", str(entry_id))
    os.makedirs(out_dir, exist_ok=True)

    history = fpl_api.get_entry_history(entry_id)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved entry {entry_id} history "
          f"({len(history.get('past', []))} past seasons, "
          f"{len(history.get('current', []))} gameweeks this season)")

    picks_dir = os.path.join(out_dir, "picks")
    os.makedirs(picks_dir, exist_ok=True)
    saved = 0
    for gw in range(1, current_gw + 1):
        picks = fpl_api.get_entry_picks(entry_id, gw)
        if picks is None:
            continue
        with open(os.path.join(picks_dir, f"gw{gw}.json"), "w", encoding="utf-8") as f:
            json.dump(picks, f, indent=2)
        saved += 1
    print(f"Saved picks for {saved} finished gameweek(s)")


def main():
    print("Fetching bootstrap-static...")
    bootstrap = fpl_api.get_bootstrap_static()
    season = _season_label(bootstrap)
    print(f"Season detected: {season}")

    bootstrap_path = snapshot_bootstrap(bootstrap, season)
    print(f"Saved bootstrap snapshot: {bootstrap_path}")

    print("Fetching fixtures...")
    fixtures = fpl_api.get_fixtures()
    fixtures_path = snapshot_fixtures(fixtures, season)
    print(f"Saved fixtures: {fixtures_path}")

    snapshot_gameweek_stats(bootstrap, season)

    if ENTRY_ID:
        finished_gws = sum(1 for e in bootstrap["events"] if e["finished"])
        print(f"Snapshotting entry {ENTRY_ID} (finished gameweeks so far: {finished_gws})...")
        snapshot_entry(int(ENTRY_ID), season, finished_gws)
    else:
        print("FPL_ENTRY_ID not set — skipping manager entry snapshot.")


if __name__ == "__main__":
    main()
