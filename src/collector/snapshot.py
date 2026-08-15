"""Snapshot the current FPL season's data to disk.

Designed to run on a frequent, cheap schedule (e.g. daily) rather than a fixed
"weekly" cron time, since gameweeks don't land on a fixed day — fixtures get
rearranged, some gameweeks span midweek, and blank/double gameweeks skip or double
up entirely. Instead, each run checks the FPL API's own gameweek-completion flags
and only does the expensive part (fetching all 587+ players' histories) when a
gameweek has actually finished and been data-checked since the last successful run.

Usage:
    python snapshot.py               # normal run: check, and snapshot if needed
    python snapshot.py --check-only  # print whether a snapshot is needed, don't run one
    python snapshot.py --force       # always do a full snapshot, ignoring last-run state
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpl_api

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")
STATE_PATH = os.path.join(RAW_DIR, "collector_state.json")

# Set FPL_ENTRY_ID to snapshot a specific manager's team history/picks alongside the
# league-wide data. Leave unset to skip that step.
ENTRY_ID = os.environ.get("FPL_ENTRY_ID")


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"last_snapshotted_gw": 0}


def _save_state(state: dict) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _latest_data_checked_gw(bootstrap: dict) -> int:
    """Highest gameweek id that's both finished and data-checked (bonus points and
    stats finalized — these can still change for a day or two after 'finished').
    Returns 0 if none yet."""
    checked_gws = [e["id"] for e in bootstrap["events"] if e["finished"] and e["data_checked"]]
    return max(checked_gws, default=0)


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

    # Real name (player_first_name/player_last_name) and team name -- fetched
    # separately from history/picks below since it's a different endpoint
    # (entry/{id}/, not entry/{id}/history/), saved once here so the
    # dashboard can display "Rahul Bhaskaran" instead of a bare numeric id.
    entry_info = fpl_api.get_entry(entry_id)
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(entry_info, f, indent=2)
    print(f"Saved entry {entry_id} info "
          f"({entry_info.get('player_first_name')} {entry_info.get('player_last_name')})")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print whether a new gameweek is ready to snapshot, then exit without doing one."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Always run a full snapshot, ignoring the last-run state."
    )
    args = parser.parse_args()

    print("Fetching bootstrap-static (cheap check)...")
    bootstrap = fpl_api.get_bootstrap_static()
    season = _season_label(bootstrap)
    latest_checked_gw = _latest_data_checked_gw(bootstrap)

    state = _load_state()
    last_snapshotted_gw = state.get("last_snapshotted_gw", 0)

    needs_snapshot = args.force or latest_checked_gw > last_snapshotted_gw
    print(f"Season: {season} | latest data-checked GW: {latest_checked_gw} | "
          f"last snapshotted GW: {last_snapshotted_gw} | needs snapshot: {needs_snapshot}")

    if args.check_only:
        sys.exit(0 if needs_snapshot else 1)

    if not needs_snapshot:
        print("Nothing new since the last snapshot — skipping the full run.")
        return

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

    _save_state({"last_snapshotted_gw": latest_checked_gw, "season": season})
    print(f"Updated collector state: last_snapshotted_gw={latest_checked_gw}")


if __name__ == "__main__":
    main()
