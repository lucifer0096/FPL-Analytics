"""Regenerate every data/dashboard_*.json fallback file from whatever
data/raw/ the collector (snapshot.py) most recently produced.

Why this exists: data/raw/ is gitignored (personal manager picks/history
live alongside league-wide snapshots there), so a fresh deploy (Streamlit
Cloud) has none of it. The dashboard falls back to a handful of
deliberately-committed, non-timestamped copies instead -- see each
data/dashboard_*.json file's corresponding loader docstring in app/app.py
for exactly which fields each one needs and why a stale-by-a-day copy is an
acceptable tradeoff (matches the staleness this project already accepts for
data/dashboard_bootstrap.json's prices).

Run this AFTER src/collector/snapshot.py in the same working directory, then
commit the resulting data/dashboard_*.json files. The scheduled GitHub
Actions workflow (.github/workflows/weekly-collector.yml) does both steps
and commits the result automatically -- this script is what makes that
possible without duplicating fallback-generation logic in the workflow YAML
itself.

This script intentionally does NOT commit personal data beyond what's
already treated as public by this project (see each fallback's docstring in
app/app.py): the manager's real name, historical points/rank, current
squad/points, and mini-league standings are already published on this
project's own GitHub Pages manager-history page, linked throughout the
dashboard.
"""

import glob
import json
import os
import sys

RAW_DIR = os.path.join("data", "raw")
DATA_DIR = "data"

ENTRY_ID = os.environ.get("FPL_ENTRY_ID")


def _latest(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


def _current_season_dir() -> str:
    """The most recent season directory data/raw/ actually has (e.g.
    "2026-27"), found by globbing RAW_DIR/* rather than a hardcoded
    literal -- fixes a real bug found in an audit: refresh_current_squad(),
    refresh_fixtures(), and refresh_leagues() used to hardcode "2026-27"
    while refresh_bootstrap() correctly globbed RAW_DIR/* -- the three
    hardcoded ones would have silently pointed at a nonexistent directory
    (just a non-fatal "skipping" print, no error) the day the season rolls
    over to 2027-28. Returns None if data/raw/ has no season directories at
    all yet (expected before the collector's first run)."""
    season_dirs = sorted(d for d in glob.glob(os.path.join(RAW_DIR, "*")) if os.path.isdir(d))
    return os.path.basename(season_dirs[-1]) if season_dirs else None


def refresh_bootstrap() -> bool:
    path = _latest(os.path.join(RAW_DIR, "*", "bootstrap", "bootstrap_*.json"))
    if not path:
        print("No bootstrap snapshot found -- skipping dashboard_bootstrap.json")
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out_path = os.path.join(DATA_DIR, "dashboard_bootstrap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Refreshed {out_path} from {path}")
    return True


def refresh_entry_info_and_history(entry_id: str) -> bool:
    info_path = _latest(os.path.join(RAW_DIR, "*", "entry", entry_id, "info.json"))
    history_path = _latest(os.path.join(RAW_DIR, "*", "entry", entry_id, "history.json"))
    refreshed = False

    if info_path:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        out_path = os.path.join(DATA_DIR, "dashboard_entry_info.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": info.get("id"),
                "player_first_name": info.get("player_first_name"),
                "player_last_name": info.get("player_last_name"),
            }, f, indent=2)
        print(f"Refreshed {out_path} from {info_path}")
        refreshed = True

    if history_path:
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        out_path = os.path.join(DATA_DIR, "dashboard_entry_history.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Refreshed {out_path} from {history_path}")
        refreshed = True

    return refreshed


def refresh_current_squad(entry_id: str) -> bool:
    """Bundles this manager's most recent gameweek's picks + that
    gameweek's live per-player points into ONE fallback file (rather than
    two separate ones) since app.py's load_current_squad_picks/
    load_live_gw_points always need both together to render a squad."""
    season = _current_season_dir()
    if season is None:
        print("No season directory found -- skipping dashboard_current_squad.json")
        return False
    picks_dir = os.path.join(RAW_DIR, season, "entry", entry_id, "picks")
    if not os.path.isdir(picks_dir):
        print("No picks directory found -- skipping dashboard_current_squad.json")
        return False

    gws = sorted(
        int(f[2:-5]) for f in os.listdir(picks_dir) if f.startswith("gw") and f.endswith(".json")
    )
    if not gws:
        print("No picks files found -- skipping dashboard_current_squad.json")
        return False

    latest_gw = gws[-1]
    with open(os.path.join(picks_dir, f"gw{latest_gw}.json"), encoding="utf-8") as f:
        picks = json.load(f)

    live_path = os.path.join(RAW_DIR, season, "live", f"gw{latest_gw}.json")
    live_points = {}
    if os.path.exists(live_path):
        with open(live_path, encoding="utf-8") as f:
            live = json.load(f)
        live_points = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}

    out_path = os.path.join(DATA_DIR, "dashboard_current_squad.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gw": latest_gw, "picks": picks, "live_points": live_points}, f, indent=2)
    print(f"Refreshed {out_path} (GW{latest_gw})")
    return True


def refresh_fixtures() -> bool:
    """Copies data/raw/{current season}/fixtures.csv straight to
    data/dashboard_fixtures.csv (not glob-latest like the others -- there's
    only ever one live fixtures.csv per season, re-fetched in place, not a
    new timestamped file per run). data/raw/ is gitignored, so without this
    a fresh Streamlit Cloud deploy has no fixtures data at all -- silently
    breaking the PL Table tab and every fixture-difficulty feature (see
    shared.py's _fixtures_path()) until the next scheduled collector run
    happens to also be a same-day deploy."""
    season = _current_season_dir()
    if season is None:
        print("No season directory found -- skipping dashboard_fixtures.csv")
        return False
    path = os.path.join(RAW_DIR, season, "fixtures.csv")
    if not os.path.exists(path):
        print(f"No {season} fixtures.csv found -- skipping dashboard_fixtures.csv")
        return False
    with open(path, encoding="utf-8") as f:
        contents = f.read()
    out_path = os.path.join(DATA_DIR, "dashboard_fixtures.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(contents)
    print(f"Refreshed {out_path} from {path}")
    return True


def refresh_leagues(entry_id: str) -> bool:
    season = _current_season_dir()
    if season is None:
        print("No season directory found -- skipping dashboard_leagues.json")
        return False
    leagues_dir = os.path.join(RAW_DIR, season, "entry", entry_id, "leagues")
    if not os.path.isdir(leagues_dir):
        print("No leagues directory found -- skipping dashboard_leagues.json")
        return False

    leagues = []
    for fname in sorted(os.listdir(leagues_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(leagues_dir, fname), encoding="utf-8") as f:
            leagues.append(json.load(f))

    if not leagues:
        print("No league files found -- skipping dashboard_leagues.json")
        return False

    out_path = os.path.join(DATA_DIR, "dashboard_leagues.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(leagues, f, indent=2)
    print(f"Refreshed {out_path} ({len(leagues)} league(s))")
    return True


def main():
    any_refreshed = refresh_bootstrap()
    any_refreshed = refresh_fixtures() or any_refreshed
    if ENTRY_ID:
        any_refreshed = refresh_entry_info_and_history(ENTRY_ID) or any_refreshed
        any_refreshed = refresh_current_squad(ENTRY_ID) or any_refreshed
        any_refreshed = refresh_leagues(ENTRY_ID) or any_refreshed
    else:
        print("FPL_ENTRY_ID not set -- skipping entry/squad/league fallbacks.")

    sys.exit(0 if any_refreshed else 1)


if __name__ == "__main__":
    main()
