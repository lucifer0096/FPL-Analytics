# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API — started because [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (the community dataset this project used historically) is archived at the end of the 2024-25 season and won't be updated for 2025-26 onward.

## Status

**Stage 1 (in progress): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history.

Planned next: an expected-points (xP) model trained on 2022-23 through 2024-25 data (the seasons with full xG/xA fields), then a squad optimizer, then a live dashboard.

## Project Structure

```text
FPL-Analytics/
├── src/
│   └── collector/
│       ├── fpl_api.py     # Thin client for the FPL API endpoints
│       └── snapshot.py    # Snapshots current season data to data/raw/
├── data/
│   ├── raw/                # Gitignored — raw API snapshots, regenerate anytime
│   └── processed/          # Gitignored — cleaned/feature-engineered data
├── notebooks/               # EDA and model development
├── models/                  # Gitignored — trained model artifacts
└── requirements.txt
```

## Data Sources

- **Official FPL API** (free, public, no auth): `bootstrap-static` for all players/teams, `element-summary/{id}` for per-gameweek player history, `fixtures` for the season schedule, `entry/{id}` for a manager's team/history/picks.
- **Historical seasons (2022-23 to 2024-25, full xG/xA fields)**: sourced from the vaastav dataset (`E:\Fantasy-Premier-League`, cloned separately, not part of this repo) for model training, since the FPL API itself only exposes the current season's gameweek-by-gameweek data.

## Running the Collector

```bash
pip install -r requirements.txt

# Optional: snapshot a specific manager's team history/picks too
export FPL_ENTRY_ID=1132016   # or set FPL_ENTRY_ID on Windows

python src/collector/snapshot.py
```

This writes to `data/raw/{season}/`:
- `bootstrap/bootstrap_{timestamp}.json` — full player/team snapshot
- `fixtures.csv` — season fixture list with difficulty ratings
- `gw_history.csv` — one row per player per finished gameweek (empty until gameweeks have been played)
- `entry/{entry_id}/history.json` — a manager's season-by-season totals + current-season gameweek record
- `entry/{entry_id}/picks/gw{n}.json` — a manager's squad for each finished gameweek

### Dynamic scheduling

`snapshot.py` doesn't assume gameweeks land on a fixed day — fixtures get rearranged, some gameweeks span midweek, and blank/double gameweeks skip or double up entirely. Instead, each run checks the FPL API's own `finished` and `data_checked` flags per gameweek and only does the expensive part (fetching every player's history) when a new gameweek is actually ready. State (the last gameweek snapshotted) is tracked in `data/raw/collector_state.json`.

```bash
python src/collector/snapshot.py --check-only  # exit 0 if a snapshot is needed, 1 if not; no run
python src/collector/snapshot.py               # normal run: checks, snapshots only if needed
python src/collector/snapshot.py --force       # always snapshot, ignoring saved state
```

### Automated collection

`.github/workflows/weekly-collector.yml` runs daily (06:00 UTC) via GitHub Actions: it runs `--check-only` first, and only does a full snapshot when a new gameweek is ready, uploading the result as a 90-day build artifact. Collector state is cached between runs so the check works across separate CI runs, not just locally. To also capture your own team's history/picks, add an `FPL_ENTRY_ID` repository secret.

## Manager History

[`docs/my-fpl-history.html`](docs/my-fpl-history.html) is a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Open it directly in a browser, or view it live via GitHub Pages once enabled for this repo. The season figures are hardcoded from a point-in-time snapshot rather than fetched live — it'll be superseded by the planned dashboard, which will read directly from the collector's saved history instead.

## Future Improvements

- Build the xP model (gradient boosting or similar) on 2022-23–2024-25 data, validated against the FPL API's own naive `xP` field as a baseline.
- Build a squad optimizer (integer/linear programming) that picks the best 15-man squad under budget and formation constraints using model predictions.
- Build a live dashboard (Streamlit) showing current-gameweek predictions, transfer suggestions, and how a manager's actual picks compare to what the model would have chosen.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
