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

Run this periodically (e.g. weekly, after each gameweek's matches finish) to build up the season's history locally.

## Future Improvements

- Build the xP model (gradient boosting or similar) on 2022-23–2024-25 data, validated against the FPL API's own naive `xP` field as a baseline.
- Build a squad optimizer (integer/linear programming) that picks the best 15-man squad under budget and formation constraints using model predictions.
- Build a live dashboard (Streamlit) showing current-gameweek predictions, transfer suggestions, and how a manager's actual picks compare to what the model would have chosen.
- Schedule the collector to run automatically after each gameweek.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
