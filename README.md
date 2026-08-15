# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API — started because [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (the community dataset this project used historically) is archived at the end of the 2024-25 season and won't be updated for 2025-26 onward.

**[View the manager history page →](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)**

## Status

**Stage 1 (done): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history. Runs dynamically via GitHub Actions — see [Automated collection](#automated-collection) below.

**Stage 2 (in progress): expected-points (xP) model.** Historical data loading and feature engineering are done — 205,835 player-gameweek rows across all 9 tracked seasons (2016-17 to 2024-25), with rolling form and availability features built on top. See [Historical Training Data](#historical-training-data) and [Feature Engineering](#feature-engineering) below. Model training is next.

Planned after that: a squad optimizer, then a live dashboard.

## Project Structure

```text
FPL-Analytics/
├── src/
│   ├── collector/
│   │   ├── fpl_api.py         # Thin client for the FPL API endpoints
│   │   └── snapshot.py        # Snapshots current season data to data/raw/
│   └── model/
│       ├── load_historical.py # Loads/unifies 9 seasons of vaastav data for training
│       └── features.py        # Rolling form, availability, and gw-count features
├── data/
│   ├── raw/                # Gitignored — raw API snapshots, regenerate anytime
│   └── processed/          # Gitignored — historical_gw.parquet, features.parquet
├── docs/
│   └── my-fpl-history.html # Manager history page, served via GitHub Pages
├── notebooks/               # EDA and model development
├── models/                  # Gitignored — trained model artifacts
└── requirements.txt
```

## Data Sources

- **Official FPL API** (free, public, no auth): `bootstrap-static` for all players/teams, `element-summary/{id}` for per-gameweek player history, `fixtures` for the season schedule, `entry/{id}` for a manager's team/history/picks.
- **Historical seasons (2016-17 to 2024-25)**: sourced from the vaastav dataset (`E:\Fantasy-Premier-League`, cloned separately, not part of this repo) for model training, since the FPL API itself only exposes the current season's gameweek-by-gameweek data — past seasons only exist in point-in-time archives like this one. Only the official FPL API can be scraped live going forward; there's no way to retroactively scrape data for seasons that have already ended.

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

## Historical Training Data

```bash
python src/model/load_historical.py
```

Loads and unifies all 9 seasons (2016-17 to 2024-25) of vaastav's per-season `merged_gw.csv` files into one table, saved to `data/processed/historical_gw.parquet` (205,835 player-gameweek rows). Uses the 33 columns present in every season (minutes, goals, assists, bonus, BPS, ICT index, value, etc.) — xG/xA fields are excluded, since they only exist for 2022-23 onward and including them would mean dropping two-thirds of the training data.

Two real data-quality issues surfaced and fixed while building this loader, both silent-corruption risks if missed:
- **`team` was two different types across seasons** — a numeric, season-scoped id pre-2020-21, a name string from 2020-21 onward. Concatenating as-is broke parquet serialization; older seasons' ids are now resolved to the same name strings via `master_team_list.csv`.
- **`element` (the in-file player id) is reassigned every season** — id `1` is a different real player in each of the 9 seasons. Verified directly: Salah's `element` changed every year (234, 253, 191, 254, 233, 283, 308, 328) while `players_raw.csv`'s `code` field stayed fixed at 118748 throughout. `player_code` is now joined in as the stable cross-season identifier — any rolling/lagged feature must group by this, not `element`.

## Feature Engineering

```bash
python src/model/features.py
```

Builds the actual predictive features on top of the unified historical table, saved to `data/processed/features.parquet`:
- Rolling 3- and 5-gameweek averages for points, minutes, BPS, and ICT index
- Last-gameweek minutes and a "started" flag, to capture short-term availability separate from a season-long average
- Career and season-to-date gameweek counts (season count resets at each season boundary; career count doesn't)

Everything is grouped by `player_code` (not `element` — see the caveat above) and shifted by one gameweek before any rolling calculation, so a gameweek's own outcome can never leak into its own feature row. Verified two ways: a built-in check confirms zero rows at a player's first-ever tracked gameweek still carry a non-null rolling average (which would indicate leakage), and Salah's first five gameweeks of 2017-18 were hand-checked against the actual output (e.g. `total_points_avg_last_3` at GW5 = 4.33, matching (1+11+1)/3 from GW2–4, correctly excluding GW5's own score). Rolling form also correctly carries across season boundaries — a player's form entering a new season's GW1 reflects their last games of the previous season rather than resetting to null.

## Manager History

**[Live page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)** — a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Source: [`docs/my-fpl-history.html`](docs/my-fpl-history.html). The season figures are hardcoded from a point-in-time snapshot rather than fetched live — it'll be superseded by the planned dashboard, which will read directly from the collector's saved history instead.

## Future Improvements

- Add fixture difficulty and home/away as features (not yet included — currently only player-form and availability features exist).
- Train the xP model (gradient boosting or similar) on the unified 9-season feature set, validated against the FPL API's own naive `xP` field as a baseline.
- Consider a secondary model or extra features using xG/xA for 2022-23 onward once the core model is validated, since that signal is only available for a third of the training window.
- Build a squad optimizer (integer/linear programming) that picks the best 15-man squad under budget and formation constraints using model predictions.
- Build a live dashboard (Streamlit) showing current-gameweek predictions, transfer suggestions, and how a manager's actual picks compare to what the model would have chosen.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
