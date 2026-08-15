# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API. Historical training data comes from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), which is actively maintained (contrary to this project's original premise — its README previously carried a stale "archived after 2024-25" notice; the maintainer has since posted 3 major updates a season instead of weekly ones, most recently adding full 2025-26 data and initial 2026-27 data). Going forward, this project's own [collector](#running-the-collector) captures each gameweek as it happens, since even an actively maintained third-party archive is still a dependency this project shouldn't rely on indefinitely.

**[View the manager history page →](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)**

## Status

**Stage 1 (done): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history. Runs dynamically via GitHub Actions — see [Automated collection](#automated-collection) below.

**Stage 2 (in progress): expected-points (xP) model.** Historical data loading, feature engineering, and a first trained model are all done — 253,578 player-gameweek rows across 10 tracked seasons (2016-17 to 2025-26), with rolling form, availability, team form, and fixture difficulty features on top, and a LightGBM model trained and validated. See [Historical Training Data](#historical-training-data), [Feature Engineering](#feature-engineering), and [Model Training](#model-training) below.

Planned after that: tuning/iterating on the model, then a squad optimizer, then a live dashboard.

## Project Structure

```text
FPL-Analytics/
├── src/
│   ├── collector/
│   │   ├── fpl_api.py         # Thin client for the FPL API endpoints
│   │   └── snapshot.py        # Snapshots current season data to data/raw/
│   └── model/
│       ├── load_historical.py # Loads/unifies 10 seasons of vaastav data for training
│       ├── features.py        # Rolling form, availability, team form, fixture difficulty
│       └── train.py           # Chronological train/validation split, LightGBM model
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
- **Historical seasons (2016-17 to 2025-26)**: sourced from the vaastav dataset (`E:\Fantasy-Premier-League`, cloned directly from `vaastav/Fantasy-Premier-League` — not a fork, and not part of this repo) for model training, since the FPL API itself only exposes the current season's gameweek-by-gameweek data. 2026-27 isn't included yet — that season hasn't started (first fixture 21 Aug 2026), so there's no gameweek data for it anywhere; this project's own collector will capture it as it happens.

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

Loads and unifies 10 seasons (2016-17 to 2025-26) of vaastav's per-season `merged_gw.csv` files into one table, saved to `data/processed/historical_gw.parquet` (253,578 player-gameweek rows). Uses the 33 columns present in every season (minutes, goals, assists, bonus, BPS, ICT index, value, etc.) — xG/xA fields are excluded, since they only exist for 2022-23 onward and including them would mean dropping most of the training data.

Four real data-quality issues surfaced and fixed while building this loader, all silent-corruption risks if missed:
- **`team` was two different types across seasons** — a numeric, season-scoped id pre-2020-21, a name string from 2020-21 onward. Concatenating as-is broke parquet serialization; older seasons' ids are now resolved to the same name strings via `master_team_list.csv`.
- **`opponent_team` is a numeric, season-scoped id in EVERY season** — the same bug class as `team`, but not caught by the first fix since it's a separate column present even where `team` was already a string. Verified directly: id `4` is Chelsea in 2016-17, Burnley in 2020-21, Brentford in 2022-23. Resolved via `teams.csv` (2019-20 onward) with a `master_team_list.csv` fallback for 2016-17 to 2018-19.
- **`element` (the in-file player id) is reassigned every season** — id `1` is a different real player in each season. Verified directly: Salah's `element` changed every year (234, 253, 191, 254, 233, 283, 308, 328) while `players_raw.csv`'s `code` field stayed fixed at 118748 throughout. `player_code` is now joined in as the stable cross-season identifier — any rolling/lagged feature must group by this, not `element`.
- **2024-25 introduced a "pick a Manager" feature** (`position == "AM"`, 322 rows) — a real-life manager selectable alongside your 15 players, scored on entirely different rules. Not a player, so excluded from the loader outright rather than left for downstream feature engineering or training to special-case.

## Feature Engineering

```bash
python src/model/features.py
```

Builds the actual predictive features on top of the unified historical table, saved to `data/processed/features.parquet`:
- Rolling 3- and 5-gameweek averages for points, minutes, BPS, and ICT index
- Last-gameweek minutes and a "started" flag, to capture short-term availability separate from a season-long average
- Career and season-to-date gameweek counts (season count resets at each season boundary; career count doesn't)
- Rolling 5-match team-level goals-for/against, for both the player's own team and their opponent
- `fixture_difficulty`: prefers FPL's own published 1-5 rating (`fixtures.csv`, 2018-19 through 2025-26) over a hand-built proxy, since it's a materially better signal (also weighs defense, home advantage, and other factors this project doesn't have data for) — the proxy is used only for 2016-17/2017-18, where FPL's rating doesn't exist in this dataset

Player-level features are grouped by `player_code` (not `element` — see the caveat above) and shifted by one gameweek before any rolling calculation, so a gameweek's own outcome can never leak into its own feature row. `fixture_difficulty` is the one exception that's joined in directly without a shift — FPL publishes it before kickoff, so using it isn't a leak. Verified multiple ways, not just by confirming the code runs without erroring:
- A built-in check confirms zero rows at a player's first-ever tracked gameweek still carry a non-null rolling average (would indicate leakage).
- Salah's first five gameweeks of 2017-18 were hand-checked against the actual output (e.g. `total_points_avg_last_3` at GW5 = 4.33, matching (1+11+1)/3 from GW2–4). Rolling form also correctly carries across season boundaries rather than resetting to null.
- Building team-form features initially exploded the row count — traced to `players_raw.csv`'s end-of-season team snapshot misattributing a transferred player's early-season games to their later club, producing two contradictory score rows for the same (season, GW, team). Fixed by dropping any team-match row with more than one distinct score before building the rolling average, rather than silently keeping the corruption.
- The 2016-17/2017-18 `fixture_difficulty` fallback was on a different scale entirely from FPL's real rating (mean ~1.4 vs ~2.9) — caught by comparing per-season distributions after the row-count fix, not assumed correct. Rescaled via quantile binning so the column means roughly the same thing regardless of era.

## Model Training

```bash
python src/model/train.py
```

Trains a LightGBM regressor on a fixed allowlist of pre-match-known features (not an exclusion list — a new leaky column added later can't silently become a model input). Uses a **chronological** split, not a random one: everything before 2024-25 trains the model, 2024-25 is the validation season, and 2025-26 is held out entirely, untouched by any training or tuning decision, as a true final check once the model is otherwise finalized. A random split would let the model "see the future" within a season.

**Two baselines, reported with different confidence:**
- A genuinely leak-free naive baseline — the player's own rolling 5-gameweek average (already a model feature, shifted by 1 gameweek).
- FPL's own published `xP`. This one carries an explicit caveat from the data source's own maintainer: `xP` is scraped from FPL's `ep_this` field, and since the scraper runs after each gameweek ends, the archived value may contain information FPL updated post-match — the update cadence for that field isn't documented. This makes it an informative but not fully trustworthy comparison, not a guaranteed-clean pre-match target. Reported as such rather than treated as ground truth.

**Current validation results (2024-25 season):**

| | MAE | RMSE |
|---|---|---|
| Trained model | 1.003 | 1.921 |
| Naive baseline (rolling-5 average) | 1.052 | 2.069 |
| FPL's own xP (caveat above) | 0.904 | 1.757 |

The model beats the clean naive baseline, which is the trustworthy comparison. It doesn't yet beat FPL's own xP, though that comparison isn't fully apples-to-apples given the caveat above — a real gap remains either way, and closing it is the next round of work (see Future Improvements).

Highest-importance features in the current model: `ict_index_avg_last_5`, `bps_avg_last_5`, `ict_index_avg_last_3`, `career_gw_count`, `bps_avg_last_3` — the rolling advanced-stat averages dominate over the newer team-form and fixture-difficulty features.

## Manager History

**[Live page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)** — a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Source: [`docs/my-fpl-history.html`](docs/my-fpl-history.html). The season figures are hardcoded from a point-in-time snapshot rather than fetched live — it'll be superseded by the planned dashboard, which will read directly from the collector's saved history instead.

## Future Improvements

- Close the gap to FPL's own xP baseline — try adding `value` (price) as a feature, tuning LightGBM's hyperparameters, and/or a two-stage model (predict minutes/start probability first, then points conditional on playing), since ~64% of validation rows are players who didn't play that gameweek at all.
- Consider a secondary model or extra features using xG/xA for 2022-23 onward once the core model is validated, since that signal is only available for a third of the training window.
- Run the model against the 2025-26 final holdout only once no further tuning decisions remain, to get an honest read on generalization.
- Build a squad optimizer (integer/linear programming) that picks the best 15-man squad under budget and formation constraints using model predictions.
- Build a live dashboard (Streamlit) showing current-gameweek predictions, transfer suggestions, and how a manager's actual picks compare to what the model would have chosen.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
