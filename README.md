# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API. **FPL's own live API is this project's actual basis, not vaastav's archive** — vaastav/Fantasy-Premier-League is used ONLY as a historical bootstrap for 2016-17 through 2025-26, because that's the one thing the live API genuinely cannot provide: verified directly against the API that once a season ends, `element-summary`'s per-gameweek `history` empties out and `history_past` only ever returns SEASON-TOTAL aggregates (total_points, minutes, etc. summed for the whole season) — there is no way, official or otherwise, to pull old seasons' gameweek-by-gameweek data from FPL itself, so a third-party archive is the only source for that window. Every season from 2026-27 onward is captured entirely by this project's own [collector](#running-the-collector) as it happens, straight from the live API, with no vaastav dependency at all — and it captures MORE than vaastav's schema ever could: fields like `in_dreamteam`, `defensive_contribution` (part of FPL's 2025-26 scoring overhaul), `starts`, and real `expected_goals`/`expected_assists` that vaastav's CSVs simply don't carry for any season (see `load_live.py`).

**[View the manager history page →](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)**

## Status

**Stage 1 (done): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history. Runs dynamically via GitHub Actions — see [Automated collection](#automated-collection) below.

**Stage 2 (done): expected-points (xP) model.** 253,578 player-gameweek rows across 10 tracked seasons (2016-17 to 2025-26), with rolling form, availability, team form, fixture difficulty, and new-player-baseline features, and a trained/validated LightGBM model. See [Historical Training Data](#historical-training-data), [Feature Engineering](#feature-engineering), and [Model Training](#model-training) below.

**Stage 3 (done): squad optimizer.** Squad builder, transfer optimizer, and chip-timing advisor, all encoding FPL's real rules (verified against the live API, not assumed) and tested against real historical data. See [Squad Optimizer](#squad-optimizer) below.

**Stage 4 (done): dashboard.** A two-page Streamlit app: a Home page for the manager's real, live 2026-27 squad/points/league standings, and a Historical & Model page for demo squad-building modes and methodology. See [Dashboard](#dashboard) below.

**Stage 5 (in progress): live season.** The 2026-27 season started 21 Aug 2026 — the collector now captures real gameweek data as it happens (see the `_latest_live_gw` fix in Dashboard below), and the Home page shows the manager's actual squad/points/transfers/league standings rather than a demo.

Planned next: retrain the xP model once enough live 2026-27 gameweeks exist to be worth incorporating (currently trained on 2016-17 through 2025-26 only); build a real multi-gameweek Chip Advisor projection for the live squad once there's a genuine upcoming-fixtures window to project against.

## Project Structure

```text
FPL-Analytics/
├── src/
│   ├── collector/
│   │   ├── fpl_api.py         # Thin client for the FPL API endpoints
│   │   └── snapshot.py        # Snapshots current season data to data/raw/
│   └── model/
│       ├── load_historical.py # Loads/unifies 10 seasons of vaastav data (2016-17-2025-26) + appends this project's own collected 2026-27+ data
│       ├── load_live.py       # Loads this project's OWN collector snapshots -- preserves FPL-only fields (in_dreamteam, defensive_contribution, xG/xA, starts) vaastav's data never has, for any season
│       ├── features.py        # Rolling form, availability, team form, fixture difficulty
│       ├── train.py           # Chronological train/validation split, LightGBM model
│       ├── optimizer.py       # Squad builder + transfer optimizer (PuLP)
│       ├── chips.py           # Chip-timing advisor (Bench Boost/Triple Captain/Free Hit/Wildcard)
│       ├── test_optimizer.py  # Squad/transfer optimizer checks against real data
│       └── test_chips.py      # Chip advisor checks against real historical data
├── data/
│   ├── raw/                # Gitignored — raw API snapshots, regenerate anytime
│   └── processed/          # Gitignored — historical_gw.parquet, features.parquet
├── app/
│   ├── app.py              # Streamlit dashboard, Home page -- live squad, transfers, chips, league tracker
│   ├── shared.py           # Data-loading/pool-building/pitch-rendering helpers shared by every page
│   └── pages/
│       └── 1_Historical_and_Model.py  # Second page -- demo Squad Builder modes, model metrics, past seasons
├── docs/
│   └── my-fpl-history.html # Manager history page, served via GitHub Pages
├── notebooks/               # EDA and model development
├── models/                  # Gitignored — trained model artifacts
└── requirements.txt
```

## Data Sources

- **Official FPL API** (free, public, no auth): `bootstrap-static` for all players/teams, `element-summary/{id}` for per-gameweek player history, `fixtures` for the season schedule, `entry/{id}` for a manager's team/history/picks.
- **Historical seasons (2016-17 to 2025-26)**: sourced from the vaastav dataset (`E:\Fantasy-Premier-League`, cloned directly from `vaastav/Fantasy-Premier-League` — not a fork, and not part of this repo) for model training, since the FPL API itself only exposes the current season's gameweek-by-gameweek data. 2026-27 is captured by this project's own collector as it happens (see `src/model/load_live.py`), starting from the season's first gameweek.

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
- `gw_history.csv` — one row per player per finished gameweek (empty until gameweeks have been played), with EVERY field the live API returns per gameweek (not a fixed subset) — including `in_dreamteam`, `defensive_contribution`, `starts`, and real `expected_goals`/`expected_assists`, none of which vaastav's historical data has for any season (see Historical Training Data above). `src/model/load_live.py` reads this file back for feature engineering/training.
- `entry/{entry_id}/info.json` — a manager's real name and team name, used by the dashboard's Manager History tab so it shows a real name instead of a bare numeric entry id
- `entry/{entry_id}/history.json` — a manager's season-by-season totals + current-season gameweek record
- `entry/{entry_id}/picks/gw{n}.json` — a manager's real squad/picks for each gameweek that's at least started (not gated on `data_checked` — a manager's own picks/points are correct, if provisional, the moment that gameweek's deadline passes; see `_latest_live_gw` in `snapshot.py`)
- `entry/{entry_id}/leagues/{league_id}.json` — full real standings for every PRIVATE classic league this manager has joined by code (excludes FPL's own auto-generated global/region/club leagues), used by the dashboard's League Tracker tab
- `live/gw{n}.json` — every player's real points for one gameweek, one API call (`event/{gw}/live`) instead of per-player `element-summary` calls — used to compute a manager's real per-player point breakdown for their own squad

### Dynamic scheduling

`snapshot.py` doesn't assume gameweeks land on a fixed day — fixtures get rearranged, some gameweeks span midweek, and blank/double gameweeks skip or double up entirely. Instead, each run checks the FPL API's own `finished` and `data_checked` flags per gameweek and only does the expensive part (fetching every player's history) when a new gameweek is actually ready. State (the last gameweek snapshotted) is tracked in `data/raw/collector_state.json`.

```bash
python src/collector/snapshot.py --check-only  # exit 0 if a snapshot is needed, 1 if not; no run
python src/collector/snapshot.py               # normal run: checks, snapshots only if needed
python src/collector/snapshot.py --force       # always snapshot, ignoring saved state
```

### Automated collection

`.github/workflows/weekly-collector.yml` runs daily (06:00 UTC) via GitHub Actions: it runs `--check-only` first, and only does a full snapshot when a new gameweek is ready, uploading the result as a 90-day build artifact. Collector state is cached between runs so the check works across separate CI runs, not just locally. To also capture your own team's history/picks/league standings, add an `FPL_ENTRY_ID` repository secret.

**Keeping the deployed dashboard's data fresh.** Uploading a build artifact isn't enough on its own — the deployed Streamlit Cloud app reads the repo checkout directly, not this workflow's temporary artifacts, and `data/raw/` (where the real snapshot lives) is gitignored. So after the collector step, the workflow also runs `src/collector/refresh_dashboard_fallbacks.py`, which regenerates every `data/dashboard_*` fallback file (bootstrap, **fixtures.csv**, entry info/history, current-squad-picks, league standings) from whatever `data/raw/` that run just produced, and commits the result straight back to `main` (`permissions: contents: write` is set on the job for this). This means the deployed app's live squad/points/league-tracker/PL-table/fixture-difficulty/pre-season data is never more than a day stale while the workflow keeps running — same staleness bound already accepted for `data/dashboard_bootstrap.json`'s prices, just now covering the newer live-season fallbacks too. Not a new privacy decision: everything committed here is already published on this project's own public manager-history GitHub Pages page, linked throughout the dashboard.

**Fixture data specifically** (`data/dashboard_fixtures.csv`): `data/raw/2026-27/fixtures.csv` is what `premier_league_table()`, `team_upcoming_fixtures()`, and every feature built on top of them (PL Table tab, fixture-difficulty strips on squad cards, Transfers' fixture-based adjustment) actually read — but that path lives under the gitignored `data/raw/`, so a fresh Streamlit Cloud deploy had none of it at all until this fallback was added (`shared.py`'s `_fixtures_path()` checks the live path first, then falls back to this committed copy, same pattern as `_latest_bootstrap_path()`). Before this fix, those features weren't "waiting for end of week" — they were silently reading a file that would never exist on the deployed app, regardless of how much real gameweek data existed. Verified directly by hiding the local `data/raw/2026-27/fixtures.csv` and confirming both functions correctly fall back and still return real GW1 results.

## Historical Training Data

```bash
python src/model/load_historical.py
```

Loads and unifies 10 seasons (2016-17 to 2025-26) of vaastav's per-season `merged_gw.csv` files into one table (`load_all_seasons()`), then appends any 2026-27+ gameweeks this project's own collector has captured directly from the live FPL API (`load_live.py`, via `load_all_seasons_with_live()` — a no-op until the collector has captured at least one finished gameweek). Saved to `data/processed/historical_gw.parquet` (253,578 vaastav rows currently; grows as live gameweeks are collected). Uses the 33 columns present in every vaastav season (minutes, goals, assists, bonus, BPS, ICT index, value, etc.) — xG/xA fields from vaastav are excluded, since they only exist for 2022-23 onward there and including them would mean dropping most of the training data.

**vaastav is a historical bootstrap, not this project's basis.** It exists solely to cover 2016-17–2025-26, the one window FPL's own API cannot provide — verified directly: once a season ends, `element-summary`'s per-gameweek `history` is empty and `history_past` only ever returns season-TOTAL aggregates, never gameweek-by-gameweek rows, for any past season. There's no official or unofficial way to pull old per-gameweek data from FPL itself. Every 2026-27+ gameweek, by contrast, is captured live and directly from FPL — see `load_live.py` below — and carries several real fields vaastav's schema can never have, for any season:

| Field | What it is | Why vaastav can't have it |
|---|---|---|
| `in_dreamteam` | FPL's own official "Team of the Week" flag | Not part of `merged_gw.csv`'s schema in any season |
| `defensive_contribution` | Points for defensive actions (tackles, clearances, blocks, interceptions) | Introduced in FPL's 2025-26 scoring overhaul — postdates every vaastav season's schema |
| `starts` | Whether a player started the match (distinct from playing any minutes) | Not part of `merged_gw.csv`'s schema in any season |
| `expected_goals` / `expected_assists` / `expected_goal_involvements` / `expected_goals_conceded` | Real underlying xG/xA from FPL itself | vaastav only has these for 2022-23 onward, and via a different (Understat-sourced) pipeline — FPL's own numbers are a distinct, more directly relevant source |

Five real data-quality issues surfaced and fixed while building this loader, all silent-corruption risks if missed:
- **`team` was two different types across seasons** — a numeric, season-scoped id pre-2020-21, a name string from 2020-21 onward. Concatenating as-is broke parquet serialization; older seasons' ids are now resolved to the same name strings via `master_team_list.csv`.
- **`opponent_team` is a numeric, season-scoped id in EVERY season** — the same bug class as `team`, but not caught by the first fix since it's a separate column present even where `team` was already a string. Verified directly: id `4` is Chelsea in 2016-17, Burnley in 2020-21, Brentford in 2022-23. Resolved via `teams.csv` (2019-20 onward) with a `master_team_list.csv` fallback for 2016-17 to 2018-19.
- **`element` (the in-file player id) is reassigned every season** — id `1` is a different real player in each season. Verified directly: Salah's `element` changed every year (234, 253, 191, 254, 233, 283, 308, 328) while `players_raw.csv`'s `code` field stayed fixed at 118748 throughout. `player_code` is now joined in as the stable cross-season identifier — any rolling/lagged feature must group by this, not `element`.
- **2024-25 introduced a "pick a Manager" feature** (`position == "AM"`, 322 rows) — a real-life manager selectable alongside your 15 players, scored on entirely different rules. Not a player, so excluded from the loader outright rather than left for downstream feature engineering or training to special-case.
- **`merged_gw.csv`'s text encoding isn't consistent across the dataset** — verified at the raw-byte level: 2016-17/2017-18/2018-19 are genuinely Latin-1, 2019-20 onward are genuine UTF-8. Reading every season as Latin-1 (an earlier version of this loader) decodes UTF-8's 2-byte accented-character sequences as two separate wrong characters — surfaced visibly on the deployed dashboard as e.g. "JÃ©rÃ©my Doku" instead of "Jérémy Doku". Now uses the correct per-season encoding. `players_raw.csv` is UTF-8 in all 10 seasons with no exceptions, so it always reads as UTF-8 regardless of season — a separate rule from `merged_gw.csv`'s.

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
- `new_player_baseline`: a position/price-band fallback for players with no rolling form yet — newly promoted teams' players and new signings (~70-130 players every season) have every rolling-average feature null at their first tracked gameweek, leaving the model with no signal for exactly the players a manager most needs guidance on early in a season. No Championship/lower-league data source is used (that would need a separate third-party API with unverified free-tier depth); instead this is what similarly priced players in the same position scored on average league-wide, using only gameweeks already played.

Player-level features are grouped by `player_code` (not `element` — see the caveat above) and shifted by one gameweek before any rolling calculation, so a gameweek's own outcome can never leak into its own feature row. `fixture_difficulty` is the one exception that's joined in directly without a shift — FPL publishes it before kickoff, so using it isn't a leak. Verified multiple ways, not just by confirming the code runs without erroring:
- A built-in check confirms zero rows at a player's first-ever tracked gameweek still carry a non-null rolling average (would indicate leakage).
- Salah's first five gameweeks of 2017-18 were hand-checked against the actual output (e.g. `total_points_avg_last_3` at GW5 = 4.33, matching (1+11+1)/3 from GW2–4). Rolling form also correctly carries across season boundaries rather than resetting to null.
- Building team-form features initially exploded the row count — traced to `players_raw.csv`'s end-of-season team snapshot misattributing a transferred player's early-season games to their later club, producing two contradictory score rows for the same (season, GW, team). Fixed by dropping any team-match row with more than one distinct score before building the rolling average, rather than silently keeping the corruption.
- The 2016-17/2017-18 `fixture_difficulty` fallback was on a different scale entirely from FPL's real rating (mean ~1.4 vs ~2.9) — caught by comparing per-season distributions after the row-count fix, not assumed correct. Rescaled via quantile binning so the column means roughly the same thing regardless of era.
- `new_player_baseline` is a league-wide (not per-player) statistic, so its leakage-safety shift happens at the `(season, position, price_band, GW)` group level rather than per player. Verified: 100% of zero-history rows (1,940 across all seasons except 2016-17, which has no prior data to draw on at all) get a non-null baseline with sensible values (cheap defenders/mids around 0.5–0.7 pts), while 2016-17's true first gameweek correctly has zero coverage — confirming no leakage at the actual start of history, not just trusted by construction.

## Model Training

```bash
python src/model/train.py
```

Trains LightGBM models on a fixed allowlist of pre-match-known features (not an exclusion list — a new leaky column added later can't silently become a model input). Uses a **chronological** split, not a random one: everything before 2024-25 trains the models, 2024-25 is the validation season, and 2025-26 is held out entirely, untouched by any training or tuning decision, as a true final check once the model is otherwise finalized. A random split would let the model "see the future" within a season.

**Two model architectures, compared directly:**
- **Single-stage** — one LightGBM regressor predicts `total_points` for every row.
- **Two-stage** — since ~64% of rows are players who didn't play that gameweek at all, this splits the problem into a play classifier (P(plays), AUC 0.934) and a points-conditional-on-playing regressor, combined as `P(plays) × E[points | plays]`, on the hypothesis that a single regressor was spending most of its error budget on the play/didn't-play distinction. **Result: this didn't help much** — two-stage MAE (0.984) barely beat single-stage (0.986). The single model was already implicitly learning that distinction well via its existing minutes-based features. Kept in the pipeline and reported on every run as a documented negative result, not discarded.

**Two baselines, reported with different confidence:**
- A genuinely leak-free naive baseline — the player's own rolling 5-gameweek average (already a model feature, shifted by 1 gameweek).
- FPL's own published `xP`. This one carries an explicit caveat from the data source's own maintainer: `xP` is scraped from FPL's `ep_this` field, and since the scraper runs after each gameweek ends, the archived value may contain information FPL updated post-match — the update cadence for that field isn't documented. This makes it an informative but not fully trustworthy comparison, not a guaranteed-clean pre-match target. Reported as such rather than treated as ground truth.

**Current validation results (2024-25 season, full dataset):**

| | MAE | RMSE |
|---|---|---|
| Single-stage model | 0.986 | 1.914 |
| Two-stage model | 0.984 | 1.914 |
| Naive baseline (rolling-5 average) | 1.052 | 2.069 |
| FPL's own xP (caveat above) | 0.904 | 1.757 |

Both models beat the clean naive baseline. Neither yet beats FPL's own xP overall — but restricting to rows where the player actually played tells a different story:

| (played rows only, n=11,566) | MAE |
|---|---|
| Single-stage model | 1.832 |
| Naive baseline | 2.053 |
| FPL's own xP | 1.759 |

The gap to FPL's xP shrinks from ~0.08 (full dataset) to ~0.07 (played only) — most of the overall gap is concentrated in non-playing rows, where FPL's xP likely draws on real injury/team-news signals (press conferences, training reports) that a model built purely on historical box-score stats has no way to see. This reframes the next step: closing the remaining gap isn't primarily a modeling problem, it's a data problem — see Future Improvements.

Adding `new_player_baseline` (see Feature Engineering) improved both models measurably: single-stage MAE 1.003 → 0.986, two-stage 1.001 → 0.984 — real, verified gains from giving the model signal for the ~70-130 players every season who otherwise had no rolling history to draw on.

Highest-importance features in the single-stage model: `new_player_baseline` is now the single most important feature in the model (ahead of `ict_index_avg_last_5`, `ict_index_avg_last_3`, `bps_avg_last_5`, `bps_avg_last_3`) — a stronger confirmation of its value than the MAE improvement alone.

## Squad Optimizer

FPL's actual rules, verified against the live API's `bootstrap-static` `game_settings` and `chips` fields (2026-27 season) rather than assumed:

| Rule | Value |
|---|---|
| Squad size | 15 (2 GK / 5 DEF / 5 MID / 3 FWD) |
| Budget | £100.0m |
| Max players per club | 3 |
| Free transfers per week | 1, banked if unused |
| Max banked free transfers | 4 extra → **5 max in a single week** |
| Transfer cost beyond free ones | −4 pts each |
| Sell price on a player who's risen in value | Only 50% of the price rise is refunded, not the full current price |
| Wildcard / Free Hit / Bench Boost / Triple Captain | 1 each **per season half** (this season: GW1/2–19, GW20–38 — the boundary is a gameweek FPL sets each year, not literally "December") |

### Squad builder

```bash
python src/model/optimizer.py       # library — see test_optimizer.py for usage
python src/model/test_optimizer.py  # ad-hoc sanity check against real player pools
```

`optimize_squad()` (integer programming via PuLP) solves for the 15-man squad that maximizes total predicted points under the rules above. A nested optimization then picks the best valid starting XI (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD) from within that squad.

Deliberately decoupled from the trained model — it takes a plain DataFrame of `(player_id, position, team, cost, predicted_points)`, so it's usable and testable independent of where those predictions come from (the trained xP model, FPL's own xP, or a manual watchlist).

Verified two ways:
- Against the full 587-player live bootstrap pool: correct squad size, exact position quotas, under budget, ≤3 per team, valid starting XI formation. Pre-season `form` is 0 for every player right now (no gameweeks played yet), so this only proves the solver logic is correct — not that it picks good players.
- Against real historical data (2025-26 GW20, using each player's rolling-5 average as a stand-in for predicted points) — 2025-26 rather than the 2024-25 validation season deliberately, since it's the untouched final-holdout season and its player pool is far closer to who's actually in the Premier League now: correctly selects strong current picks — Hugo Ekitiké (8.4 pts, Liverpool), Rayan Cherki (8.0 pts, Man City), Matheus Nunes (8.0 pts, Man City) — within budget (£91.7m/£100m), with the highest scorers in the starting XI and lower scorers correctly benched.

### Transfer optimizer

`optimize_transfers()` solves for the transfer(s) — if any — worth actually making, not just whatever's technically nonnegative. It solves separately for every transfer count (0, 1, 2, ... up to free transfers + a paid cap) and walks up ONE transfer at a time, stopping the first time an individual additional transfer doesn't clear its own bar:
- **Free transfers** need a real minimum gain on their own (`MIN_GAIN_PER_FREE_TRANSFER`, not just >0) — a free transfer still has a cost (a banked resource spent chasing what may just be noise in a model with ~1 point of validation MAE per player-gameweek), so "positive" and "worth using" aren't the same bar.
- **Paid transfers** (a -4pt hit each) need their net gain to clear a real safety margin over breaking even (`MIN_NET_GAIN_PER_HIT`), since a marginal net gain is well within the model's own prediction error, not a genuine edge.

Critically, this is judged per-transfer, not as a batch average — a strong 1st transfer can't subsidize a weak 5th one riding along on the average. Having more free transfers banked never forces more of them to be used: holding is always the answer once nothing clears the bar, and the walk starts from 0 every time. Caught directly during testing: with 5 banked free transfers, the un-gated solver proposed a 5th transfer worth only +0.4 predicted points on its own, purely because a free transfer "cost nothing" in the raw objective — now it correctly stops at however many transfers actually pull their weight (2, in that same test case), holding the rest.

Applies the 50%-sell-fee rule via a `sell_price_col` parameter, since FPL doesn't refund a risen player's full current price.

`load_latest_prices()` always pulls the most recent collector-written bootstrap snapshot, never a cached DataFrame — prices move week to week based on transfer momentum, so a transfer's budget math needs current prices, not last week's.

Verified against real data (2025-26 GW20 squad → GW21 pool, 1 free transfer available): correctly proposes swapping James Tarkowski (DEF, £5.7m, rolling avg 4.6 pts) for Nathan Collins (DEF, £4.9m, rolling avg 7.0 pts) — a cheaper upgrade with better recent form — using the free transfer at zero cost, for a net +2.4 predicted points. Also verified the necessity gating: forcing a squad's best starter's predicted points to 0 (simulating an injury) with 0 free transfers correctly triggers a hit (net +4.0 after the -4pt cost, clearly worth it); the same squad with no injury and 0 free transfers correctly recommends nothing, since no available transfer clears the hit safety margin.

**Unlimited-transfer situations** (`unlimited_transfers=True`) bypass all of the above gating entirely, rather than incorrectly applying it: verified directly against the live API's `game_settings` that GW1 has a `transfers_cap` of 20 (effectively unlimited) instead of the normal 1-5 banked limit, and Wildcard/Free Hit make every transfer free by chip definition with no cap and no hit, for that one gameweek. None of these are "a lot of free transfers" — they're a genuinely different situation where the marginal-gain bars (built to protect a SCARCE weekly resource) don't apply, since the resource isn't scarce that gameweek. This path (`_optimize_transfers_unlimited`) reduces to `optimize_squad()` under a budget of `bank + full sell value of the current squad`, so it freely rebuilds however much of the squad is worth changing, with no hit cost ever applied — verified it reaches the full available gain (+6.4 predicted points in the same GW20→GW21 test case above) that the gated path deliberately declined to fully pursue. Exposed in the dashboard's Transfers tab as a checkbox ("Playing Wildcard or Free Hit this gameweek (or this is GW1)") that swaps which path runs.

### Chip advisor

```bash
python src/model/chips.py       # library — see test_chips.py for usage
python src/model/test_chips.py  # ad-hoc sanity check against real historical data
```

Ranks candidate gameweeks for each chip, given per-gameweek player projections the caller supplies:
- **Bench Boost** — total predicted points on the bench (non-starters), since that's the value this chip specifically unlocks.
- **Triple Captain** — the single best starter's projected points that gameweek (the extra value over a normal 2x captaincy).
- **Free Hit** — the gap between the current squad's projection and a freshly optimized squad's projection for the same gameweek, supplied per-candidate-gameweek by the caller (this function doesn't silently run the solver N times itself).
- **Wildcard** reuses the Free Hit logic but is explicitly flagged as a weaker signal — it only captures one gameweek's gap, not the multi-week strategic value a permanent squad change unlocks. No fully automated Wildcard-timing suggestion exists yet (see Future Improvements).

Verified against real historical data (2025-26, squad built at GW10, projected across GW10–14): real, plausible players surface as Triple Captain candidates (Haaland 9.6 pts, Gabriel dos Santos Magalhães 11.0 pts), Bench Boost values differ meaningfully by gameweek, and Free Hit correctly identifies GW14 as the week the squad had drifted furthest from optimal (34.0-point gap).

## Dashboard

```bash
streamlit run app/app.py
```

A two-page Streamlit app (Streamlit's native multipage support: `app.py` is the entry point/Home page, `app/pages/1_Historical_and_Model.py` is the second page, selectable from the sidebar). Split this way once the 2026-27 season actually started and there was real live data to show — see "Why two pages" below. Shared data-loading, pool-building, and pitch-rendering logic lives in `app/shared.py`, imported by both pages, so a fix in one place (a new fallback path, a rendering bug) automatically covers both.

**Look and feel**: `inject_shared_css()` (in `shared.py`, called once per page) is CSS-only styling on top of stock Streamlit — no custom components, so it degrades gracefully if Streamlit's internal class names shift. Deliberately pushed away from a default "admin dashboard" look: a real Google Font pairing (Sora for headings, Manrope for body), pill-shaped tabs with a colored gradient active state, glassy gradient-bordered metric cards, a shimmering gradient hero banner, and gradient pill buttons — all using the same green/purple palette as the pitch view and FPL's own brand purple, not arbitrary new colors. Every player card (`_player_card_html`) has a hover lift and a staggered fade/slide-in entrance so a squad appears card-by-card rather than popping in all at once; tabs, metrics, and the hero get their own subtle entrance animations. All motion is wrapped in a `prefers-reduced-motion` media query that disables it outright for anyone who's asked their OS for reduced motion. Every color stays translucent (rgba) or explicitly white-on-gradient rather than hardcoding a background, so it renders correctly under both Streamlit's light and dark themes.

### Home page — live, current season

Everything about the manager's REAL, current team, not a demo:
- **My Squad** — this manager's REAL squad and points for the most recent gameweek the collector has picks for, read from `entry/{id}/picks/gw{n}.json` and rendered on the same pitch-view styling as every optimizer-built squad (real headshots, captain/vice-captain badges, correct starting-XI/bench split) — not a prediction, not the optimizer's choice, the manager's actual picks. Per-player points come from `event/{gw}/live` (one call for every player, see `load_live_gw_points`), matched exactly against FPL's own reported gameweek total as a sanity check. Available (if provisional, before bonus points are finalized) as soon as that gameweek's deadline passes — see the `_latest_live_gw` fix below. A player with genuinely zero minutes that gameweek shows **"No game time"** on their card instead of a bare "0 pts" (`load_live_gw_minutes()`/`did_not_play` in `shared.py`) — a real 0 for a bench player who wasn't selected reads as an unfairly harsh result when it's really just "nothing to report yet." Each card also shows a small color-coded strip of that player's next 3 real fixtures (FPL's own 1-5 difficulty rating, green=easy through red=hard — same color scheme FPL's own site uses) — the same data Transfers uses to adjust its recommendations, not a separate display. A "P" badge marks each squad member FPL confirms as their club's PRIMARY penalty taker (`penalties_order == 1`, verified directly against 20 real current takers league-wide) — real, confirmed set-piece duty, not a guess, same signal also surfaced on the Differential picks list below. A real **captain suggestion** (`suggest_captain()`) uses FPL's own `ep_next` field (their real, published "expected points next gameweek" number) across every STARTING XI player in this same squad — never a bench player, regardless of their number — with a real position-based tiebreak (FWD > MID > DEF > GK, favoring attacking upside) for the ties that genuinely occur (verified: 3 real starters tied at the same `ep_next` in a real GW1 squad). Only shown when it actually differs from the manager's real choice, so agreement produces no redundant noise. Also shows **2026-27 progress** underneath: gameweek-by-gameweek rank/points for the season actually in progress, read live from `entry/{id}/history.json`'s `current` array (the live API only ever exposes this level of detail for the CURRENT season, never retroactively for a finished one), including a real running **"Points left on bench"** total (FPL's own `points_on_bench` field, summed across every gameweek so far) — points the bench scored that never counted toward the total, since only the starting XI's points do. Points are also shown against FPL's own real **average score across ALL managers** for that gameweek (`average_entry_score`, from bootstrap-static's events) and a real **"top X%"** figure (`overall_rank_percentage`, already computed by FPL itself) — a bare points/rank number means little without something real to compare it against.
- **Transfers** — suggests the transfer(s), if any, worth actually making from the manager's REAL squad (stored in session_state under a distinct `"live_squad"` sentinel, separate from the Historical page's `(season, gw)`/`"completed_season"`/`None` sentinels), checked against the live current player pool. Same gating logic as before: only recommends a transfer that clears a real minimum gain on its own, never forces unused free transfers into play, and a Wildcard/Free Hit checkbox switches to the unlimited-transfers path.
  - **Real free-transfer tracking**: the free-transfers slider is now pre-filled from `calculate_free_transfers()`, which simulates the manager's actual banked transfers from real `event_transfers` history (FPL's real rule: 1 per week, banking up to 5 max — verified via `game_settings.max_extra_free_transfers == 4`), not a manual guess. GW1's squad build is correctly excluded from the simulation (it isn't a "transfer" in FPL's own accounting), so free-transfer tracking genuinely starts from GW2.
  - **Stricter hit-safety margin for this pool specifically**: `min_net_gain_per_hit` is doubled to 4.0 here (vs. the default 2.0 the historical-gameweek path uses), since this pool's predicted_points is last season's closing form, not the trained model's real predictions — a materially noisier signal. Caught directly against real GW1 data: at the default margin, that noise alone justified 4 hits (-12pts) for a real squad; at 4.0 it correctly settles to 1 sensible free transfer with no hit, while a genuinely obvious case (a starter's predicted points zeroed, simulating an injury) still clears the stricter bar and takes the hit.
  - **Explicit caveat shown in the UI**: since predicted_points here is pre-season/last-season form, not real 2026-27 in-season data, the tab now shows an explicit warning that suggestions are a rough signal, not a confident recommendation, until enough live gameweeks exist for the trained model to be used here instead.
  - **A real, documented API limitation**: if a manager makes a transfer for the *next* gameweek before that gameweek's deadline passes, the dashboard can't see it yet — checked directly against the live API: `entry/{id}/event/{gw}/picks` 404s until that gameweek's own deadline passes, and there's no public endpoint for a squad mid-transfer-window. The My Squad tab now explicitly warns about this rather than silently showing stale data with no explanation; it resolves itself automatically once the next gameweek's deadline passes and the collector captures it.
  - **Real injury/suspension/doubt flagging**: before optimizing, the tab checks every squad member against FPL's own live `status`/`news`/`chance_of_playing_next_round` fields (verified directly: e.g. a genuinely injured player shows `status == "i"` with a real "Back injury - Unknown return date" news string) and shows any flagged player explicitly, by name, with FPL's own stated reason — not inferred from a predicted-points gap. Flagged players have their predicted_points zeroed for that optimization run, so a genuine injury/suspension (not just a form dip) is what drives a transfer-out suggestion. Verified end-to-end against a real, currently-injured squad member: correctly identified as the sole transfer-out candidate, replaced with an available alternative for a clear net gain.
  - **Real upcoming fixture difficulty**: predicted_points is adjusted (±15% at the extremes, neutral at FPL's own difficulty rating of 3) by each player's team's average fixture difficulty over the next 3 gameweeks (`team_upcoming_fixtures()`/`average_fixture_difficulty()` in `shared.py`), using FPL's own published 1-5 rating from `fixtures.csv` — the SAME fixture data shown on every squad card's color-coded fixture strip in the My Squad tab, not a separate, disconnected calculation. "Upcoming" is computed from the live bootstrap's own current-gameweek number, not `fixtures.csv`'s own `finished` flag, since that flag stays `False` for an already-played match until bonus points are fully locked in (checked directly against real GW1 data) — using it directly would have misclassified already-played fixtures as still upcoming.
  - **Differential picks** (`differential_finder()`): a real "differential" list — players owned by ≤10% of managers (FPL's own `selected_by_percent`) with meaningful real upside (FPL's own `ep_next`, the same field the captain suggestion and Chip Advisor already use) — quantifying a genuine FPL strategy concept (a rank-gaining edge from a player few other managers have) from real, checkable numbers rather than a vague "under the radar" guess. Also flags confirmed set-piece duty (see below) for each differential, since real penalty-taker status makes a low-owned player's upside more trustworthy than `ep_next` alone. Shown in a collapsed expander right next to the existing transfer-gating explanation, not as a separate standalone tool.
  - **League-wide injury/suspension feed** (`league_wide_status_flags()`): the exact same real `status`/`news`/`chance_of_playing_next_round` fields already used above to flag a squad member as injured/suspended/doubtful, applied to every player league-wide instead of just this manager's 15 — so a potential transfer-in (from the Differential picks list or anywhere else) can be checked for their OWN real availability before being brought in, not just the current squad's. Sorted by real ownership so the flags most managers actually need to know about surface first. Shown right next to Differential picks, reusing the same `STATUS_LABELS` mapping rather than a second copy of the same logic.
- **Chip Advisor** — a real, honest single-gameweek check, not a full multi-week projection: checked directly against the live API that FPL only ever publishes `ep_next` (expected points for the SINGLE next gameweek), never anything further out, so a genuine "which of the next 5 gameweeks is best" ranking can't be built honestly yet. What IS shown: your bench's real summed `ep_next` (Bench Boost's real value if used next gameweek) and the same captain suggestion from My Squad (Triple Captain's real extra value being one more multiple of that same real number). Both explicitly labeled as a single data point for the one real upcoming gameweek, not a "best gameweek" recommendation — a genuine multi-gameweek version needs either FPL to publish further-out predictions or this project's own trained model to have enough live 2026-27 gameweeks to project forward with.
- **League Tracker** — real standings for every PRIVATE classic (points-based) mini-league this manager has joined by code, read from `entry/{id}/leagues/{league_id}.json` (FPL's own `leagues-classic/{id}/standings` endpoint via the collector), falling back to `data/dashboard_leagues.json` (refreshed daily by the collector workflow, see Automated Collection) when `data/raw/` is unavailable. Deliberately excludes FPL's own auto-generated global/region/favourite-club leagues (`league_type == "s"`) — those aren't leagues a manager actually "joined." This manager's own row is highlighted in the standings table. A league selector lets you switch between every private league you're in.
- **Price Changes** — real 2026-27 price movement so far this season, split into risers/fallers (`live_price_changes()` in `shared.py`). Unlike the Historical page's price-riser analysis (which has to reconstruct a season's start/end price from per-gameweek rows, since old seasons never had this tracked live), this uses FPL's OWN `cost_change_start` field directly — the real number FPL itself uses to track cumulative price change since the season's opening prices — so no GW-by-GW historical table is needed here at all. Reads from the same bootstrap snapshot every other live-price feature uses, which the scheduled collector workflow refreshes daily — so this tab keeps updating week by week automatically as the season progresses, with no separate wiring. Empty (with an explicit, correct "nothing yet" message, not an error) very early in a season before FPL's price-change algorithm has reacted to any real transfer activity.
  - **"Likely to move next"** (`likely_price_movers()`): players with real, current transfer momentum (`transfers_in_event`/`transfers_out_event`, both real FPL fields — this gameweek's activity, not cumulative) who haven't had their price move yet, explicitly excluding anyone already shown in the risers/fallers above so the two sections don't silently overlap. This is a real, public leading indicator FPL's own price-change algorithm reacts to, not a guess. Verified against real data in a striking, independently-confirming way: Pedro Porro (the exact player flagged injured and correctly identified as a transfer-out target earlier) showed up as the single biggest net-transfers-OUT player league-wide, and Riccardo Calafiori (who this manager actually transferred in) as the single biggest net-transfers-IN player — real momentum matching real manager behavior, not coincidence.
- **PL Table** — the real, current Premier League table for the season in progress (`premier_league_table()` in `shared.py`), computed directly from `fixtures.csv`'s own recorded `team_h_score`/`team_a_score` for every match with an actual result, not FPL's strength ratings or any derived metric. Deliberately keys off a real score being present rather than the `finished` column — verified directly that `finished` stays `False` on a played match for hours after full time, until bonus points are finalized, which would otherwise hide already-known results. Standard real PL sort order (points, then goal difference, then goals for). Updates automatically as the collector runs each day, same as every other live tab.

### Historical & Model page — demo modes and methodology

- **Overview** — pipeline summary and headline stats.
- **Season Insights** — real, verifiable facts about a completed (or completing) season, not a squad-building tool: top scorers, best value picks (points per £m spent, minimum 450 minutes played so a small sample doesn't dominate), top scorer by position, and biggest real price risers. Every number here is a real historical total (`season_insights()` in `shared.py`), not a prediction.
  - **Price risers show the real start AND end price, within that season only**: each player's own first and last gameweek price WITHIN the selected season (never their current 2026-27 price, and never mixed across seasons — checked directly: 2017-18's risers correctly show Salah at £9.0m→£10.6m from that actual season, not compared against any later season's price). A player's true season-start price is their value at THEIR OWN first gameweek that season, not a fixed "GW1" lookup — checked directly: 151 players in 2025-26 have no GW1 row at all (mid-season signings, promoted-team players not yet in the dataset that early), so a fixed-GW1 comparison would have silently produced undefined deltas for them.
- **Team of the Season** — a LOOK BACK at an already-completed season (or a narrower gameweek window — "Team of the Week" — via a slider), not a prediction: the optimal squad ranked by each player's real **total points scored** across the window, at their final-gameweek price, rendered on the same real FPL-style pitch layout (real headshots, formation read from the squad) every squad view in this app uses. Ranked by total, not a rate — a great points-per-game over a handful of appearances can't outrank a player who played most of the window and produced far more for a real squad. `points_per_game` (matching FPL's own published metric exactly) is shown on each card as context, not as what drives selection.
- **Model Performance** — the validation table and played-vs-full-dataset breakdown from Model Training, read live from `models/metrics.json` (written by `train.py`'s own last run), not hardcoded — retraining the model keeps this tab honest automatically. Shows an explicit "run train.py first" message if that file doesn't exist yet.
- **Past Seasons** — season-by-season totals for 2016-17 through 2025-26, read live from the collector's own saved `entry/{id}/history.json` (not a hardcoded table — the tab used to carry a copy-pasted static snapshot that would silently go stale, fixed), the same shape as the standalone [manager history page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html) (which still carries its own hardcoded copy, since it's statically hosted on GitHub Pages and can't run this project's Python collector).

**Squad Builder was removed from this page entirely.** It originally had five modes (Historical gameweek, Team of the Season, 2026-27 pre-season, Scout Picks, manual entry) — once the live season actually started, "build an optimal squad for an already-played historical gameweek" and the pre-season-framed modes had no real purpose left: there's nothing to plan for a gameweek that's already happened, and Home now has the manager's actual live squad for real decisions. Only Team of the Season survived, since it's a genuine standalone historical-analysis question ("who actually performed best"), not a demo of live functionality — promoted to its own tab. The removed pre-season/Scout Picks player-pool logic (`preseason_pool()`, `scout_picks_pool()` in `shared.py`) stays in the codebase for potential reuse (e.g. Scout Picks fits Home's "what should I do" framing better than this page's "what actually happened" framing) but isn't wired into either page's UI right now.

**Why two pages, not one crowded tab bar**: once the season actually started, the dashboard had two genuinely different jobs — "what should I do this gameweek" (needs to be immediate, uncluttered) and "how does this project work / prove it against real history" (a slower, browsable methodology tour) — that a single seven-tab page increasingly conflated. Split via Streamlit's native `pages/` mechanism (one app, one deploy, shared `shared.py` module) rather than a separate deployment.

**`data/raw/` fallbacks**: it's gitignored (personal manager data), so a fresh deploy falls back to `data/dashboard_entry_history.json`/`data/dashboard_entry_info.json` (past-seasons/name), `data/dashboard_current_squad.json` (My Squad), and `data/dashboard_leagues.json` (League Tracker) — single, deliberately-committed copies of this same manager's data, same fallback pattern as `data/dashboard_bootstrap.json`, not a new privacy decision since this exact data is already public on the linked manager-history page. The current-squad and league fallbacks are refreshed daily by the scheduled collector workflow (see Automated Collection above), so they're never more than a day stale.

**A real, verified collector bug fixed once the season actually started**: `snapshot.py`'s entry-picks fetch was gated on the SAME condition as the expensive league-wide 587-player stats fetch (`data_checked`, which only flips a day or two after a gameweek fully finishes) — but a manager's own picks/points are available, if provisional, the moment that gameweek's deadline passes. Confirmed directly: GW1 returned real picks/points via the live API while `data_checked` was still `false`. Fixed with a separate `_latest_live_gw()` check (`finished` OR `is_current`, not `finished AND data_checked`) so a manager's own squad/points/league standings are fetched immediately rather than being invisible for days after they were actually available.

**Known gap:** the pre-season/manual modes' zero-prediction players (~270 of 587 — new signings and promoted-team players with no 2025-26 Premier League record) could in principle use their stats from wherever they played last season (Championship, or another country's top flight), but there's no free, reliable way to get that right now. Researched directly rather than assumed: FPL's own API has no field for a player's previous club/league at all; the one genuinely free, no-registration data source checked (StatsBomb's open data) has no Championship coverage and nothing recent for other leagues; FBref sits behind an active Cloudflare bot-challenge that blocks automated access outright, including to `robots.txt`, so scraping it isn't something to build against. The remaining realistic option (football-data.org or similar) needs a registered API key with unverified free-tier depth for player-level (not just match-result-level) stats — see Future Improvements.

## Manager History

**[Live page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)** — a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Source: [`docs/my-fpl-history.html`](docs/my-fpl-history.html). The same data is also shown inline in the [Dashboard](#dashboard)'s Manager History tab. Both are hardcoded from a point-in-time snapshot rather than fetched live — see Future Improvements for reading directly from the collector's saved history instead.

## Future Improvements

- The remaining gap to FPL's xP is concentrated in non-playing rows (see Model Training) — likely needs a real availability/team-news signal (injury status, press-conference reports, starting-XI news close to the deadline) rather than more historical-stats feature engineering, since that gap doesn't look like a modeling problem.
- Try adding `value` (price) as a feature and tuning LightGBM's hyperparameters on the played-only subset, where the model is already closer to FPL's baseline.
- Consider a secondary model or extra features using xG/xA for 2022-23 onward once the core model is validated, since that signal is only available for a third of the training window.
- Run the model against the 2025-26 final holdout only once no further tuning decisions remain, to get an honest read on generalization.
- Historical-mode Squad Builder/Transfers/Chip Advisor now score players with the trained single-stage xP model (incorporates `fixture_difficulty`, team/opponent form, recent minutes — see Model Training), not a rolling average. Pre-season/manual-entry mode still uses last season's closing rolling-5 form, deliberately NOT run through the model, since there's no genuine 2026-27 fixture list yet to supply `fixture_difficulty` from (see `preseason_pool()`'s docstring) — feeding it a fabricated fixture input would look like a real prediction while not being one. Once the 2026-27 season starts and the collector has real per-gameweek fixtures, wire the model into pre-season/manual mode too.
- Cross-league prior-season stats for the pre-season mode's ~270 zero-prediction players (new signings, promoted-team players). Researched, not just assumed unavailable: FPL's API has no prior-club/league field at all; StatsBomb's free open data has no Championship coverage and nothing recent for other leagues; FBref actively blocks automated access via a Cloudflare bot-challenge (including `robots.txt`), so it's not a source to scrape against. Would need a registered API key (e.g. football-data.org) with unverified free-tier depth for player-level stats, not just match results — needs a human to create that account before this is buildable.
- **Partially done:** cross-checking the dashboard's squads against FPL's own published "Dream Team". `load_live.py` now preserves `in_dreamteam` (and `defensive_contribution`, `starts`, real `expected_goals`/`expected_assists`) for every 2026-27+ gameweek this project's collector captures — see Historical Training Data above. Still needed: once real gameweeks are collected, surface `in_dreamteam` in the dashboard itself (e.g. a checkmark next to Team of the Week's picks showing whether FPL's own Dream Team agreed) and consider retraining the xP model with `defensive_contribution`/FPL's own xG/xA as added features once enough live gameweeks exist to make that worthwhile (a handful of gameweeks isn't enough signal to retrain on yet).
- Build a real multi-gameweek lookahead for Wildcard timing — the current chip advisor explicitly flags its Wildcard suggestion as a weaker single-gameweek-gap signal, not the multi-week strategic value a permanent squad change actually unlocks.
- Track chip/free-transfer availability across a season (which chips are still unused this half, how many free transfers are currently banked) rather than requiring the caller to pass those in by hand each time — the dashboard's transfer tab currently asks the user to enter free transfers manually every time.
- **Partially done:** the dashboard's Manager History tab now reads live from the collector's saved entry history (both past-season totals and 2026-27's own gameweek-by-gameweek progress) instead of a hardcoded table — see Dashboard above. `docs/my-fpl-history.html` still carries its own hardcoded copy, since it's statically hosted on GitHub Pages and can't run this project's Python collector; would need a small build step (e.g. a GitHub Action that regenerates the static page's embedded data from the collector's output on each run) to close that gap too.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- Deploy the dashboard (Streamlit Community Cloud, matching the pattern used for other projects in this account's [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo) so it's a live link, not just something run locally.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
