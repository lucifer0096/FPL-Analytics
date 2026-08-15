# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API. Historical training data comes from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), which is actively maintained (contrary to this project's original premise — its README previously carried a stale "archived after 2024-25" notice; the maintainer has since posted 3 major updates a season instead of weekly ones, most recently adding full 2025-26 data and initial 2026-27 data). Going forward, this project's own [collector](#running-the-collector) captures each gameweek as it happens, since even an actively maintained third-party archive is still a dependency this project shouldn't rely on indefinitely.

**[View the manager history page →](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)**

## Status

**Stage 1 (done): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history. Runs dynamically via GitHub Actions — see [Automated collection](#automated-collection) below.

**Stage 2 (done): expected-points (xP) model.** 253,578 player-gameweek rows across 10 tracked seasons (2016-17 to 2025-26), with rolling form, availability, team form, fixture difficulty, and new-player-baseline features, and a trained/validated LightGBM model. See [Historical Training Data](#historical-training-data), [Feature Engineering](#feature-engineering), and [Model Training](#model-training) below.

**Stage 3 (done): squad optimizer.** Squad builder, transfer optimizer, and chip-timing advisor, all encoding FPL's real rules (verified against the live API, not assumed) and tested against real historical data. See [Squad Optimizer](#squad-optimizer) below.

**Stage 4 (done): dashboard.** A Streamlit app tying the model, optimizer, and manager history together. See [Dashboard](#dashboard) below.

Planned next: wire live 2026-27 predictions into the optimizer/dashboard once the season starts (first fixture 21 Aug 2026).

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
│       ├── train.py           # Chronological train/validation split, LightGBM model
│       ├── optimizer.py       # Squad builder + transfer optimizer (PuLP)
│       ├── chips.py           # Chip-timing advisor (Bench Boost/Triple Captain/Free Hit/Wildcard)
│       ├── test_optimizer.py  # Squad/transfer optimizer checks against real data
│       └── test_chips.py      # Chip advisor checks against real historical data
├── data/
│   ├── raw/                # Gitignored — raw API snapshots, regenerate anytime
│   └── processed/          # Gitignored — historical_gw.parquet, features.parquet
├── app/
│   └── app.py              # Streamlit dashboard (model, optimizer, chips, manager history)
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

`optimize_transfers()` solves for the transfer(s) — if any — that maximize `predicted_points_gained − 4 × paid_transfers`, holding the same squad constraints. Applies the 50%-sell-fee rule via a `sell_price_col` parameter, since FPL doesn't refund a risen player's full current price.

`load_latest_prices()` always pulls the most recent collector-written bootstrap snapshot, never a cached DataFrame — prices move week to week based on transfer momentum, so a transfer's budget math needs current prices, not last week's.

Verified against real data (2025-26 GW20 squad → GW21 pool, 1 free transfer available): correctly proposes swapping James Tarkowski (DEF, £5.7m, rolling avg 4.6 pts) for Nathan Collins (DEF, £4.9m, rolling avg 7.0 pts) — a cheaper upgrade with better recent form — using the free transfer at zero cost, for a net +2.4 predicted points.

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

A Streamlit app with six tabs:
- **Overview** — pipeline summary and headline stats.
- **Squad Builder** — pick a historical season/gameweek, build the optimal squad for it.
- **Transfers** — using the squad just built, find the best transfer(s) into the following gameweek.
- **Chip Advisor** — using the squad just built, rank the next few gameweeks for Bench Boost, Triple Captain, and Free Hit.
- **Model Performance** — the validation table and played-vs-full-dataset breakdown from Model Training.
- **Manager History** — the same season-by-season points/rank chart as the standalone [manager history page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html), inline.

**Every tool in this app currently runs against historical seasons, not live 2026-27 data** — the season hasn't started yet (first fixture 21 Aug 2026), so there's no rolling form or fixture history for any player this season. The Overview tab states this explicitly rather than silently showing meaningless pre-season numbers. Squad Builder, Transfers, and Chip Advisor are chained via session state — build a squad in one tab, and the other two operate on that same squad — the same pattern used throughout this project's test scripts (`test_optimizer.py`, `test_chips.py`).

Verified end-to-end: ran the app headlessly (clean startup, no traceback), and separately exercised all three interactive code paths (squad build, transfer optimization, chip suggestions) outside Streamlit to confirm no runtime errors, reproducing the same real results seen in earlier testing.

## Manager History

**[Live page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)** — a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Source: [`docs/my-fpl-history.html`](docs/my-fpl-history.html). The same data is also shown inline in the [Dashboard](#dashboard)'s Manager History tab. Both are hardcoded from a point-in-time snapshot rather than fetched live — see Future Improvements for reading directly from the collector's saved history instead.

## Future Improvements

- The remaining gap to FPL's xP is concentrated in non-playing rows (see Model Training) — likely needs a real availability/team-news signal (injury status, press-conference reports, starting-XI news close to the deadline) rather than more historical-stats feature engineering, since that gap doesn't look like a modeling problem.
- Try adding `value` (price) as a feature and tuning LightGBM's hyperparameters on the played-only subset, where the model is already closer to FPL's baseline.
- Consider a secondary model or extra features using xG/xA for 2022-23 onward once the core model is validated, since that signal is only available for a third of the training window.
- Run the model against the 2025-26 final holdout only once no further tuning decisions remain, to get an honest read on generalization.
- Wire the trained model's live-gameweek predictions into the optimizer/transfer/chip modules AND the dashboard once the 2026-27 season starts and the collector has real per-gameweek data to predict from — currently everything runs against historical data or a meaningless pre-season player pool.
- Build a real multi-gameweek lookahead for Wildcard timing — the current chip advisor explicitly flags its Wildcard suggestion as a weaker single-gameweek-gap signal, not the multi-week strategic value a permanent squad change actually unlocks.
- Track chip/free-transfer availability across a season (which chips are still unused this half, how many free transfers are currently banked) rather than requiring the caller to pass those in by hand each time — the dashboard's transfer tab currently asks the user to enter free transfers manually every time.
- Make the dashboard's Manager History tab and the standalone `docs/my-fpl-history.html` page both read live from the collector's saved entry history instead of each carrying its own hardcoded season data.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- Deploy the dashboard (Streamlit Community Cloud, matching the pattern used for other projects in this account's [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo) so it's a live link, not just something run locally.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
