# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API. **FPL's own live API is this project's actual basis, not vaastav's archive** — vaastav/Fantasy-Premier-League is used ONLY as a historical bootstrap for 2016-17 through 2025-26, because that's the one thing the live API genuinely cannot provide: verified directly against the API that once a season ends, `element-summary`'s per-gameweek `history` empties out and `history_past` only ever returns SEASON-TOTAL aggregates (total_points, minutes, etc. summed for the whole season) — there is no way, official or otherwise, to pull old seasons' gameweek-by-gameweek data from FPL itself, so a third-party archive is the only source for that window. Every season from 2026-27 onward is captured entirely by this project's own [collector](#running-the-collector) as it happens, straight from the live API, with no vaastav dependency at all — and it captures MORE than vaastav's schema ever could: fields like `in_dreamteam`, `defensive_contribution` (part of FPL's 2025-26 scoring overhaul), `starts`, and real `expected_goals`/`expected_assists` that vaastav's CSVs simply don't carry for any season (see `load_live.py`).

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
- `gw_history.csv` — one row per player per finished gameweek (empty until gameweeks have been played), with EVERY field the live API returns per gameweek (not a fixed subset) — including `in_dreamteam`, `defensive_contribution`, `starts`, and real `expected_goals`/`expected_assists`, none of which vaastav's historical data has for any season (see Historical Training Data above). `src/model/load_live.py` reads this file back for feature engineering/training.
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

A Streamlit app with six tabs:
- **Overview** — pipeline summary and headline stats.
- **Squad Builder** — five modes, all rendered on a real FPL-style pitch layout (starting XI positioned by row — GK, then DEF/MID/FWD moving up the pitch, formation read directly from the squad — with the 4 bench players shown separately below; the full table is also available in a collapsed expander). The squad's single highest-predicted_points player gets a ⭐ badge — labeled **MVP** (most points scored that season/window) for a Team of the Season squad, or **Player of the Week** (top predicted scorer that gameweek) for a Historical gameweek squad. A separate ✓ badge marks FPL's own official Team of the Week picks (`in_dreamteam` — see Historical Training Data) wherever that data exists (2026-27+ gameweeks the collector has captured; historical seasons never have it). No ⭐ badge is shown for pre-season/manual-entry squads, since predicted_points there is last season's closing form, not a real per-gameweek or full-season figure — an MVP/POTW label would overstate what that number means.
  - *Historical gameweek* — pick a past season/gameweek, build the optimal squad for it. predicted_points comes from the trained xP model (fixture difficulty, team/opponent form, recent minutes — see Model Training), since this simulates "what would you do at this specific gameweek," a genuinely forward-looking question even though the gameweek itself is in the past.
  - *Team of the Season* — a LOOK BACK at an already-completed season (or a narrower gameweek window — "Team of the Week" — via a slider), not a prediction (already-played gameweeks, so there's nothing left to predict): the optimal squad ranked by each player's real **total points scored** across the window, at their final-gameweek price. Ranked by total, not a rate — a great points-per-game over a handful of appearances can't outrank a player who played most of the window and produced far more for a real squad, so no budget-cap-free "Team of the Season" here ever ends up with someone on ~50 points just because their rate looked good in a small sample. `points_per_game` — matching FPL's own published metric exactly (total points ÷ appearances, an appearance being any gameweek with minutes > 0) — is shown on each card as context, not as what drives selection, and only once a player has enough appearances (scaled to the window size) to make the rate trustworthy.
  - *2026-27 pre-season* — build a squad recommendation for the season opener using each player's **live current price** (pulled from the latest collector snapshot) paired with their **rolling-5 average at the end of 2025-26** (their most recent known real form) as a predicted-points stand-in. Joined on `player_code`, since prices come from this season's numeric ids while form comes from last season's. Players with no 2025-26 Premier League record (new signings, promoted-team players) get 0 rather than a fabricated guess — see the caveat below. Deliberately NOT run through the xP model, unlike Historical gameweek — there's no real 2026-27 fixture list yet to supply `fixture_difficulty` from.
  - *Scout Picks (2026-27 season opener)* — this project's own take on FPL's editorial "Scout Picks" article, NOT a scrape of it: checked directly that it isn't structured API data (it's prose commentary) and has no stable weekly URL (each article gets an unpredictable numeric id, and the listing page needs JS rendering this project's lightweight `urllib`-based collector can't do). Built from `preseason_pool()`'s base pool plus two REAL, verified boosts on top — a genuinely easy GW1 fixture against one of this season's actual promoted teams (confirmed directly by diffing this season's team list against last season's: Hull City, Ipswich Town, Coventry City — the same three teams FPL's own Scout Picks article names for exactly this reason), and confirmed set-piece duty via FPL's own `penalties_order` field (`== 1`, i.e. the club's primary penalty taker). Each boosted player gets a shown, checkable reason (🔍 hover tooltip on their card) — never an unexplained number.
  - *My squad (enter manually)* — pick your own real 15-man squad by name, mark your starting XI, and see it validated (position quotas, budget, per-club limit, valid formation) and rendered the same way as an optimizer-built squad — for checking your actual picks against the same tools, not just seeing what the optimizer would choose instead.
- **Transfers** — using the squad just built, find the best transfer(s). For a *Historical gameweek* squad, checks against the following gameweek's pool; for *pre-season* or *manually entered* squads (no "next gameweek" concept), re-checks against the current live pool instead. Not available for *Team of the Season* squads — there's no "next gameweek" for an already-completed season to transfer into.
- **Chip Advisor** — using the squad just built, rank the next few gameweeks for Bench Boost, Triple Captain, and Free Hit. Needs a real multi-gameweek future window, so only available for *Historical gameweek* squads.
- **Model Performance** — the validation table and played-vs-full-dataset breakdown from Model Training, read live from `models/metrics.json` (written by `train.py`'s own last run), not hardcoded — retraining the model (e.g. once live 2026-27 data joins the training set) keeps this tab honest automatically, with no separate number to remember to update by hand. Shows an explicit "run train.py first" message if that file doesn't exist yet, rather than silently falling back to stale numbers.
- **Manager History** — two sections, both read live from the collector's own saved `entry/{id}/history.json`, not a hardcoded table (the tab used to carry a copy-pasted static snapshot that would silently go stale — fixed): a **2026-27 live progress** section (gameweek-by-gameweek rank/points for the season actually in progress, empty until the collector captures a finished gameweek — the live API only ever exposes this level of detail for the CURRENT season, never retroactively for a finished one, same limitation noted in Historical Training Data) and a **past seasons** table/chart (season-by-season totals for 2016-17 through 2025-26, the same shape as the standalone [manager history page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html) — that page still carries its own hardcoded copy of this same data, since it's statically hosted on GitHub Pages and can't run this project's Python collector). `data/raw/` (where the real `history.json` lives) is gitignored, so a fresh deploy falls back to `data/dashboard_entry_history.json` — a single, deliberately-committed, non-timestamped copy of this same manager's history, same fallback pattern as `data/dashboard_bootstrap.json`. Not a new privacy decision: this exact data is already public on the linked manager-history page.

**Historical-gameweek tools run against past seasons, not live 2026-27 results** — the season hasn't started yet (first fixture 21 Aug 2026), so there's no rolling form or fixture history for any player *this* season. Pre-season mode and manual entry are the honest middle ground: real current prices, real prior-season form, clearly framed as an estimate rather than a live prediction. Squad Builder, Transfers, and Chip Advisor are chained via session state — build a squad in one tab, and the other two operate on that same squad — the same pattern used throughout this project's test scripts (`test_optimizer.py`, `test_chips.py`).

**Known gap:** the pre-season/manual modes' zero-prediction players (~270 of 587 — new signings and promoted-team players with no 2025-26 Premier League record) could in principle use their stats from wherever they played last season (Championship, or another country's top flight), but there's no free, reliable way to get that right now. Researched directly rather than assumed: FPL's own API has no field for a player's previous club/league at all; the one genuinely free, no-registration data source checked (StatsBomb's open data) has no Championship coverage and nothing recent for other leagues; FBref sits behind an active Cloudflare bot-challenge that blocks automated access outright, including to `robots.txt`, so scraping it isn't something to build against. The remaining realistic option (football-data.org or similar) needs a registered API key with unverified free-tier depth for player-level (not just match-result-level) stats — see Future Improvements.

Verified end-to-end: ran the app headlessly (clean startup, no traceback), and separately exercised all interactive code paths (squad build in all three modes, transfer optimization, chip suggestions, pitch rendering) outside Streamlit to confirm no runtime errors, reproducing the same real results seen in earlier testing — e.g. manual-entry mode correctly resolved the exact 15-player squad discussed earlier in this project's development to the same £100.0m total and 2/5/5/3 split confirmed by hand, with accented names (Guéhi, Ekitiké, Rúben dos Santos Gato Alves Dias) rendering correctly throughout.

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
