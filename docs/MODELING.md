# Model & methodology

> Split verbatim from the [README](../README.md) so the front page stays scannable. Heading anchors are unchanged, so former `README.md#<anchor>` links resolve to the same heading in this file.

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

The gap to FPL's xP shrinks from ~0.08 (full dataset) to ~0.07 (played only) — most of the overall gap is concentrated in non-playing rows, where FPL's xP likely draws on real injury/team-news signals (press conferences, training reports) that a model built purely on historical box-score stats has no way to see. This reframes the next step: closing the remaining gap isn't primarily a modeling problem, it's a data problem — see [Future Improvements](../README.md#future-improvements).

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
- **Wildcard** reuses the Free Hit logic but is explicitly flagged as a weaker signal — it only captures one gameweek's gap, not the multi-week strategic value a permanent squad change unlocks. No fully automated Wildcard-timing suggestion exists yet (see [Future Improvements](../README.md#future-improvements)).

Verified against real historical data (2025-26, squad built at GW10, projected across GW10–14): real, plausible players surface as Triple Captain candidates (Haaland 9.6 pts, Gabriel dos Santos Magalhães 11.0 pts), Bench Boost values differ meaningfully by gameweek, and Free Hit correctly identifies GW14 as the week the squad had drifted furthest from optimal (34.0-point gap).
