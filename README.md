# FPL Analytics

A Fantasy Premier League expected-points model, squad optimizer, and dashboard, built on the free public FPL API. **FPL's own live API is this project's actual basis, not vaastav's archive** — vaastav/Fantasy-Premier-League is used ONLY as a historical bootstrap for 2016-17 through 2025-26, because that's the one thing the live API genuinely cannot provide: verified directly against the API that once a season ends, `element-summary`'s per-gameweek `history` empties out and `history_past` only ever returns SEASON-TOTAL aggregates (total_points, minutes, etc. summed for the whole season) — there is no way, official or otherwise, to pull old seasons' gameweek-by-gameweek data from FPL itself, so a third-party archive is the only source for that window. Every season from 2026-27 onward is captured entirely by this project's own [collector](#running-the-collector) as it happens, straight from the live API, with no vaastav dependency at all — and it captures MORE than vaastav's schema ever could: fields like `in_dreamteam`, `defensive_contribution` (part of FPL's 2025-26 scoring overhaul), `starts`, and real `expected_goals`/`expected_assists` that vaastav's CSVs simply don't carry for any season (see `load_live.py`).

**[Open the live dashboard →](https://fpl-analytics-dashboard.streamlit.app/)**

**[View the manager history page →](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)**

## Quickstart

```bash
pip install -r requirements.txt   # exact tested pins — Python 3.12+ (3.14 matches CI and Streamlit Cloud)

streamlit run app/app.py          # dashboard; also runs fully offline via the committed data/dashboard_* fallbacks

python src/collector/snapshot.py  # refresh data from the live FPL API (--check-only / --bootstrap-only / --force)
export FPL_ENTRY_ID=1132016       # optional: also snapshot your own team's history, picks and leagues

pytest -q                         # deterministic tests + offline dashboard render tests (~10s, no network)
pytest -m live                    # live FPL-API checks (needs network)
ruff check app src                # lint — the same rule set CI gates on
```

The dashboard is [deployed on Streamlit Community Cloud](https://fpl-analytics-dashboard.streamlit.app/) and the manager-history page runs on GitHub Pages (see [Manager History](#manager-history)). Dependency layout: `requirements.txt` holds the exact pins everything installs, `requirements.in` the ranges to re-resolve from, `requirements-dev.txt` dev-only tools.

## Status

**Stage 1 (done): data collector.** A lightweight client for the official FPL API (`bootstrap-static`, `element-summary`, `fixtures`, `entry`) that snapshots each gameweek's data to disk as the season progresses, since the live API only exposes current state, not history. Runs dynamically via GitHub Actions — see [Automated collection](docs/COLLECTOR.md#automated-collection).

**Stage 2 (done): expected-points (xP) model.** 253,578 player-gameweek rows across 10 tracked seasons (2016-17 to 2025-26), with rolling form, availability, team form, fixture difficulty, and new-player-baseline features, and a trained/validated LightGBM model. See [Historical Training Data](#historical-training-data), [Feature Engineering](#feature-engineering), and [Model Training](#model-training) below.

**Stage 3 (done): squad optimizer.** Squad builder, transfer optimizer, and chip-timing advisor, all encoding FPL's real rules (verified against the live API, not assumed) and tested against real historical data. See [Squad Optimizer](#squad-optimizer) below.

**Stage 4 (done): dashboard.** A two-page Streamlit app: a Home page for the manager's real, live 2026-27 squad/points/league standings, and a Historical & Model page for demo squad-building modes and methodology. See [Dashboard](#dashboard) below.

**Stage 5 (in progress): live season.** The 2026-27 season started 21 Aug 2026 — the collector now captures real gameweek data as it happens (see the `_latest_live_gw` fix in [Dashboard](#dashboard) below), and the Home page shows the manager's actual squad/points/transfers/league standings rather than a demo.

Planned next: retrain the xP model once enough live 2026-27 gameweeks exist to be worth incorporating (currently trained on 2016-17 through 2025-26 only); build a real multi-gameweek Chip Advisor projection for the live squad once there's a genuine upcoming-fixtures window to project against.

The live-season training boundary is now future-safe: feature engineering accepts newly collected `YYYY-YY` seasons, while validation training uses only seasons before the configured validation season. Later live seasons remain available for a deliberately separate forward-training decision rather than silently contaminating the historical validation result.

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
│   └── llm/
├── data/
│   ├── raw/                # Gitignored — raw API snapshots, regenerate anytime
│   └── processed/          # Gitignored — historical_gw.parquet, features.parquet
├── app/
│   ├── app.py              # Streamlit dashboard, Home page -- live squad, transfers, chips, league tracker
│   ├── shared.py           # Data-loading/pool-building/pitch-rendering helpers shared by every page
│   ├── test_shared_live.py # Live-sync checks against real FPL data (see Automated Collection)
│   ├── test_app_offline.py # Offline dashboard render tests via AppTest — no network needed (see Known Issues)
│   └── pages/
│       └── 1_Historical_and_Model.py  # Second page -- demo Squad Builder modes, model metrics, past seasons
├── docs/
│   ├── my-fpl-history.html # Manager history page, served via GitHub Pages
│   ├── COLLECTOR.md        # Deep-dive: every file the collector writes, scheduling, automation, live vs fallback data
│   ├── MODELING.md         # Deep-dive: training data, feature engineering, model training, optimizer internals
│   ├── DASHBOARD.md        # Deep-dive: tab-by-tab dashboard documentation and design rationale
│   └── KNOWN-ISSUES.md     # Deep-dive: dated investigation log of bugs found and fixed
├── notebooks/               # EDA and model development
├── models/                  # Gitignored — trained model artifacts
├── requirements.in          # Top-level dependency ranges (source of truth for re-pinning)
├── requirements.txt         # Exact tested pins — what CI, the collector and Streamlit Cloud install
├── requirements-dev.txt     # Dev-only tools (ruff); never part of the deploy
├── pytest.ini               # Test discovery; live-API tests excluded from the default run
├── ruff.toml                # Pinned lint rules for `ruff check app src`
├── .gitattributes           # LF line endings for every text file (no CRLF drift)
└── .github/                 # CI + scheduled collector workflows
```

## Data Sources

- **Official FPL API** (free, public, no auth): `bootstrap-static` for all players/teams, `element-summary/{id}` for per-gameweek player history, `fixtures` for the season schedule, `entry/{id}` for a manager's team/history/picks.
- **Historical seasons (2016-17 to 2025-26)**: sourced from the vaastav dataset (`E:\Fantasy-Premier-League`, cloned directly from `vaastav/Fantasy-Premier-League` — not a fork, and not part of this repo) for model training, since the FPL API itself only exposes the current season's gameweek-by-gameweek data. 2026-27 is captured by this project's own collector as it happens (see `src/model/load_live.py`), starting from the season's first gameweek.

## Running the Collector

`pip install -r requirements.txt`, optionally `export FPL_ENTRY_ID=...`, then `python src/collector/snapshot.py`. Full documentation — every file it writes, dynamic `--check-only` scheduling, the 30-minute GitHub Actions automation, and what is genuinely live vs. batch-refreshed: **[docs/COLLECTOR.md](docs/COLLECTOR.md#running-the-collector)**.

## Historical Training Data

The 10 vaastav seasons unified with this project's own live 2026-27+ gameweeks, and the five silent data-quality bugs fixed along the way: **[docs/MODELING.md](docs/MODELING.md#historical-training-data)**.

## Feature Engineering

Rolling form, availability, team form, fixture difficulty and new-player-baseline features — built leak-free, and verified that way: **[docs/MODELING.md](docs/MODELING.md#feature-engineering)**.

## Model Training

LightGBM on a chronological split (2024-25 validation, 2025-26 untouched as the final holdout), single- vs two-stage architectures, leak-free baselines. Current result: **MAE 0.986 vs 1.052 for the naive baseline, against FPL's own xP at 0.904** — the remaining gap is a data problem, not a modeling one: **[docs/MODELING.md](docs/MODELING.md#model-training)**.

## Squad Optimizer

FPL's real rules (verified against the live API, not assumed), the PuLP squad builder, the per-transfer-gated transfer optimizer, and the chip advisor — each verified against real data: **[docs/MODELING.md](docs/MODELING.md#squad-optimizer)**.

## Dashboard

Two-page Streamlit app — **`streamlit run app/app.py`**, or open the **[live deployment →](https://fpl-analytics-dashboard.streamlit.app/)**.

- **Home (live, current season)** — 8 tabs: My Squad (your real picks + live points), Transfers (gated optimizer, real free-transfer tracking, injury flags, differentials), Chip Advisor (real chip availability), League Tracker (real private-league standings), Price Changes (risers/fallers + tonight's projections), PL Table, Season Insights, Fixtures & Results. Every number is live-first from FPL's API with a 60s cache, falling back to the committed files only if the API is unreachable.
- **Historical & Model** — completed-season insights, Team of the Season, Model Performance (live from `models/metrics.json`), Past Seasons (live from the collector's saved history).

Tab-by-tab detail, look-and-feel notes and the two-page design rationale: **[docs/DASHBOARD.md](docs/DASHBOARD.md#dashboard)**.

## Known Issues Found & Fixed

The investigation log — 15+ real bugs found and fixed with direct evidence (stale live data, silent data corruption, deploy-time crashes, UX gaps), each with how it was caught and the permanent regression guard left behind: **[docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md#known-issues-found--fixed)**.

## Manager History

**[Live page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)** — a static page charting one manager's points and overall rank across all 10 tracked seasons (2016/17–2025/26), pulled from `entry/{id}/history`. Source: [`docs/my-fpl-history.html`](docs/my-fpl-history.html). The same data is also shown inline in the [Dashboard](#dashboard)'s Manager History tab. Both are hardcoded from a point-in-time snapshot rather than fetched live — see Future Improvements for reading directly from the collector's saved history instead.

## Future Improvements

- The remaining gap to FPL's xP is concentrated in non-playing rows (see Model Training) — likely needs a real availability/team-news signal (injury status, press-conference reports, starting-XI news close to the deadline) rather than more historical-stats feature engineering, since that gap doesn't look like a modeling problem.
- Try adding `value` (price) as a feature and tuning LightGBM's hyperparameters on the played-only subset, where the model is already closer to FPL's baseline.
- Consider a secondary model or extra features using xG/xA for 2022-23 onward once the core model is validated, since that signal is only available for a third of the training window.
- Run the model against the 2025-26 final holdout only once no further tuning decisions remain, to get an honest read on generalization.
- Historical-mode Squad Builder/Transfers/Chip Advisor now score players with the trained single-stage xP model (incorporates `fixture_difficulty`, team/opponent form, recent minutes — see Model Training), not a rolling average. Pre-season/manual-entry mode still uses last season's closing rolling-5 form, deliberately NOT run through the model, since there's no genuine 2026-27 fixture list yet to supply `fixture_difficulty` from (see `preseason_pool()`'s docstring) — feeding it a fabricated fixture input would look like a real prediction while not being one. Once the 2026-27 season starts and the collector has real per-gameweek fixtures, wire the model into pre-season/manual mode too.
- Cross-league prior-season stats for the pre-season mode's ~270 zero-prediction players (new signings, promoted-team players). Researched, not just assumed unavailable: FPL's API has no prior-club/league field at all; StatsBomb's free open data has no Championship coverage and nothing recent for other leagues; FBref actively blocks automated access via a Cloudflare bot-challenge (including `robots.txt`), so it's not a source to scrape against. Would need a registered API key (e.g. football-data.org) with unverified free-tier depth for player-level stats, not just match results — needs a human to create that account before this is buildable.
- **Done:** cross-checking the dashboard's squads against FPL's own published "Dream Team". `load_live.py` preserves `in_dreamteam` (and `defensive_contribution`, `starts`, real `expected_goals`/`expected_assists`) for every 2026-27+ gameweek this project's collector captures, and `build_live_squad_df()` now attaches `in_dreamteam` to My Squad's own real pitch cards — a real ✓ badge (see Dashboard above). Remaining: consider retraining the xP model with `defensive_contribution`/FPL's own xG/xA as added features once enough live gameweeks exist to make that worthwhile (a handful of gameweeks isn't enough signal to retrain on yet).
- Build a real multi-gameweek lookahead for Wildcard timing — the current chip advisor explicitly flags its Wildcard suggestion as a weaker single-gameweek-gap signal, not the multi-week strategic value a permanent squad change actually unlocks.
- **Done:** track chip availability across a season — `chip_usage_status()` shows real Wildcard/Free Hit/Bench Boost/Triple Captain usage on Chip Advisor, straight from FPL's own entry history (see Dashboard above). Free-transfer availability was already tracked (`calculate_free_transfers()`); Transfers now also shows real transfers-made-this-gameweek alongside it.
- **Partially done:** the dashboard's Manager History tab now reads live from the collector's saved entry history (both past-season totals and 2026-27's own gameweek-by-gameweek progress) instead of a hardcoded table — see Dashboard above. `docs/my-fpl-history.html` still carries its own hardcoded copy, since it's statically hosted on GitHub Pages and can't run this project's Python collector; would need a small build step (e.g. a GitHub Action that regenerates the static page's embedded data from the collector's output on each run) to close that gap too.
- Once there's a processed, league-wide dataset (no personal data), commit it back to the repo each run like NZ-Jobs-Dashboard's sync workflow does, rather than only uploading artifacts.
- **Done:** the dashboard is deployed on Streamlit Community Cloud as a live link, not just run locally.
- **Done:** deployment dependencies can no longer drift — `requirements.txt` used to be unpinned ranges (`pandas>=2.0` etc.), so the Streamlit Cloud deploy could resolve newer majors than CI's separately-pinned file ever tested. Both now install the same exact pins (`requirements.in` holds the ranges for the next re-resolve; the old `requirements.lock` was folded in as redundant duplication), with a weekly Dependabot job (`.github/dependabot.yml`) opening PRs against both the Python pins and the workflows' `actions/*` versions instead of letting them age silently.
- **Done:** the dashboard itself is under test in CI — `app/test_app_offline.py` renders the Home page (all 8 tabs), clicks the Transfers solve through both the gated and unlimited paths, and renders the Historical page via Streamlit's `AppTest`, all with every live FPL call forced to fail and `data/raw/` absent — the exact fresh-checkout/deploy condition. This is the test that would have caught the Transfers `KeyError: 'sell_price'` (see Known Issues) before it shipped. A companion static check pins `shared.py`'s import contract, because a lint autofix that deletes an "unused" name there (only `app.py` imports it *through* shared) surfaces merely as an `ImportError` at startup.
- **Done:** `pytest.ini` gives a bare `pytest -q` one consistent meaning — `testpaths = src app` (previously `test_optimizer.py`/`test_chips.py` silently collected zero tests and were only ever run as scripts), plus `-m "not live"` by default, with everything that talks to FPL's real API marked `live` for explicit `pytest -m live` runs.
- **Done:** `ruff check app src` gates every push (rule set pinned in `ruff.toml` so local and CI agree) — first run found two dead result-variables (one in the transfer optimizer's gating loop), five ambiguous `l` names, and 45 unused-import/placeholder-f-string findings across the tree.
- **Done:** `.gitattributes` forces LF line endings — this checkout lives on a Windows mount where editors default to CRLF, so git had been storing CRLF bytes verbatim: every tracked file showed as permanently modified and any commit became a whole-file diff.
- **Done:** CI skips pushes touching only `data/**`/`models/**` — the collector commits fallback refreshes to `main` every 30 minutes, and each was queueing a full test run that exercised no code.
- **Done:** replaced all 26 deprecated `st.dataframe(..., use_container_width=True)` calls with the documented equivalent `width="stretch"` (also `st.dataframe`'s default), clearing the deprecation warnings every page render emitted on Streamlit ≥1.63.
- **Done:** a full FPL-enthusiast UI pass over the Home page, all CSS-only and both-theme/reduced-motion safe — styled deadline decision bar (pulsing LIVE pill while a gameweek runs), transfer proposal cards (OUT → IN with a per-pair xP-delta badge, Free/Hit/Unlimited state chips and next-3 fixture difficulty chips), real +/- deltas on My Squad's headline metrics (GW points vs FPL's real all-managers average; overall-rank places moved), glassy chip-availability cards, a League Tracker **Move since last GW** column (FPL's own `last_rank`, 🥇🥈🥉 medals, your-rank delta), value bars + medals on every Season Insights leaderboard (pure-CSS `Styler.bar`, no new dependency), and "✓ n of yours" pills on Fixtures & Results. The offline AppTest suite pins the proposal cards and fixture pills to real element markup rather than the injected CSS class names, so a silently-omitted widget fails CI instead of passing on a substring match. See Dashboard above.
- Split `app/app.py` (1,200+ lines of tab renderers) into per-tab modules, and add `ruff format` as a CI step — deliberately not started: reformatting the whole tree in the same pass as functional changes would bury the latter in churn, so lint (correctness rules only) is what's enforced today.
- Once the model and dashboard are solid, port a clean version of this project into the main [data-portfolio](https://github.com/lucifer0096/data-portfolio) repo.
