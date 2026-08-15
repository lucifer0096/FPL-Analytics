"""FPL Analytics dashboard.

The 2026-27 season hasn't started yet (first fixture 21 Aug 2026), so there's
no live gameweek data to predict from -- every tab below uses historical
seasons as a working demonstration of what each tool does, clearly labeled as
such. Once the collector has real 2026-27 gameweek results, the squad/transfer/
chip tools can be pointed at live predictions instead (see this repo's README,
Future Improvements).
"""

import glob
import json
import os
import sys

import lightgbm as lgb
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "model"))

from optimizer import optimize_squad, optimize_transfers, load_latest_prices, select_starting_xi, POSITION_REQUIREMENTS, DEFAULT_BUDGET
from chips import suggest_bench_boost, suggest_triple_captain, suggest_free_hit_or_wildcard
from predict import load_model, predict_points
from train import FEATURE_COLUMNS

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Scoped visual styling only -- no logic here. Colors match the pitch view's
# own green (see render_pitch) so the two don't clash, plus FPL's own purple
# as an accent (their real brand color, not invented). Kept to CSS only, no
# custom components, so it degrades gracefully if Streamlit's internal class
# names shift in a future version -- worst case it looks like plain Streamlit.
#
# IMPORTANT: does NOT hardcode .stApp's background -- an earlier version of
# this forced a dark background unconditionally, which broke the experience
# for anyone using Streamlit's light theme (metric-card gradients tuned for
# dark also looked washed out on light). Every color below is deliberately
# translucent (rgba with a low alpha) or white-text-on-a-solid-gradient
# (app-hero, which is a fixed-color element by design, same as the pitch
# view), so it reads correctly against BOTH themes rather than assuming one.
st.markdown("""
<style>
    h1, h2, h3, h4 { font-family: "Segoe UI", Roboto, sans-serif; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(42,150,80,0.14), rgba(90,60,180,0.12));
        border: 1px solid rgba(42,150,80,0.4);
        border-radius: 10px;
        padding: 12px 14px 8px 14px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.8; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    .app-hero {
        background: linear-gradient(120deg, #1f7a3f 0%, #2a9650 55%, #5a3cb4 130%);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .app-hero h1 {
        color: white;
        margin: 0 0 6px 0;
        font-size: 1.9rem;
    }
    .app-hero p {
        color: rgba(255,255,255,0.92);
        margin: 0;
        font-size: 0.95rem;
    }
    .section-card {
        background: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.15);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

FEATURES_PATH = os.path.join(PROJECT_DIR, "data", "processed", "features.parquet")
HISTORICAL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "historical_gw.parquet")
MANAGER_ENTRY_ID = 1132016

# Streamlit session_state is purely in-memory -- a browser refresh (or the
# server restarting) wipes it entirely, silently discarding a manually-typed
# 15-man squad with no way to recover it. Persisting just the player-id list
# to a small local file lets it survive a refresh; re-picking on a genuinely
# different machine/session is still expected.
MANUAL_SQUAD_SAVE_PATH = os.path.join(PROJECT_DIR, "data", "manual_squad.json")


def _load_saved_manual_squad_ids() -> list:
    if os.path.exists(MANUAL_SQUAD_SAVE_PATH):
        with open(MANUAL_SQUAD_SAVE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_manual_squad_ids(player_ids: list) -> None:
    with open(MANUAL_SQUAD_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(list(player_ids), f)

SEASON_ORDER = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
VALIDATION_SEASON = "2024-25"
FINAL_HOLDOUT_SEASON = "2025-26"


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def _get_xp_model():
    """Cached across reruns (st.cache_resource, not cache_data -- this holds a
    live LightGBM Booster object, not a serializable value). Returns None if
    the model file isn't present (e.g. a fresh clone before running train.py),
    so callers can fall back to the naive rolling-average estimate instead of
    crashing the whole dashboard."""
    try:
        return load_model()
    except lgb.basic.LightGBMError:
        return None


def gw_pool(df: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Build an optimizer-ready player pool for one (season, GW). predicted_points
    comes from the trained single-stage xP model (incorporates fixture_difficulty,
    opponent/team form, recent minutes/form -- see train.py's FEATURE_COLUMNS),
    falling back to the player's own rolling-5 average only if no trained model
    file is present.

    in_dreamteam (FPL's own official "Team of the Week" flag) is carried
    through when present -- only true for gameweeks this project's own
    collector captured live (see load_live.py); vaastav's historical seasons
    never have this column at all, so it's simply absent there, not False for
    every player (that would misrepresent "no data" as "not selected")."""
    sub = df[(df["season"] == season) & (df["GW"] == gw)].copy()
    sub["player_id"] = sub["player_code"]
    sub["cost"] = sub["value"] / 10.0

    model = _get_xp_model()
    if model is not None and set(FEATURE_COLUMNS).issubset(sub.columns):
        sub["predicted_points"] = predict_points(sub, model)
    else:
        sub["predicted_points"] = sub["total_points_avg_last_5"].fillna(0).clip(lower=0)

    cols = ["player_id", "name", "position", "team", "cost", "predicted_points"]
    if "in_dreamteam" in sub.columns:
        cols.append("in_dreamteam")

    return sub.drop_duplicates(subset="player_id")[cols]


MIN_GAMES_FOR_SEASON_RATE = 5  # below this, points-per-game is too noisy to trust as a DISPLAYED rate (a single big haul would otherwise look like a huge per-game rate) -- doesn't affect selection, which ranks by total points regardless of appearances


def min_games_for_window(gw_start: int, gw_end: int) -> int:
    """Scale the minimum-appearances floor to the window size -- MIN_GAMES_FOR_SEASON_RATE
    (5, out of a ~38-week season) only makes sense across a full season. A
    single-gameweek window can have at most 1 appearance, so demanding 5 there
    would exclude every player. Scaled proportionally (min 1 game), same
    "needs a few real appearances to trust the rate" reasoning, just for a
    shorter window."""
    n_gws = max(gw_end - gw_start + 1, 1)
    return max(round(MIN_GAMES_FOR_SEASON_RATE * n_gws / 38), 1)


@st.cache_data
def season_pool(df: pd.DataFrame, season: str, gw_start: int = None, gw_end: int = None) -> pd.DataFrame:
    """Build a 'Team of the Season' (or 'Team of the Week(s)', if gw_start/gw_end
    narrow it to a window) player pool for an ALREADY-COMPLETED season/window --
    this is a look-back at who actually performed best, not a prediction, so it
    uses each player's real total_points and appearances across the gameweeks
    in question, not a rolling window or the xP model (which exist to estimate
    an UNKNOWN future gameweek; there's nothing unknown here).

    predicted_points (what the optimizer actually selects on) is the player's
    real TOTAL points across the window, not a rate -- ranking by a per-game/
    per-90 rate instead would let a player with a great rate over just 5-6
    appearances outrank someone who played 35 games and produced 200+ points,
    which isn't what "who actually performed best for a squad this season"
    means. total_points_sum's own presence with no games-played floor makes
    that impossible: a player who only played 1 game and scored big still only
    contributes that 1 game's points to the squad total, same as reality.

    points_per_game is still computed and returned as a separate, DISPLAY-ONLY
    column -- matches FPL's OWN published `points_per_game` field exactly
    (checked directly against bootstrap-static): total_points / appearances,
    where an "appearance" is any gameweek with minutes > 0 -- NOT minutes
    played (FPL doesn't publish a points-per-90 stat anywhere; a 3-minute
    cameo and a full 90 both count as one game here, same as FPL's own number
    does). Only null for players below the window's appearance floor (see
    min_games_for_window) -- too small a sample makes the RATE unreliable
    (e.g. one big haul off the bench would otherwise look like a huge
    points-per-game rate), even though it doesn't affect selection."""
    sub = df[df["season"] == season].copy()
    if gw_start is not None:
        sub = sub[(sub["GW"] >= gw_start) & (sub["GW"] <= gw_end)]
    last_gw = sub["GW"].max()
    min_games = min_games_for_window(gw_start or 1, gw_end or last_gw)

    sub["_appeared"] = sub["minutes"] > 0
    totals = sub.groupby("player_code").agg(
        total_points_sum=("total_points", "sum"),
        games_played=("_appeared", "sum"),
    )
    totals["points_per_game"] = (totals["total_points_sum"] / totals["games_played"]).where(
        totals["games_played"] >= min_games
    )

    latest = (
        sub[sub["GW"] == last_gw]
        .drop_duplicates(subset="player_code")
        .set_index("player_code")[["name", "position", "team", "value"]]
    )

    merged = totals.join(latest, how="inner").reset_index()
    merged["player_id"] = merged["player_code"]
    merged["cost"] = merged["value"] / 10.0
    merged["predicted_points"] = merged["total_points_sum"]

    return merged[
        ["player_id", "name", "position", "team", "cost", "predicted_points",
         "total_points_sum", "games_played", "points_per_game"]
    ]


@st.cache_data
def preseason_pool(_features_df: pd.DataFrame, prior_season: str = "2025-26") -> pd.DataFrame:
    """Build a 2026-27 pre-season player pool: each player's LIVE current price
    (this season's actual cost, pulled from the latest collector snapshot) paired
    with their predicted_points estimated from their own rolling-5 average at
    the END of the prior season -- their most recent known real form, not
    diluted by a full-season average that includes early-season benching/injury
    spells. Joined on player_code (the stable cross-season id -- see
    load_historical.py), NOT element (resets every season).

    Players present in the live pool with no prior_season record at all (new
    signings from outside the league, promoted-team players with no top-flight
    history) get predicted_points = 0 rather than a guess -- same honest
    treatment as new_player_baseline's fallback for genuinely unknown players,
    just without that feature's league-wide price-band averaging here (a
    simpler stand-in, since this is a demo view, not a training feature).

    Deliberately NOT run through the trained xP model (unlike gw_pool) --
    fixture_difficulty is a real, published-ahead-of-kickoff feature for a
    given match (see features.py), but there IS no 2026-27 fixture list here
    yet, so there's nothing genuine to feed the model for it. Feeding it a
    placeholder (e.g. last season's fixture) would look like a real prediction
    while actually being fabricated -- the honest rolling-average estimate is
    preferable to a model output built on an invented input."""
    prior = _features_df[_features_df["season"] == prior_season].copy()
    last_gw = prior["GW"].max()
    closing_form = (
        prior[prior["GW"] == last_gw]
        .drop_duplicates(subset="player_code")
        .set_index("player_code")["total_points_avg_last_5"]
    )

    # load_latest_prices() returns FPL's raw numeric `id` as player_id -- that
    # resets every season (see load_historical.py's caveat on `element`), so it
    # can't be used to look up prior_season data directly. Map it to the stable
    # `code` field via the same bootstrap snapshot, same pattern used throughout
    # this project wherever cross-season identity matters.
    live = load_latest_prices()

    with open(_latest_bootstrap_path(), encoding="utf-8") as f:
        raw = json.load(f)
    code_by_id = {p["id"]: p["code"] for p in raw["elements"]}
    live["player_code"] = live["player_id"].map(code_by_id)

    live["predicted_points"] = live["player_code"].map(closing_form).fillna(0).clip(lower=0)

    return live[["player_id", "name", "position", "team", "cost", "predicted_points", "player_code"]]


@st.cache_data
def load_model_metrics() -> dict:
    """Real validation metrics from the last train.py run, read from
    models/metrics.json -- NOT hardcoded numbers. train.py writes this file
    itself each time it runs, so retraining (e.g. once live 2026-27 data joins
    the training set) automatically keeps this tab honest without anyone
    needing to remember to hand-edit numbers here too. Returns None if
    train.py has never been run in this environment."""
    path = os.path.join(PROJECT_DIR, "models", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DASHBOARD_ENTRY_HISTORY_FALLBACK = os.path.join(PROJECT_DIR, "data", "dashboard_entry_history.json")


def _find_entry_history_path(entry_id: int) -> str:
    """Most recent data/raw/{season}/entry/{entry_id}/history.json the collector
    has written -- checks every season folder (not just the latest), since a
    manager's PAST-season totals (what this tab shows) don't change once a
    season ends, so whichever collector run captured them is still valid.

    data/raw/ is entirely gitignored (see .gitignore's note on personal
    manager data), so a fresh deploy (Streamlit Cloud) has nothing here --
    same problem as load_latest_prices()'s bootstrap fallback. Falls back to
    data/dashboard_entry_history.json, a single deliberately-committed,
    non-timestamped copy of this manager's real history -- not a new privacy
    decision, since this exact data is already published on the public
    manager-history GitHub Pages page linked throughout this app; this is
    just giving the dashboard tab the same data that page already has.
    Refreshed by hand/CI, same as the bootstrap fallback -- goes stale
    between refreshes but only for PAST seasons, which don't change once
    finished anyway."""
    pattern = os.path.join(PROJECT_DIR, "data", "raw", "*", "entry", str(entry_id), "history.json")
    paths = sorted(glob.glob(pattern))
    if paths:
        return paths[-1]
    if os.path.exists(_DASHBOARD_ENTRY_HISTORY_FALLBACK):
        return _DASHBOARD_ENTRY_HISTORY_FALLBACK
    return None


@st.cache_data
def load_manager_history(entry_id: int) -> pd.DataFrame:
    """Real season-by-season totals for one manager, read directly from the
    collector's own saved entry/{id}/history.json -- NOT a hardcoded table.
    Returns an empty DataFrame (not an error) if the collector has never
    snapshotted this entry, which is an expected state (FPL_ENTRY_ID is
    optional -- see snapshot.py), not a bug to raise on."""
    path = _find_entry_history_path(entry_id)
    if path is None:
        return pd.DataFrame(columns=["season", "points", "rank", "top_pct"])

    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    past = history.get("past", [])
    df = pd.DataFrame([
        {
            "season": s["season_name"],
            "points": s["total_points"],
            "rank": s["rank"],
            "top_pct": int(s["rank_percentage"]),
        }
        for s in past
    ])
    return df


@st.cache_data
def load_current_season_progress(entry_id: int) -> pd.DataFrame:
    """This manager's GAMEWEEK-BY-GAMEWEEK progress for the CURRENT (in
    progress) season -- distinct from load_manager_history's PAST, season-
    total-only rows, which the live API never updates mid-season for old
    seasons (see fpl_api.get_entry_history's docstring: the public API only
    exposes gw-by-gw detail for the season actually happening right now).
    Returns an empty DataFrame before the first gameweek finishes, which is
    the correct/expected state right now (2026-27 hasn't started), not a bug."""
    path = _find_entry_history_path(entry_id)
    if path is None:
        return pd.DataFrame(columns=["gw", "points", "total_points", "overall_rank", "bank", "value"])

    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    current = history.get("current", [])
    if not current:
        return pd.DataFrame(columns=["gw", "points", "total_points", "overall_rank", "bank", "value"])

    df = pd.DataFrame([
        {
            "gw": g["event"],
            "points": g["points"],
            "total_points": g["total_points"],
            "overall_rank": g["overall_rank"],
            "bank": g["bank"] / 10.0,
            "value": g["value"] / 10.0,
        }
        for g in current
    ])
    return df.sort_values("gw").reset_index(drop=True)


@st.cache_data
def scout_picks_pool(_features_df: pd.DataFrame, prior_season: str = "2025-26") -> pd.DataFrame:
    """This project's own 'Scout Picks' -- a pre-season recommended squad in
    the spirit of FPL's official editorial Scout Picks article, but built from
    real, checkable signals instead of human commentary (which isn't
    structured API data and has no stable weekly URL to fetch -- checked
    directly: article ids are unpredictable and the listing page needs JS
    rendering this project's lightweight collector can't do). Starts from
    preseason_pool()'s predicted_points (live price + prior-season closing
    form) and adds two real, verified boosts on top:

    1. EASY GW1 FIXTURE: a real boost for players whose actual GW1 opponent
       (from the collector's own fixtures.csv, resolved via the matching
       bootstrap snapshot's team-id mapping) is a team newly promoted to the
       Premier League this season -- verified directly by diffing this
       season's team list against last season's (features.parquet), not
       assumed from names: confirmed {Hull City, Ipswich Town, Coventry City}
       are the actual 2026-27 new arrivals, matching what FPL's own Scout
       Picks article names for exactly this reason.
    2. SET-PIECE DUTY: a real boost for players FPL's own bootstrap-static
       flags as a club's primary penalty taker (penalties_order == 1) --
       genuine set-piece priority data, not a guess.

    Returns the same shape as preseason_pool() plus a `scout_reasons` column
    (a short string explaining why a player got a boost, or empty if none
    applied) so the dashboard can show its own reasoning per player, same
    spirit as the real article's player-by-player commentary."""
    pool = preseason_pool(_features_df, prior_season)

    with open(_latest_bootstrap_path(), encoding="utf-8") as f:
        raw = json.load(f)
    team_id_to_name = {t["id"]: t["name"] for t in raw["teams"]}
    penalty_takers = {p["id"] for p in raw["elements"] if p.get("penalties_order") == 1}

    prior_teams = set(_features_df[_features_df["season"] == prior_season]["team"].unique())
    current_teams = set(team_id_to_name.values())
    promoted_teams = current_teams - prior_teams

    fixtures_path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "fixtures.csv")
    gw1_opponent_by_team = {}
    if os.path.exists(fixtures_path):
        fx = pd.read_csv(fixtures_path)
        gw1 = fx[fx["event"] == 1]
        for _, row in gw1.iterrows():
            home, away = team_id_to_name.get(row["team_h"]), team_id_to_name.get(row["team_a"])
            if home and away:
                gw1_opponent_by_team[home] = away
                gw1_opponent_by_team[away] = home

    EASY_FIXTURE_BOOST = 1.5
    PENALTY_TAKER_BOOST = 1.0

    pool = pool.copy()
    pool["scout_reasons"] = ""
    boosted = pool["predicted_points"].copy()

    for idx, row in pool.iterrows():
        reasons = []
        opponent = gw1_opponent_by_team.get(row["team"])
        if opponent and opponent in promoted_teams:
            boosted[idx] += EASY_FIXTURE_BOOST
            reasons.append(f"GW1 vs promoted {opponent}")
        if row["player_id"] in penalty_takers:
            boosted[idx] += PENALTY_TAKER_BOOST
            reasons.append("club's #1 penalty taker")
        pool.at[idx, "scout_reasons"] = ", ".join(reasons)

    pool["predicted_points"] = boosted
    return pool


def _latest_bootstrap_path() -> str:
    """Same fallback logic as optimizer.py's load_latest_prices(): data/raw/ is
    gitignored, so a fresh deploy has no timestamped snapshot -- falls back to
    the deliberately-committed data/dashboard_bootstrap.json."""
    pattern = os.path.join(PROJECT_DIR, "data", "raw", "*", "bootstrap", "bootstrap_*.json")
    paths = sorted(glob.glob(pattern))
    if paths:
        return paths[-1]
    fallback = os.path.join(PROJECT_DIR, "data", "dashboard_bootstrap.json")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"No bootstrap snapshot found, and no fallback at {fallback} -- "
        f"run the collector (src/collector/snapshot.py) at least once first."
    )


@st.cache_data
def _team_name_to_badge_code() -> dict:
    """Team name (e.g. "Arsenal", the string every pool already carries as
    `team`) -> FPL's team `code` (e.g. 3), the id FPL's own real badge CDN
    uses: https://resources.premierleague.com/premierleague/badges/70/t{code}.png
    -- checked directly, returns a real 200 for Arsenal (t3) and Bournemouth
    (t91). Used as a fallback visual for a player with no headshot on FPL's
    CDN (confirmed some current signings genuinely have none yet -- a real
    403 from FPL's own servers, not a bug here), so the placeholder at least
    shows the player's real team instead of a blank box."""
    with open(_latest_bootstrap_path(), encoding="utf-8") as f:
        raw = json.load(f)
    return {t["name"]: t["code"] for t in raw["teams"]}


def _player_card_html(row: pd.Series, badge_label: str = None) -> str:
    """One player's shirt-style card: name, price, predicted points, and an
    optional corner badge (e.g. MVP / Player of the Week) for the single
    highest-predicted_points player in the squad.

    IMPORTANT: this is passed to st.markdown(unsafe_allow_html=True), which
    runs the string through Markdown parsing BEFORE rendering HTML -- Markdown
    treats 4+ leading spaces as a literal code block, so any indentation here
    (however readable in Python) prints as visible text on the page instead of
    rendering as HTML. Every line must start at column 0, no exceptions."""
    badge_html = (
        f'<div title="{badge_label}" style="position: absolute; top: -8px; right: -6px; '
        f'background: #ffb300; color: #1a1a1a; border-radius: 50%; width: 20px; height: 20px; '
        f'font-size: 12px; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">⭐</div>'
    ) if badge_label else ""
    # FPL's own official Team of the Week flag (in_dreamteam) -- distinct from
    # the ⭐ MVP/POTW badge above (this project's OWN top-pick call, which can
    # only ever mark one player), since FPL's real Dream Team has up to 11
    # players and is a fact about that gameweek, not a recommendation. Only
    # present at all for live-collected gameweeks (see gw_pool's docstring) --
    # absent (not False) for historical seasons, so nothing is shown rather
    # than a wrong "not selected" implication for data that doesn't exist.
    dreamteam_html = (
        f'<div title="FPL official Team of the Week pick" style="position: absolute; top: -8px; left: -6px; '
        f'background: #2a9650; color: white; border-radius: 50%; width: 20px; height: 20px; '
        f'font-size: 11px; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">✓</div>'
        if "in_dreamteam" in row.index and pd.notna(row["in_dreamteam"]) and row["in_dreamteam"] else ""
    )
    # points_per_game only exists on Team of the Season pool rows (see
    # season_pool) -- predicted_points there is already the real season/window
    # total (what selection is ranked on), so points_per_game (FPL's own rate
    # metric) is shown underneath as extra context, not a replacement label.
    is_season_pool = "points_per_game" in row.index
    points_label = f'{row["predicted_points"]:.0f} pts' if is_season_pool else f'{row["predicted_points"]:.1f} pts'
    ppg_html = (
        f'<div style="font-size: 9.5px; color: #777; margin-top: 0px;">'
        f'{row["points_per_game"]:.1f} pts/game</div>'
        if is_season_pool and pd.notna(row["points_per_game"]) else ""
    )
    # scout_reasons only exists on Scout Picks pool rows (see
    # scout_picks_pool) -- a real, checkable reason (easy GW1 fixture vs a
    # verified promoted team, or confirmed set-piece duty) this project's own
    # boost was applied, shown as a hover tooltip since card space is tight.
    scout_html = (
        f'<div title="{row["scout_reasons"]}" style="position: absolute; bottom: -6px; right: -6px; '
        f'background: #1a1a1a; color: #ffb300; border-radius: 50%; width: 16px; height: 16px; '
        f'font-size: 10px; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">🔍</div>'
        if "scout_reasons" in row.index and row["scout_reasons"] else ""
    )
    # Real FPL player headshot -- checked directly against the live CDN:
    # https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png
    # returns 200 for a real player code (confirmed with Raya, code 154561),
    # a genuine 220x280 PNG (2x the "110x140" the URL implies) -- checked
    # PIXEL DIMENSIONS directly from the file header, not assumed. A player
    # with no real photo on FPL's CDN gets a 403, not a 404 or placeholder
    # image, confirmed directly against a fabricated player code -- genuinely
    # missing for some current signings (e.g. Adrien Truffert, Enzo Le Fée
    # both 403 at every resolution checked), not something wrong on this
    # project's end.
    #
    # `onerror` (a JS event handler) turned out to NOT fire reliably through
    # Streamlit's sanitized st.markdown(unsafe_allow_html=True) rendering --
    # broken photos were showing the browser's ugly broken-image icon
    # instead of being hidden. Fixed with a CSS-only approach instead: a
    # wrapping div carries a fallback BEHIND the image via CSS
    # background-image (the player's real team badge, also from FPL's own
    # CDN -- checked directly, t{code}.png returns 200 for Arsenal/
    # Bournemouth), and object-fit: contain (not cover, which was cropping
    # heads off at a mismatched box height -- the box was 52px tall against
    # a photo shaped roughly 5:6.4) at a box matched to the real aspect
    # ratio means the whole photo fits inside its box without distortion or
    # crop. A failed <img> renders with no visible content of its own, so
    # the wrapping div's own background-image shows through automatically
    # -- no JS needed, and the fallback is the player's correct real team,
    # not a generic gray box.
    photo_code = row["player_code"] if "player_code" in row.index and pd.notna(row.get("player_code")) else row["player_id"]
    team_code = _team_name_to_badge_code().get(row["team"])
    badge_fallback_css = (
        f'background-image: url(https://resources.premierleague.com/premierleague/badges/70/t{team_code}.png); '
        f'background-size: 44px; background-repeat: no-repeat; background-position: center;'
        if team_code else ""
    )
    img_html = (
        f'<div style="width: 100%; height: 68px; background-color: #d8dde3; {badge_fallback_css} '
        f'border-radius: 6px; margin-bottom: 2px; overflow: hidden;">'
        f'<img src="https://resources.premierleague.com/premierleague/photos/players/110x140/p{photo_code}.png" '
        f'style="width: 100%; height: 100%; object-fit: contain;" loading="lazy" />'
        f'</div>'
    )
    return (
        f'<div style="position: relative; background: rgba(255,255,255,0.94); border-radius: 8px; '
        f'padding: 6px 8px; min-width: 92px; max-width: 118px; text-align: center; '
        f'box-shadow: 0 2px 6px rgba(0,0,0,0.25); font-family: sans-serif;">'
        f'{badge_html}'
        f'{dreamteam_html}'
        f'{scout_html}'
        f'{img_html}'
        f'<div style="font-weight: 600; font-size: 12px; color: #1a1a1a; line-height: 1.2; '
        f'white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row["name"]}</div>'
        f'<div style="font-size: 10.5px; color: #555; margin-top: 2px;">£{row["cost"]:.1f}m</div>'
        f'<div style="font-size: 11px; color: #0a6b2f; font-weight: 700; margin-top: 1px;">'
        f'{points_label}</div>'
        f'{ppg_html}'
        f'</div>'
    )


def render_pitch(squad: pd.DataFrame, top_player_badge: str = None) -> None:
    """A real FPL-style pitch layout: starting XI positioned by row (GK at the
    back, then DEF/MID/FWD moving up the pitch, formation read directly from
    the squad rather than assumed), with the 4 bench players shown below in
    their own strip.

    top_player_badge, if given, is the label ("MVP" for a full-season pool,
    "Player of the Week" for a single-gameweek pool) shown on the single
    highest-predicted_points player in the squad -- the badge's meaning
    depends on what predicted_points means for that pool, so the caller
    (which knows which Squad Builder mode built this squad) decides the label
    rather than this function guessing from the data alone.

    IMPORTANT: every line of HTML built here must start at column 0 -- see
    _player_card_html's docstring for why (Markdown-then-HTML rendering via
    st.markdown(unsafe_allow_html=True) treats indentation as a code block)."""
    starters = squad[squad["in_starting_xi"]]
    bench = squad[~squad["in_starting_xi"]]

    top_player_id = squad.loc[squad["predicted_points"].idxmax(), "player_id"] if top_player_badge and len(squad) else None

    def _card(row):
        badge = top_player_badge if row["player_id"] == top_player_id else None
        return _player_card_html(row, badge)

    rows_html = ""
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = starters[starters["position"] == pos]
        if pos_players.empty:
            continue
        cards = "".join(_card(r) for _, r in pos_players.iterrows())
        rows_html += (
            '<div style="display: flex; justify-content: center; gap: 14px; '
            f'margin: 14px 0; flex-wrap: wrap;">{cards}</div>'
        )

    formation = "-".join(
        str((starters["position"] == pos).sum()) for pos in ["DEF", "MID", "FWD"]
    )

    bench_cards = "".join(_card(r) for _, r in bench.iterrows())

    pitch_html = (
        '<div style="background: linear-gradient(180deg, #1f7a3f 0%, #2a9650 50%, '
        '#1f7a3f 100%); border-radius: 12px; padding: 20px 12px; margin-top: 8px; '
        'border: 2px solid rgba(255,255,255,0.3);">'
        '<div style="text-align: center; color: white; font-size: 12px; opacity: 0.85; '
        'margin-bottom: 8px; font-family: sans-serif; letter-spacing: 0.5px;">'
        f'FORMATION {formation}</div>'
        f'{rows_html}'
        '</div>'
        '<div style="background: #222; border-radius: 10px; padding: 14px 12px; '
        'margin-top: 10px;">'
        '<div style="text-align: center; color: #aaa; font-size: 11px; margin-bottom: 8px; '
        'font-family: sans-serif; letter-spacing: 0.5px;">BENCH</div>'
        '<div style="display: flex; justify-content: center; gap: 14px; flex-wrap: wrap;">'
        f'{bench_cards}'
        '</div>'
        '</div>'
    )
    st.markdown(pitch_html, unsafe_allow_html=True)


st.markdown("""
<div class="app-hero">
<h1>⚽ FPL Analytics</h1>
<p>Expected-points model, squad optimizer, and chip advisor for Fantasy Premier League —
built on the live FPL API, not a third-party scrape. The 2026-27 season hasn't started yet,
so every tool below runs against historical seasons as a working demo; it becomes a live
tool automatically once real gameweeks exist.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚽ FPL Analytics")
    st.caption("Model-driven FPL squad, transfer, and chip planning.")
    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. Live FPL API collector\n"
        "2. Unified historical dataset\n"
        "3. Feature engineering\n"
        "4. Trained LightGBM model\n"
        "5. Squad/transfer optimizer\n"
        "6. Chip-timing advisor"
    )
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/lucifer0096/FPL-Analytics) · "
        "[Manager history page](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)"
    )

tab_overview, tab_squad, tab_transfers, tab_chips, tab_model, tab_history = st.tabs(
    ["📋 Overview", "🧠 Squad Builder", "🔁 Transfers", "🃏 Chip Advisor", "📈 Model Performance", "🏆 Manager History"]
)

# =============================================================================
# OVERVIEW
# =============================================================================
with tab_overview:
    st.header("Project overview")

    col1, col2, col3, col4 = st.columns(4)
    df = load_features()
    overview_metrics = load_model_metrics()
    mae_display = f"{overview_metrics['single_stage']['mae']:.3f}" if overview_metrics else "N/A"
    col1.metric("Seasons tracked", f"{df['season'].nunique()}")
    col2.metric("Player-gameweek rows", f"{len(df):,}")
    col3.metric("Model validation MAE", mae_display)
    col4.metric("Squad rules encoded", "8")

    st.subheader("Pipeline")
    pipeline_steps = [
        "FPL API collector", "Historical dataset", "Feature engineering",
        "LightGBM model", "Squad / transfer optimizer", "Chip advisor",
    ]
    pills = "".join(
        f'<span style="background: linear-gradient(135deg, #2a9650, #5a3cb4); '
        f'color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; '
        f'font-weight: 600; white-space: nowrap;">{step}</span>'
        + ('<span style="color: #888; font-size: 1.1rem;">→</span>' if i < len(pipeline_steps) - 1 else '')
        for i, step in enumerate(pipeline_steps)
    )
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; '
        f'margin: 4px 0 18px 0;">{pills}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "All of FPL's real rules used here (squad composition, budget, transfer costs, "
        "the 50% sell-fee, chip-per-half structure) were verified directly against the "
        "live API's `bootstrap-static` `game_settings` and `chips` fields — see the "
        "project README for the full rules table and how each was confirmed."
    )

    st.subheader("Why historical data, not live predictions, right now")
    st.info(
        "The 2026-27 season's first fixture is 21 August 2026. Every player currently "
        "has zero gameweeks played this season, so there's no rolling form, no fixture "
        "history, and no meaningful model prediction to show yet. The tabs in this app "
        "demonstrate each tool against real historical seasons instead — once the "
        "collector has captured a few 2026-27 gameweeks, these same tools work "
        "unchanged against live data."
    )

# =============================================================================
# SQUAD BUILDER
# =============================================================================
with tab_squad:
    st.header("Squad builder")
    st.caption(
        "Solves for the 15-man squad that maximizes total predicted points under FPL's "
        "real constraints (2 GK / 5 DEF / 5 MID / 3 FWD, £100m budget, max 3 per club)."
    )

    mode = st.radio(
        "Player pool",
        ["Historical gameweek", "Team of the Season", "2026-27 pre-season (live prices)",
         "Scout Picks (2026-27 season opener)", "My squad (enter manually)"],
        key="squad_mode",
        horizontal=True,
    )

    # A squad built in one mode otherwise stays visible after switching to
    # another (e.g. a historical-gameweek squad still shown once you switch to
    # "My squad (enter manually)", even though you haven't picked anything
    # yet) -- session_state persists across reruns by design, but that reads as
    # "already pre-filled" here. Clear it on a genuine mode change.
    if st.session_state.get("_last_squad_mode") != mode:
        st.session_state.pop("built_squad", None)
        st.session_state.pop("built_squad_season_gw", None)
        st.session_state["_last_squad_mode"] = mode

    df = load_features()

    if mode == "Historical gameweek":
        col_a, col_b = st.columns(2)
        season = col_a.selectbox("Season", SEASON_ORDER, index=SEASON_ORDER.index("2025-26"), key="squad_season")
        max_gw = int(df[df["season"] == season]["GW"].max())
        gw = col_b.slider("Gameweek", 1, max_gw, min(20, max_gw), key="squad_gw")
        pool = gw_pool(df, season, gw)
        pool_source = (season, gw)
        model_note = "the trained xP model (fixture difficulty, form, minutes)" if _get_xp_model() is not None else "each player's rolling-5 average as a fallback (no trained model file found)"
        st.caption(f"Player pool: {len(pool)} players, predicted-points from {model_note} for this specific past gameweek.")

        if pool is not None and st.button("Build optimal squad", key="build_squad_btn"):
            with st.spinner("Solving..."):
                squad = optimize_squad(pool)
            st.session_state["built_squad"] = squad
            st.session_state["built_squad_season_gw"] = pool_source

    elif mode == "Team of the Season":
        season = st.selectbox("Season", SEASON_ORDER, index=SEASON_ORDER.index("2025-26"), key="tots_season")
        max_gw = int(df[df["season"] == season]["GW"].max())
        gw_start, gw_end = st.slider(
            "Gameweek range (full season by default — narrow it to build a Team of the Week instead)",
            1, max_gw, (1, max_gw), key="tots_gw_range",
        )
        is_full_season = (gw_start, gw_end) == (1, max_gw)
        pool = season_pool(df, season, gw_start, gw_end)
        window_desc = f"the full {season} season" if is_full_season else f"GW{gw_start}–GW{gw_end} of {season}"
        st.caption(f"Player pool: {len(pool)} players who appeared at least once across {window_desc}. A real LOOK BACK, not a prediction — ranked by total points scored, no budget cap.")
        with st.expander("Why this ranking, and why no budget cap?"):
            st.markdown(
                f"This is a LOOK BACK at who actually produced the most REAL points in this "
                f"window, not a prediction — these gameweeks are already complete, so there's "
                f"nothing to predict.\n\n"
                f"Ranked by **total points scored**, not a per-game rate — a great rate over a "
                f"handful of games can't outrank someone who played most of the window and "
                f"produced far more for a real squad. Points-per-game (FPL's own metric: total "
                f"points ÷ appearances, shown once a player has a few games to make it "
                f"trustworthy) is shown on each card as context, not as what drives selection.\n\n"
                f"**No budget cap** — this is the best XI the window actually produced, not a "
                f"squad you could have afforded on day one."
            )

        build_label = "Build team of the season" if is_full_season else f"Build team of GW{gw_start}–{gw_end}"
        if pool is not None and st.button(build_label, key="build_tots_btn"):
            with st.spinner("Solving..."):
                # No budget constraint: Team of the Season is "who were the best
                # performers", not a squad buildable on a real £100m budget --
                # a giant budget cap effectively disables the constraint while
                # reusing the same optimizer (still bound by position quotas
                # and max-3-per-club).
                squad = optimize_squad(pool, budget=10_000.0)
            st.session_state["built_squad"] = squad
            st.session_state["built_squad_season_gw"] = "completed_season"  # distinct from pre-season/manual's None -- there's no live pool to fall back to for an already-finished season
            st.session_state["built_squad_window_label"] = "this season" if is_full_season else f"GW{gw_start}–{gw_end}"

    elif mode == "2026-27 pre-season (live prices)":
        try:
            pool = preseason_pool(df)
            pool_source = None  # no "next gameweek" exists yet -- Transfers/Chips need real gameweek data
            st.caption(
                f"Player pool: {len(pool)} players, at their LIVE current 2026-27 price, "
                f"using each player's rolling-5 average at the END of 2025-26 as a predicted-points "
                f"stand-in (their most recent known real form). Players with no 2025-26 Premier League "
                f"record (new signings, promoted-team players) get 0 rather than a guess."
            )
        except FileNotFoundError as e:
            st.error(str(e))
            pool = None

        if pool is not None and st.button("Build optimal squad", key="build_squad_btn"):
            with st.spinner("Solving..."):
                squad = optimize_squad(pool)
            st.session_state["built_squad"] = squad
            st.session_state["built_squad_season_gw"] = pool_source

    elif mode == "Scout Picks (2026-27 season opener)":
        try:
            pool = scout_picks_pool(df)
            pool_source = None
            n_reasons = (pool["scout_reasons"] != "").sum()
            st.caption(f"This project's own take on FPL's editorial 'Scout Picks' — {n_reasons} of {len(pool)} players got a real, shown boost reason (🔍 on their pitch card).")
            with st.expander("What is this, and why not FPL's actual Scout Picks?"):
                st.markdown(
                    "Not a scrape of FPL's real article — checked directly: it's not structured "
                    "API data, and has no stable weekly URL to fetch, only unpredictable per-"
                    "article ids.\n\n"
                    "Built from real signals instead, on top of the same pre-season pool as "
                    "above:\n"
                    "- A boost for a genuinely easy **GW1 fixture** against one of this season's "
                    "actual promoted teams (Hull City, Ipswich Town, Coventry City — verified by "
                    "diffing this season's team list against last season's).\n"
                    "- A boost for confirmed **set-piece duty**, via FPL's own `penalties_order` "
                    "field."
                )
        except FileNotFoundError as e:
            st.error(str(e))
            pool = None

        if pool is not None and st.button("Build Scout Picks squad", key="build_scout_btn"):
            with st.spinner("Solving..."):
                squad = optimize_squad(pool)
            st.session_state["built_squad"] = squad
            st.session_state["built_squad_season_gw"] = pool_source

    elif mode == "My squad (enter manually)":
        st.caption(
            "Pick your actual 15-man squad from the live 2026-27 player pool and see it laid out "
            "on the pitch, with the optimal starting XI worked out automatically. Uses each player's "
            "live current price and their rolling-5 average at the end of 2025-26 as a predicted-points "
            "estimate — same basis as pre-season mode, but for the 15 players YOU picked, not the "
            "optimizer's choice of who to buy."
        )
        try:
            manual_pool = preseason_pool(df)
        except FileNotFoundError as e:
            st.error(str(e))
            manual_pool = None

        if manual_pool is not None:
            manual_pool = manual_pool.copy()
            manual_pool["label"] = manual_pool.apply(
                lambda r: f"{r['name']} ({r['team']}, {r['position']}, £{r['cost']:.1f}m)", axis=1
            )
            label_to_id = dict(zip(manual_pool["label"], manual_pool["player_id"]))
            id_to_label = {v: k for k, v in label_to_id.items()}

            if "manual_squad_players" not in st.session_state:
                saved_ids = _load_saved_manual_squad_ids()
                st.session_state["manual_squad_players"] = [
                    id_to_label[pid] for pid in saved_ids if pid in id_to_label
                ]

            chosen_labels = st.multiselect(
                "Your 15 players (search by name)",
                options=sorted(manual_pool["label"]),
                key="manual_squad_players",
            )

            if chosen_labels:
                chosen_ids = [label_to_id[l] for l in chosen_labels]
                chosen_df = manual_pool[manual_pool["player_id"].isin(chosen_ids)]

                pos_counts = chosen_df["position"].value_counts().to_dict()
                total_cost = chosen_df["cost"].sum()
                team_counts = chosen_df["team"].value_counts()

                col1, col2, col3 = st.columns(3)
                col1.metric("Players picked", f"{len(chosen_df)} / 15")
                col2.metric("Total cost", f"£{total_cost:.1f}m / £{DEFAULT_BUDGET}m")
                col3.metric("Max from one club", f"{team_counts.max() if len(team_counts) else 0} / 3")

                issues = []
                for pos, quota in POSITION_REQUIREMENTS.items():
                    got = pos_counts.get(pos, 0)
                    if len(chosen_df) == 15 and got != quota:
                        issues.append(f"{pos}: need {quota}, have {got}")
                if len(chosen_df) == 15 and total_cost > DEFAULT_BUDGET + 1e-6:
                    issues.append(f"Over budget by £{total_cost - DEFAULT_BUDGET:.1f}m")
                if team_counts.max() > 3 if len(team_counts) else False:
                    issues.append(f"Too many players from one club (max 3)")

                if issues:
                    st.warning("Squad issues: " + "; ".join(issues))

                if len(chosen_df) == 15 and not issues:
                    st.caption(
                        "Starting XI is picked automatically to maximize predicted points, "
                        "same solver as Squad Builder's other modes — not something you pick by hand."
                    )
                    if st.button("Show my squad", key="show_manual_squad_btn"):
                        manual_squad = chosen_df.copy()
                        starter_ids = select_starting_xi(
                            manual_squad, "player_id", "position", "predicted_points"
                        )
                        manual_squad["in_starting_xi"] = manual_squad["player_id"].isin(starter_ids)
                        st.session_state["built_squad"] = manual_squad
                        st.session_state["built_squad_season_gw"] = None
                        _save_manual_squad_ids(chosen_df["player_id"].tolist())

    if "built_squad" in st.session_state:
        squad = st.session_state["built_squad"]
        squad_source = st.session_state["built_squad_season_gw"]

        if squad_source == "completed_season":
            # predicted_points here is each player's real total points across
            # the window -- no £100m cap to report against either (see the
            # "no budget cap" note above).
            window_label = st.session_state.get("built_squad_window_label", "this window")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Squad cost", f"£{squad['cost'].sum():.1f}m")
            sc2.metric("Points scored", f"{squad['predicted_points'].sum():.0f}")
            sc3.metric("Window", window_label)
            badge_label = f"MVP — most points scored in {window_label}"
        else:
            sc1, sc2 = st.columns(2)
            sc1.metric("Squad cost", f"£{squad['cost'].sum():.1f}m", f"of £{DEFAULT_BUDGET}m budget")
            sc2.metric("Predicted points", f"{squad['predicted_points'].sum():.1f}")
            badge_label = "Player of the Week — top predicted scorer this gameweek" if isinstance(squad_source, tuple) else None
        render_pitch(squad, top_player_badge=badge_label)

        with st.expander("Full squad table"):
            display = squad.copy()
            display["Role"] = display["in_starting_xi"].map({True: "Starting XI", False: "Bench"})
            st.dataframe(
                display[["name", "position", "team", "cost", "predicted_points", "Role"]]
                .sort_values(["Role", "position", "predicted_points"], ascending=[True, True, False])
                .rename(columns={"name": "Player", "position": "Pos", "team": "Team", "cost": "Cost (£m)", "predicted_points": "Pred. Pts"}),
                use_container_width=True,
                hide_index=True,
            )

# =============================================================================
# TRANSFERS
# =============================================================================
with tab_transfers:
    st.header("Transfer optimizer")
    st.caption(
        "Given a squad from one gameweek, suggests the transfer(s) that maximize "
        "predicted points gained minus the -4pt cost of any hit, into the following gameweek's pool."
    )

    if "built_squad" not in st.session_state:
        st.warning("Build a squad in the **Squad Builder** tab first.")
    else:
        df = load_features()
        current_squad = st.session_state["built_squad"]

        if st.session_state["built_squad_season_gw"] == "completed_season":
            st.info(
                "This squad is a **Team of the Season** look-back at an already-completed "
                "season — there's no 'next gameweek' to transfer into. Build a squad from "
                "the Historical gameweek, pre-season, or manual-entry modes in the Squad "
                "Builder tab to try this tool."
            )
            next_pool = None
        elif st.session_state["built_squad_season_gw"] is None:
            # Pre-season or manually-entered squad: no "next historical gameweek"
            # exists, so check against the same LIVE pool the squad was built
            # from instead (re-fetched fresh, in case prices moved).
            try:
                next_pool = preseason_pool(df)
                pool_label = "the live 2026-27 pool (re-fetched, in case prices moved since you built this squad)"
            except FileNotFoundError as e:
                st.error(str(e))
                next_pool = None
        else:
            built_season, built_gw = st.session_state["built_squad_season_gw"]
            next_gw = built_gw + 1
            max_gw_for_season = int(df[df["season"] == built_season]["GW"].max())
            if next_gw > max_gw_for_season:
                st.warning(f"GW{built_gw} is the last available gameweek in {built_season} — pick an earlier gameweek in the Squad Builder tab to leave room for a following gameweek.")
                next_pool = None
            else:
                next_pool = gw_pool(df, built_season, next_gw)
                pool_label = f"GW{next_gw} of {built_season}"

        if next_pool is not None:
            unlimited = st.checkbox(
                "Playing Wildcard or Free Hit this gameweek (or this is GW1) — unlimited free transfers",
                key="unlimited_transfers_checkbox",
                help="Verified against the live API's game_settings: GW1 allows a transfers_cap of 20 "
                     "(effectively unlimited) instead of the normal 1-5 banked limit, and every transfer "
                     "made while Wildcard or Free Hit is active is free by chip definition, with no cap "
                     "and no -4pt hit ever applying. Check this instead of setting free transfers below.",
            )
            if not unlimited:
                free_transfers = st.slider("Free transfers available", 1, 5, 1, key="ft_slider")

            if unlimited:
                st.caption(f"Checking against {pool_label}. Unlimited transfers this gameweek — no hit cost, no minimum-gain bar.")
            else:
                st.caption(f"Checking against {pool_label}. Only recommends a transfer that clears a real minimum gain on its own — never forces unused free transfers into play.")
                with st.expander("How is a transfer judged \"worth it\"?"):
                    st.markdown(
                        "Each transfer is judged individually, not as a batch average — a strong "
                        "1st transfer can't quietly subsidize a weak 5th one. A hit is only "
                        "suggested if the gain clearly outweighs its -4pt cost, not just barely "
                        "breaks even. Having more free transfers banked never forces more of them "
                        "to be used; holding is the answer whenever nothing clears the bar."
                    )

            if st.button("Find best transfer(s)", key="find_transfers_btn"):
                common_ids = set(current_squad["player_id"]) & set(next_pool["player_id"])
                squad_ids = [pid for pid in current_squad["player_id"] if pid in common_ids]
                dropped = len(current_squad) - len(squad_ids)

                if dropped:
                    st.caption(f"Note: {dropped} squad player(s) not present in this pool, excluded from this check.")

                if len(squad_ids) == sum(POSITION_REQUIREMENTS.values()):
                    with st.spinner("Solving..."):
                        result = optimize_transfers(
                            current_squad_ids=squad_ids,
                            players=next_pool,
                            free_transfers=1 if unlimited else free_transfers,
                            unlimited_transfers=unlimited,
                        )

                    if result["transfers_in"]:
                        out_names = next_pool[next_pool["player_id"].isin(result["transfers_out"])]["name"].tolist()
                        in_names = next_pool[next_pool["player_id"].isin(result["transfers_in"])]["name"].tolist()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Transfers suggested", len(result["transfers_in"]))
                        col2.metric("Hit cost", f"-{result['hit_cost']} pts")
                        col3.metric("Net points gain", f"{result['net_points_gain']:+.1f}")

                        out_col, in_col = st.columns(2)
                        with out_col:
                            st.markdown(
                                '<div style="background: rgba(200,50,50,0.12); border-left: 4px solid #c83232; '
                                'border-radius: 6px; padding: 10px 14px;"><b style="color:#e05555;">OUT</b><br>'
                                + "<br>".join(out_names) + "</div>",
                                unsafe_allow_html=True,
                            )
                        with in_col:
                            st.markdown(
                                '<div style="background: rgba(42,150,80,0.12); border-left: 4px solid #2a9650; '
                                'border-radius: 6px; padding: 10px 14px;"><b style="color:#3fb96a;">IN</b><br>'
                                + "<br>".join(in_names) + "</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No transfer improves on the current squad enough to be worth it — holding is optimal here.")
                else:
                    st.error("Too many squad players missing from this pool to run this check.")

# =============================================================================
# CHIP ADVISOR
# =============================================================================
with tab_chips:
    st.header("Chip-timing advisor")
    st.caption(
        "Ranks upcoming gameweeks for Bench Boost, Triple Captain, and Free Hit, "
        "given a built squad. Each FPL chip is usable once per season half — see "
        "the README for how that was verified against the live API."
    )

    if "built_squad" not in st.session_state:
        st.warning("Build a squad in the **Squad Builder** tab first.")
    elif st.session_state["built_squad_season_gw"] == "completed_season":
        st.info(
            "This squad is a **Team of the Season** look-back at an already-completed "
            "season — there are no upcoming gameweeks left to project chip timing "
            "against. Build a squad from a historical gameweek in the Squad Builder "
            "tab to try this tool."
        )
    elif st.session_state["built_squad_season_gw"] is None:
        st.info(
            "Chip timing needs a run of several **upcoming** gameweeks to project against, "
            "which doesn't exist yet for a live/pre-season squad — only once the collector "
            "has captured real 2026-27 gameweek results. Unlike the Transfers tab (which can "
            "check a live squad against the current live pool), this tool specifically needs "
            "a multi-gameweek future window, so it only works for a squad built from a "
            "historical gameweek in the Squad Builder tab for now."
        )
    else:
        built_season, built_gw = st.session_state["built_squad_season_gw"]
        df = load_features()
        max_gw_for_season = int(df[df["season"] == built_season]["GW"].max())
        window_end = min(built_gw + 4, max_gw_for_season)

        if window_end <= built_gw:
            st.warning(f"Not enough gameweeks left in {built_season} after GW{built_gw} to project a chip window — pick an earlier gameweek in the Squad Builder tab.")
        else:
            gw_range = range(built_gw, window_end + 1)
            st.caption(f"Projecting GW{built_gw}–GW{window_end}.")

            if st.button("Analyze chip timing", key="chip_btn"):
                squad = st.session_state["built_squad"]
                with st.spinner("Projecting..."):
                    future_points_by_gw = {}
                    optimal_points_by_gw = {}
                    for g in gw_range:
                        pool = gw_pool(df, built_season, g)
                        future_points_by_gw[g] = dict(zip(pool["player_id"], pool["predicted_points"]))
                        optimal_points_by_gw[g] = optimize_squad(pool)["predicted_points"].sum()

                def _chip_card_html(icon: str, title: str, suggestions: list) -> str:
                    rows = ""
                    for i, s in enumerate(suggestions[:3]):
                        is_best = i == 0
                        rows += (
                            f'<div style="padding: 8px 10px; margin-bottom: 6px; border-radius: 6px; '
                            f'background: {"rgba(42,150,80,0.15)" if is_best else "rgba(255,255,255,0.04)"}; '
                            f'border-left: 3px solid {"#2a9650" if is_best else "transparent"};">'
                            f'<b>GW{s.gameweek}</b>{" ⭐" if is_best else ""}<br>'
                            f'<span style="font-size: 0.85rem; opacity: 0.85;">{s.detail}</span></div>'
                        )
                    return (
                        f'<div style="background: rgba(255,255,255,0.03); border-radius: 10px; '
                        f'padding: 14px;"><h4 style="margin-top:0;">{icon} {title}</h4>{rows}</div>'
                    )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        _chip_card_html("🪑", "Bench Boost", suggest_bench_boost(squad, future_points_by_gw, gw_range)),
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        _chip_card_html("👑", "Triple Captain", suggest_triple_captain(squad, future_points_by_gw, gw_range)),
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        _chip_card_html("🔄", "Free Hit", suggest_free_hit_or_wildcard(squad, future_points_by_gw, optimal_points_by_gw, gw_range, chip="free_hit")),
                        unsafe_allow_html=True,
                    )

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
with tab_model:
    st.header("Model performance")

    metrics = load_model_metrics()

    if metrics is None:
        st.info(
            "No models/metrics.json found — run `python src/model/train.py` to train the "
            "model and generate real validation metrics. (This section used to show "
            "hardcoded numbers from a past run; now it reads train.py's own output "
            "directly, so it can't silently go stale after a retrain.)"
        )
    else:
        st.caption(
            f"Validated on {metrics['validation_season']} (chronological split — trained "
            f"only on earlier seasons). {metrics['final_holdout_season']} is held out "
            f"entirely and untouched. Read live from models/metrics.json, written by "
            f"train.py's last run — not hardcoded."
        )

        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("Model MAE", f"{metrics['single_stage']['mae']:.3f}")
        mm2.metric("Naive baseline MAE", f"{metrics['naive_baseline']['mae']:.3f}")
        if "fpl_xp_baseline" in metrics:
            improvement = metrics["naive_baseline"]["mae"] - metrics["single_stage"]["mae"]
            mm3.metric("Beats naive baseline by", f"{improvement:.3f} MAE")

        rows = [
            {"Model": "Single-stage", "MAE": metrics["single_stage"]["mae"], "RMSE": metrics["single_stage"]["rmse"]},
            {"Model": "Two-stage", "MAE": metrics["two_stage"]["mae"], "RMSE": metrics["two_stage"]["rmse"]},
            {"Model": "Naive baseline (rolling-5 avg)", "MAE": metrics["naive_baseline"]["mae"], "RMSE": metrics["naive_baseline"]["rmse"]},
        ]
        if "fpl_xp_baseline" in metrics:
            rows.append({"Model": "FPL's own xP*", "MAE": metrics["fpl_xp_baseline"]["mae"], "RMSE": metrics["fpl_xp_baseline"]["rmse"]})
        perf = pd.DataFrame(rows)
        perf["MAE"] = perf["MAE"].round(3)
        perf["RMSE"] = perf["RMSE"].round(3)
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.caption(
            "*FPL's own xP carries a caveat from the data source's maintainer: it may contain "
            "post-match information for some gameweeks (scraper runs after each gameweek ends, "
            "FPL's update cadence for the underlying field is undocumented). Treated as an "
            "informative but not fully leak-free comparison — see the README's Model Training section."
        )

        if "single_stage_played_only" in metrics:
            st.subheader("Where the model does well vs. FPL's xP")
            played_rows = [
                {"Model": "Single-stage (played rows only)", "MAE": metrics["single_stage_played_only"]["mae"]},
                {"Model": "Naive baseline (played only)", "MAE": metrics["naive_baseline_played_only"]["mae"]},
            ]
            if "fpl_xp_baseline_played_only" in metrics:
                played_rows.append({"Model": "FPL's own xP (played only)", "MAE": metrics["fpl_xp_baseline_played_only"]["mae"]})
            played_perf = pd.DataFrame(played_rows)
            played_perf["MAE"] = played_perf["MAE"].round(3)
            st.dataframe(played_perf, use_container_width=True, hide_index=True)
            st.markdown(
                "Restricting to rows where the player actually played, the model closes most of "
                "the gap to FPL's xP. Most of the remaining full-dataset gap is concentrated in "
                "non-playing rows — FPL's xP likely has access to real injury/team-news signals "
                "this project's historical-stats-only feature set can't replicate."
            )

# =============================================================================
# MANAGER HISTORY
# =============================================================================
with tab_history:
    st.header("Manager history")
    st.caption(f"Season-by-season points and overall rank, entry {MANAGER_ENTRY_ID}.")

    st.subheader("2026-27 live progress")
    current_progress = load_current_season_progress(MANAGER_ENTRY_ID)
    if current_progress.empty:
        st.info(
            "No 2026-27 gameweeks finished yet (first fixture 21 Aug 2026) — this section "
            "fills in automatically, gameweek by gameweek, once the collector captures real "
            "results. Unlike the past-seasons table below (which the live API only ever gives "
            "as season totals, never gameweek detail, for an already-finished season — see "
            "load_current_season_progress), this IS gameweek-by-gameweek, live, for the "
            "season actually in progress."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Overall rank by gameweek (lower is better)")
            st.line_chart(current_progress.set_index("gw")["overall_rank"])
        with col2:
            st.caption("Cumulative points by gameweek")
            st.line_chart(current_progress.set_index("gw")["total_points"])
        latest = current_progress.iloc[-1]
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total points", f"{latest['total_points']:.0f}")
        mcol2.metric("Overall rank", f"{latest['overall_rank']:,.0f}")
        mcol3.metric("Bank", f"£{latest['bank']:.1f}m")
        st.dataframe(
            current_progress.rename(columns={
                "gw": "GW", "points": "GW points", "total_points": "Total points",
                "overall_rank": "Overall rank", "bank": "Bank (£m)", "value": "Squad value (£m)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Past seasons")
    manager_data = load_manager_history(MANAGER_ENTRY_ID)

    if manager_data.empty:
        st.info(
            f"No collected history found for entry {MANAGER_ENTRY_ID} — the collector "
            f"snapshots this automatically when `FPL_ENTRY_ID` is set (see README's "
            f"'Running the Collector'), but hasn't been run with it yet in this "
            f"environment. Run `python src/collector/snapshot.py` with `FPL_ENTRY_ID` "
            f"set to populate this tab."
        )
    else:
        best_points_row = manager_data.loc[manager_data["points"].idxmax()]
        best_rank_row = manager_data.loc[manager_data["rank"].idxmin()]
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Seasons played", len(manager_data))
        scol2.metric("Best points season", f"{best_points_row['points']:,} pts", best_points_row["season"])
        scol3.metric("Best overall rank", f"{best_rank_row['rank']:,}", best_rank_row["season"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Points by season")
            st.bar_chart(manager_data.set_index("season")["points"])
        with col2:
            st.subheader("Overall rank by season (lower is better)")
            st.line_chart(manager_data.set_index("season")["rank"])

        st.dataframe(
            manager_data.rename(columns={"season": "Season", "points": "Points", "rank": "Rank", "top_pct": "Top %"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Read live from this project's own collector snapshot "
            f"(`data/raw/*/entry/{MANAGER_ENTRY_ID}/history.json`), not a hardcoded table — "
            "refreshes automatically whenever the collector re-runs. The most recent "
            "season's figures may be provisional if pulled before that season's final "
            "gameweek was confirmed finished. Full interactive version: "
            "[my-fpl-history.html](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html) "
            "(a separate, statically-hosted page — still carries its own hardcoded copy of this "
            "same data, since GitHub Pages can't run this project's Python collector)."
        )
