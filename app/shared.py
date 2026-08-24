"""Shared data-loading, pool-building, and rendering helpers for every page
of the FPL Analytics dashboard (app.py = Home/live page, pages/ = the
Historical & Model page). Split out so both pages import the SAME functions
rather than each carrying its own copy -- a fix applied in one place (a new
fallback path, a rendering bug) automatically covers every page.

No st.set_page_config, CSS injection, or top-level st.markdown/st.tabs calls
belong here -- those are page-specific (set_page_config in particular can
only be called once per page, and must be the first Streamlit call on that
page)."""

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
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "collector"))

from optimizer import optimize_squad, optimize_transfers, load_latest_prices, select_starting_xi, POSITION_REQUIREMENTS, DEFAULT_BUDGET, MAX_FREE_TRANSFERS_BANKED
from chips import suggest_bench_boost, suggest_triple_captain, suggest_free_hit_or_wildcard
from predict import load_model, predict_points
from train import FEATURE_COLUMNS
import fpl_api

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
def season_insights(df: pd.DataFrame, season: str) -> dict:
    """Real, verifiable analytics for one completed (or completing) season --
    not a squad-building tool, just "what actually happened." Distinct from
    season_pool(), which exists to feed the optimizer a ranked player pool;
    this returns several small, human-readable tables for direct display.

    Every number here is a real historical fact (sums/means of actual
    per-gameweek total_points, minutes, value), not a prediction -- no model,
    no rolling window, since the season in question has (mostly) already
    happened.

    Returns a dict with:
      - top_scorers: top 10 by total points (any minutes played)
      - best_value: top 10 by points-per-£m spent (>=450 minutes, so a cheap
        bench player with one big haul doesn't dominate on a tiny sample --
        same floor reasoning as season_pool's min_games_for_window)
      - position_leaders: {position: top scorer in that position}
      - biggest_price_risers: top 10 by (final gameweek value - first
        gameweek value) that season -- a real signal of who the market
        judged to be performing well as the season went on."""
    sub = df[df["season"] == season].copy()
    last_gw = sub["GW"].max()

    totals = sub.groupby("player_code").agg(
        name=("name", "last"), position=("position", "last"), team=("team", "last"),
        total_points=("total_points", "sum"), minutes=("minutes", "sum"),
    )
    latest_value = (
        sub[sub["GW"] == last_gw].drop_duplicates("player_code")
        .set_index("player_code")["value"]
    )
    # A player's TRUE season-start price is their value at their own FIRST
    # gameweek that season, not literally GW1 -- checked directly: 151
    # players in 2025-26 have no GW1 row at all (mid-season signings,
    # promoted-team players not yet in the dataset that early, etc.), so
    # using a fixed GW1 lookup would leave them with a NaN/undefined "start
    # price" rather than their own real starting point. sort_values +
    # drop_duplicates(keep="first") picks each player's own earliest row.
    first_value = (
        sub.sort_values("GW").drop_duplicates("player_code", keep="first")
        .set_index("player_code")["value"]
    )
    totals["cost"] = latest_value / 10.0
    totals["start_cost"] = first_value / 10.0
    totals["price_rise"] = totals["cost"] - totals["start_cost"]

    top_scorers = totals.sort_values("total_points", ascending=False).head(10)

    MIN_MINUTES_FOR_VALUE = 450  # ~5 full matches -- same floor reasoning as MIN_GAMES_FOR_SEASON_RATE, a cheap player with one big haul in limited minutes shouldn't dominate a per-£m ranking
    valued = totals[totals["minutes"] >= MIN_MINUTES_FOR_VALUE].copy()
    valued["pts_per_million"] = valued["total_points"] / valued["cost"]
    best_value = valued.sort_values("pts_per_million", ascending=False).head(10)

    position_leaders = {
        pos: totals[totals["position"] == pos].sort_values("total_points", ascending=False).head(3)
        for pos in ["GK", "DEF", "MID", "FWD"]
    }

    biggest_risers = totals[totals["minutes"] >= MIN_MINUTES_FOR_VALUE].sort_values("price_rise", ascending=False).head(10)

    return {
        "top_scorers": top_scorers,
        "best_value": best_value,
        "position_leaders": position_leaders,
        "biggest_price_risers": biggest_risers,
    }


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

    raw = _load_bootstrap()
    code_by_id = {p["id"]: p["code"] for p in raw["elements"]}
    live["player_code"] = live["player_id"].map(code_by_id)

    live["predicted_points"] = live["player_code"].map(closing_form).fillna(0).clip(lower=0)

    # Real, current availability -- FPL's own status/news/chance-of-playing
    # fields (verified live: e.g. Saliba genuinely flagged 'i' with a real
    # "Back injury - Unknown return date" news string right now). Used to
    # gate transfer suggestions on actual injury/suspension/doubt, not just a
    # predicted-points gap, which stale pre-season form can't reliably
    # signal on its own -- see optimize_transfers' usage of this column.
    status_by_id = {p["id"]: p.get("status") for p in raw["elements"]}
    news_by_id = {p["id"]: p.get("news") for p in raw["elements"]}
    chance_by_id = {p["id"]: p.get("chance_of_playing_next_round") for p in raw["elements"]}
    live["status"] = live["player_id"].map(status_by_id)
    live["news"] = live["player_id"].map(news_by_id)
    live["chance_of_playing_next_round"] = live["player_id"].map(chance_by_id)

    return live[[
        "player_id", "name", "position", "team", "cost", "predicted_points", "player_code",
        "status", "news", "chance_of_playing_next_round",
    ]]


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
    Refreshed automatically by the scheduled collector workflow (see
    .github/workflows/weekly-collector.yml) -- never more than a day stale
    while that workflow keeps running."""
    pattern = os.path.join(PROJECT_DIR, "data", "raw", "*", "entry", str(entry_id), "history.json")
    paths = sorted(glob.glob(pattern))
    if paths:
        return paths[-1]
    if os.path.exists(_DASHBOARD_ENTRY_HISTORY_FALLBACK):
        return _DASHBOARD_ENTRY_HISTORY_FALLBACK
    return None


@st.cache_data(ttl=60)
def _load_entry_history(entry_id: int) -> dict:
    """Live-first replacement for reading _find_entry_history_path() as a
    file: calls the real FPL API directly (fpl_api.get_entry_history(), the
    same endpoint the collector uses) so the CURRENT season's gameweek-by-
    gameweek progress (points, rank, bench points, etc.) is genuinely live
    (60s ttl) rather than a once-a-day batch snapshot -- this is what makes
    My Squad's "2026-27 progress" section update live during/after a match
    instead of waiting on the next scheduled collector run.

    Falls back to _find_entry_history_path()'s file (local data/raw/
    snapshot, then the committed data/dashboard_entry_history.json) if the
    live call fails. Returns an empty dict (not an exception) if neither is
    available -- callers already treat that as "nothing collected yet.\""""
    try:
        return fpl_api.get_entry_history(entry_id)
    except Exception:
        pass
    path = _find_entry_history_path(entry_id)
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_DASHBOARD_ENTRY_INFO_FALLBACK = os.path.join(PROJECT_DIR, "data", "dashboard_entry_info.json")


@st.cache_data(ttl=60)
def load_manager_name(entry_id: int) -> str:
    """This manager's real name (player_first_name + player_last_name).
    Tries the real FPL API directly first (fpl_api.get_entry()) so this
    stays live rather than a batch snapshot -- falls back to the bare
    numeric entry_id if unavailable, rather than failing the whole tab,
    since a name is a display nicety, not something the rest of this tab
    depends on. Falls back, in order, to the collector's saved
    entry/{id}/info.json, then the committed data/dashboard_entry_info.json
    if the live call fails."""
    try:
        info = fpl_api.get_entry(entry_id)
        first, last = info.get("player_first_name"), info.get("player_last_name")
        if first and last:
            return f"{first} {last}"
    except Exception:
        pass
    pattern = os.path.join(PROJECT_DIR, "data", "raw", "*", "entry", str(entry_id), "info.json")
    paths = sorted(glob.glob(pattern))
    path = paths[-1] if paths else (
        _DASHBOARD_ENTRY_INFO_FALLBACK if os.path.exists(_DASHBOARD_ENTRY_INFO_FALLBACK) else None
    )
    if path is None:
        return str(entry_id)
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    first, last = info.get("player_first_name"), info.get("player_last_name")
    if first and last:
        return f"{first} {last}"
    return str(entry_id)


@st.cache_data
def load_manager_history(entry_id: int) -> pd.DataFrame:
    """Real season-by-season totals for one manager -- NOT a hardcoded
    table. Reads via _load_entry_history() (live API first, file fallback).
    Returns an empty DataFrame (not an error) if nothing's available for
    this entry, which is an expected state (FPL_ENTRY_ID is optional -- see
    snapshot.py), not a bug to raise on."""
    history = _load_entry_history(entry_id)
    if not history:
        return pd.DataFrame(columns=["season", "points", "rank", "top_pct"])

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


@st.cache_data(ttl=60)
def load_current_season_progress(entry_id: int) -> pd.DataFrame:
    """This manager's GAMEWEEK-BY-GAMEWEEK progress for the CURRENT (in
    progress) season -- distinct from load_manager_history's PAST, season-
    total-only rows, which the live API never updates mid-season for old
    seasons (see fpl_api.get_entry_history's docstring: the public API only
    exposes gw-by-gw detail for the season actually happening right now).
    Reads via _load_entry_history() (live API first, 60s ttl, file fallback)
    so this genuinely live-updates during/after a match rather than waiting
    on the next scheduled collector run. Returns an empty DataFrame before
    the collector/live API has any data for a gameweek (correct/expected
    before the season's first deadline has passed, not a bug after that)."""
    empty_cols = ["gw", "points", "total_points", "overall_rank", "bank", "value", "event_transfers", "event_transfers_cost", "points_on_bench", "overall_rank_percentage", "average_entry_score"]
    history = _load_entry_history(entry_id)
    if not history:
        return pd.DataFrame(columns=empty_cols)

    current = history.get("current", [])
    if not current:
        return pd.DataFrame(columns=empty_cols)

    # FPL's own real per-gameweek average score across ALL managers --
    # verified directly against bootstrap-static's events -- gives real
    # context for whether a gameweek's points were actually good, not just
    # a bare number with nothing to compare against. Best-effort: if no
    # bootstrap is available, average_entry_score is left null rather than
    # failing the whole function over a context nicety.
    average_by_gw = {}
    try:
        bootstrap = _load_bootstrap()
        average_by_gw = {e["id"]: e.get("average_entry_score") for e in bootstrap["events"]}
    except FileNotFoundError:
        pass

    df = pd.DataFrame([
        {
            "gw": g["event"],
            "points": g["points"],
            "total_points": g["total_points"],
            "overall_rank": g["overall_rank"],
            "bank": g["bank"] / 10.0,
            "value": g["value"] / 10.0,
            "event_transfers": g["event_transfers"],
            "event_transfers_cost": g["event_transfers_cost"],
            "points_on_bench": g["points_on_bench"],
            # FPL's own real "you're in the top X%" figure for this
            # gameweek's overall rank -- already computed by FPL itself, not
            # derived here. Cast from string (FPL publishes it as one).
            "overall_rank_percentage": float(g["overall_rank_percentage"]) if g.get("overall_rank_percentage") else None,
            "average_entry_score": average_by_gw.get(g["event"]),
        }
        for g in current
    ])
    return df.sort_values("gw").reset_index(drop=True)


def calculate_free_transfers(entry_id: int) -> int:
    """This manager's REAL banked free transfers going into the NEXT
    gameweek, computed from their actual transfer history -- not a manual
    guess/slider. FPL's real rule (verified against the live API's
    game_settings: max_extra_free_transfers == 4, i.e. 1 this week + 4
    banked == 5 max in a single week -- see optimizer.py's
    MAX_FREE_TRANSFERS_BANKED): each gameweek that passes with fewer
    transfers made than were available banks the difference forward (capped
    at 5 total); each transfer made consumes one banked transfer first, with
    any beyond that counted as a paid hit (already reflected in that
    gameweek's own event_transfers_cost, which this function doesn't need to
    re-derive).

    GW1 (the initial squad build) is EXCLUDED from this simulation --
    building your season-opening squad isn't a "transfer" in FPL's own
    accounting (event_transfers is 0 for GW1 regardless of squad size), so
    treating it as a transfer gameweek would incorrectly consume a bank
    entry. Free-transfer accounting genuinely starts from GW2 onward.

    Returns 1 (the standard single free transfer) if there's no season
    history to simulate from yet (e.g. only GW1 has been played) -- the
    correct real starting point for GW2, not a guess."""
    progress = load_current_season_progress(entry_id)
    if progress.empty:
        return 1

    banked = 1  # every manager starts GW2 with exactly 1 free transfer
    for _, row in progress[progress["gw"] > 1].sort_values("gw").iterrows():
        used = min(int(row["event_transfers"]), banked)
        banked = min(banked - used + 1, 1 + MAX_FREE_TRANSFERS_BANKED)
    return banked


_DASHBOARD_CURRENT_SQUAD_FALLBACK = os.path.join(PROJECT_DIR, "data", "dashboard_current_squad.json")


@st.cache_data(ttl=60)
def load_current_squad_picks(entry_id: int, gw: int) -> dict:
    """This manager's REAL picks for one gameweek of the season actually in
    progress. Tries the real FPL API directly first (fpl_api.get_entry_picks,
    the same endpoint the collector itself uses) so a squad/points refresh
    is genuinely live (at most 60s old, st.cache_data's ttl) rather than
    waiting on the collector's own daily batch run -- this is what makes
    "live synced" real instead of a once-a-day snapshot. get_entry_picks()
    itself already returns None (not an exception) for a gameweek whose
    deadline hasn't passed yet, matching this function's existing contract.

    Falls back, in order, to: the collector's saved
    entry/{id}/picks/gw{n}.json (local dev environment with a recent
    snapshot), then data/dashboard_current_squad.json (a single,
    deliberately committed copy of this manager's own picks + that
    gameweek's live points, refreshed daily by the scheduled collector
    workflow -- see .github/workflows/weekly-collector.yml). Only reached
    if the live API call itself fails (network hiccup, FPL API down,
    offline dev environment) -- normal operation never needs these.
    Returns None if genuinely nothing's available in this environment."""
    try:
        live_picks = fpl_api.get_entry_picks(entry_id, gw)
        if live_picks is not None:
            return live_picks
    except Exception:
        pass
    path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "entry", str(entry_id), "picks", f"gw{gw}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(_DASHBOARD_CURRENT_SQUAD_FALLBACK):
        with open(_DASHBOARD_CURRENT_SQUAD_FALLBACK, encoding="utf-8") as f:
            bundle = json.load(f)
        return bundle.get("picks") if bundle.get("gw") == gw else None
    return None


_POSITION_MAP_APP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@st.cache_data(ttl=60)
def load_live_gw_points(gw: int) -> dict:
    """Every player's real points for one gameweek in progress (or finished).
    Tries the real FPL API directly first (fpl_api.get_event_live(), the
    same endpoint the collector uses, one call for every player rather than
    N per-player requests) -- refreshed at most 60s old (st.cache_data's
    ttl) so gameweek points genuinely update live during a match, matching
    how FPL's own app itself only refreshes provisional bonus/live points
    roughly once a minute.

    Falls back, in order, to the collector's saved
    data/raw/2026-27/live/gw{n}.json, then the same
    data/dashboard_current_squad.json bundle load_current_squad_picks uses.
    Only reached if the live API call fails. Returns an empty dict if
    genuinely nothing's available for this gameweek in any form."""
    try:
        live = fpl_api.get_event_live(gw)
        return {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
    except Exception:
        pass
    path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "live", f"gw{gw}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            live = json.load(f)
        return {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
    if os.path.exists(_DASHBOARD_CURRENT_SQUAD_FALLBACK):
        with open(_DASHBOARD_CURRENT_SQUAD_FALLBACK, encoding="utf-8") as f:
            bundle = json.load(f)
        if bundle.get("gw") == gw:
            return {int(k): v for k, v in bundle.get("live_points", {}).items()}
    return {}


@st.cache_data(ttl=60)
def load_live_gw_stats(gw: int) -> dict:
    """Every player's FULL real per-gameweek stats dict for one gameweek
    (minutes, goals_scored, assists, total_points, bonus, bps, etc. --
    every field FPL's own event/{gw}/live endpoint returns per player, not
    a fixed subset), keyed by player id. Same live-first/60s-ttl/fallback
    pattern as load_live_gw_points/load_live_gw_minutes -- kept as a
    separate function (rather than changing what those two return) so
    existing callers of the narrower point/minutes-only functions are
    unaffected; this one is for callers that need the fuller picture (see
    build_live_squad_df's hover-tooltip fields on My Squad).

    `bonus` here is FPL's own REAL bonus points already awarded for this
    gameweek -- 0 until bonus is calculated (mid-match or shortly after
    full time), then the real final number once it is. Deliberately not a
    projected/expected bonus derived from BPS ranking -- BPS ranking
    changes constantly while a match is live, so a "projected bonus" would
    be this project's own guess dressed up as a fact FPL doesn't publish,
    not a real number.

    Falls back to the collector's saved data/raw/2026-27/live/gw{n}.json if
    the live call fails. Returns an empty dict if genuinely nothing's
    available."""
    try:
        live = fpl_api.get_event_live(gw)
        return {e["id"]: e["stats"] for e in live["elements"]}
    except Exception:
        pass
    path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "live", f"gw{gw}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        live = json.load(f)
    return {e["id"]: e["stats"] for e in live["elements"]}


@st.cache_data(ttl=60)
def load_live_gw_minutes(gw: int) -> dict:
    """Every player's real MINUTES for one gameweek -- used to tell apart
    "played and scored 0" from "didn't get any game time at all," a
    distinction a bare points number can't make (0 pts reads as a harsh/
    unfair result for a player who simply wasn't selected to play, when it's
    really just "nothing to report yet"). Tries the real FPL API first
    (fpl_api.get_event_live(), same live endpoint as load_live_gw_points,
    60s ttl) for genuinely live minutes, not once-a-day-stale data.

    Falls back to the collector's saved data/raw/2026-27/live/gw{n}.json if
    the live call fails. No further fallback bundle for this one -- the
    committed dashboard_current_squad.json fallback only stores points, not
    minutes (see refresh_dashboard_fallbacks.py) -- returns an empty dict
    in that case, and callers should treat a missing entry as unknown, not
    as "definitely played 0 minutes.\""""
    try:
        live = fpl_api.get_event_live(gw)
        return {e["id"]: e["stats"]["minutes"] for e in live["elements"]}
    except Exception:
        pass
    path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "live", f"gw{gw}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        live = json.load(f)
    return {e["id"]: e["stats"]["minutes"] for e in live["elements"]}


@st.cache_data(ttl=60)
def _team_fixture_started(gw: int) -> dict:
    """Real, per-team "has this team's gameweek-{gw} match actually kicked
    off" flag, from fixtures.csv's own real `started` field -- checked
    directly against the live API alongside `finished`: `started` flips
    True the moment a match kicks off and stays True for the rest of the
    gameweek, while `finished`/`finished_provisional` only flip once the
    match (or its bonus points) are done. Used to tell apart a squad
    member's real 0 minutes because their match simply hasn't happened yet
    this gameweek vs. their match already being over and they didn't
    feature -- two different real situations a bare "0 pts"/generic
    "no game time" label would otherwise conflate.

    Returns {team_name: True/False}. A team missing from the result
    (shouldn't normally happen for a real fixture list) should be treated
    by callers as "unknown," not "hasn't started.\""""
    fx = _load_fixtures_df()
    if fx.empty:
        return {}
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}
    gw_fixtures = fx[fx["event"] == gw]
    started = {}
    for _, row in gw_fixtures.iterrows():
        home, away = team_by_id.get(row["team_h"]), team_by_id.get(row["team_a"])
        if home:
            started[home] = bool(row["started"])
        if away:
            started[away] = bool(row["started"])
    return started


def build_live_squad_df(picks_data: dict, gw: int) -> pd.DataFrame:
    """Turn one gameweek's real picks (from load_current_squad_picks) into a
    DataFrame in the same shape render_pitch()/optimize_squad() already
    expect (player_id/name/position/team/cost/predicted_points/
    in_starting_xi), so this manager's REAL squad can reuse the exact same
    pitch-view rendering as every optimizer-built squad -- no separate
    display logic to maintain. `predicted_points` here is each player's REAL
    points scored that gameweek (from load_live_gw_points), not a
    prediction -- named that only for compatibility with the shared
    rendering code, which doesn't care what produced the number."""
    raw = _load_bootstrap()
    element_by_id = {p["id"]: p for p in raw["elements"]}
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}
    live_points = load_live_gw_points(gw)
    live_minutes = load_live_gw_minutes(gw)
    live_stats = load_live_gw_stats(gw)
    fixtures_by_team = team_upcoming_fixtures(3)
    fixture_started_by_team = _team_fixture_started(gw)

    rows = []
    for pick in picks_data["picks"]:
        eid = pick["element"]
        player = element_by_id.get(eid)
        if player is None:
            continue
        # Distinguishes "played and scored 0" from "didn't get any minutes
        # at all" -- a bare 0 reads as an unfairly harsh result for a bench
        # player who simply wasn't selected to play, when it's really just
        # "nothing to report yet." Missing from live_minutes entirely (empty
        # dict, e.g. no fallback data) is treated as unknown, not as 0
        # minutes -- see load_live_gw_minutes' docstring.
        minutes = live_minutes.get(eid)
        team_name = team_by_id.get(player["team"])
        # A real 0-minutes player is further split by whether their team's
        # own real fixture has actually kicked off yet this gameweek (see
        # _team_fixture_started) -- "hasn't played yet" (fixture not
        # started) and "played 0 minutes in a finished/live match" are two
        # different real situations, not the same "no game time" state.
        fixture_started = fixture_started_by_team.get(team_name)
        # Real per-gameweek stats for the My Squad hover tooltip -- minutes,
        # goals, assists, real bonus already awarded, and the real
        # unmultiplied gameweek total (distinct from predicted_points below,
        # which IS multiplier-applied for captaincy -- the tooltip shows the
        # player's own real total before any captain doubling, since that's
        # the number FPL itself reports for the player individually).
        stats = live_stats.get(eid, {})
        rows.append({
            "player_id": eid,
            "player_code": player["code"],
            "name": f"{player['first_name']} {player['second_name']}",
            "position": _POSITION_MAP_APP[player["element_type"]],
            "team": team_name,
            "cost": player["now_cost"] / 10.0,
            "predicted_points": live_points.get(eid, 0) * pick["multiplier"],
            "did_not_play": minutes == 0,
            "fixture_started": fixture_started,
            "minutes_played": stats.get("minutes"),
            "goals_scored": stats.get("goals_scored"),
            "assists": stats.get("assists"),
            "bonus": stats.get("bonus"),
            "gw_total_points": stats.get("total_points"),
            "in_starting_xi": pick["multiplier"] > 0,
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            # Links this squad view into the SAME real fixture-difficulty
            # data Transfers uses (see team_upcoming_fixtures) -- shown as a
            # color-coded strip on each card by _player_card_html, rather
            # than being a separate, disconnected fixture display.
            "next_fixtures": fixtures_by_team.get(team_name, []),
            # FPL's OWN real "expected points next gameweek" prediction --
            # not derived here, taken directly from bootstrap-static, since
            # it already factors in FPL's own view of form + fixture +
            # availability. Used by suggest_captain() below to recommend a
            # captain from THIS squad using the same real number FPL itself
            # publishes, rather than a separate homegrown estimate.
            "ep_next": float(player.get("ep_next") or 0),
            # Real, confirmed set-piece duty -- FPL's own penalties_order
            # field (== 1 means this player is their club's PRIMARY penalty
            # taker, not a guess). Shown as a badge on the pitch card by
            # _player_card_html, same pattern as the captain/dreamteam
            # badges already there.
            "is_penalty_taker": player.get("penalties_order") == 1,
        })
    return pd.DataFrame(rows)


def suggest_captain(squad_df: pd.DataFrame) -> pd.Series:
    """Recommend a captain from a real live squad (build_live_squad_df's
    output) using FPL's OWN ep_next field -- the same real, published
    "expected points next gameweek" number bootstrap-static exposes for
    every player, not a separate homegrown prediction. Only considers
    STARTING XI players (captaining a bench player who might not even play
    is never the right call, regardless of their ep_next). Returns the
    starter with the highest ep_next as a pandas Series (one row of
    squad_df); returns None if the squad has no starters with a positive
    ep_next (e.g. very early in a season before FPL's own model has enough
    signal, or a completely empty squad)."""
    starters = squad_df[squad_df["in_starting_xi"]]
    candidates = starters[starters["ep_next"] > 0]
    if candidates.empty:
        return None
    # A tie on ep_next (real and not rare -- checked directly: 3 starters at
    # exactly 4.0 in a real GW1 squad) breaks toward attacking output, since
    # captaincy doubles the SAME real points either way, but an attacker's
    # ep_next carries more upside variance (a goal/assist swing) than a
    # goalkeeper's or defender's floor-heavy one -- position order here
    # (FWD > MID > DEF > GK) is the standard "attacking upside" ordering,
    # not an arbitrary tiebreak.
    POSITION_PRIORITY = {"FWD": 0, "MID": 1, "DEF": 2, "GK": 3}
    candidates = candidates.copy()
    candidates["_tiebreak"] = candidates["position"].map(POSITION_PRIORITY)
    best = candidates.sort_values(["ep_next", "_tiebreak"], ascending=[False, True]).iloc[0]
    return best.drop("_tiebreak")


_DASHBOARD_LEAGUES_FALLBACK = os.path.join(PROJECT_DIR, "data", "dashboard_leagues.json")


@st.cache_data(ttl=60)
def load_joined_leagues(entry_id: int) -> list:
    """This manager's PRIVATE classic leagues (joined by code, not FPL's
    auto-generated global/region/club leagues -- league_type == "x", same
    filter snapshot.py's collector itself uses), each with its full real
    standings. Tries the real FPL API directly first (fpl_api.get_entry()
    for the league id list, then fpl_api.get_league_standings() per league,
    same calls the collector makes) so standings are genuinely live (60s
    ttl) rather than a once-a-day batch snapshot.

    Falls back, in order, to the collector's saved
    entry/{id}/leagues/{league_id}.json files, then
    data/dashboard_leagues.json (a single, deliberately committed bundle of
    every private league's standings, refreshed daily by the scheduled
    collector workflow). Only reached if the live API calls fail. Returns
    an empty list if genuinely nothing's available in this environment."""
    try:
        entry_info = fpl_api.get_entry(entry_id)
        classic_leagues = entry_info.get("leagues", {}).get("classic", [])
        private_leagues = [l for l in classic_leagues if l.get("league_type") == "x"]
        return [fpl_api.get_league_standings(l["id"]) for l in private_leagues]
    except Exception:
        pass
    leagues_dir = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "entry", str(entry_id), "leagues")
    if os.path.isdir(leagues_dir):
        leagues = []
        for fname in sorted(os.listdir(leagues_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(leagues_dir, fname), encoding="utf-8") as f:
                leagues.append(json.load(f))
        if leagues:
            return leagues
    if os.path.exists(_DASHBOARD_LEAGUES_FALLBACK):
        with open(_DASHBOARD_LEAGUES_FALLBACK, encoding="utf-8") as f:
            return json.load(f)
    return []


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

    raw = _load_bootstrap()
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


@st.cache_data(ttl=60)
def _load_bootstrap() -> dict:
    """Live-first replacement for every `with open(_latest_bootstrap_path())`
    call site: calls the real FPL bootstrap-static API directly (the same
    endpoint fpl_api.get_bootstrap_static() the collector itself uses) so
    scores/points/prices/ownership are genuinely live -- refreshed at most
    60 seconds old (st.cache_data's ttl), not once-a-day-batch stale. Falls
    back to _latest_bootstrap_path()'s file (local data/raw/ snapshot, then
    the committed data/dashboard_bootstrap.json) if the live call fails for
    any reason (network hiccup, FPL API down, offline dev environment) --
    every caller keeps working exactly as before, just live when possible.

    A short TTL (not 0/none) rather than no caching at all: FPL's own site
    itself only updates provisional bonus/live points roughly once a minute
    during matches, so re-fetching more often than that buys nothing real
    and just adds load for no benefit."""
    try:
        return fpl_api.get_bootstrap_static()
    except Exception:
        with open(_latest_bootstrap_path(), encoding="utf-8") as f:
            return json.load(f)


def is_gameweek_live() -> bool:
    """Real, current "is a gameweek actually in progress right now" check --
    FPL's own is_current flag AND NOT finished (checked directly: is_current
    stays True for the current gameweek even after full time, until the
    NEXT gameweek's deadline passes, so is_current alone isn't enough --
    finished only flips once bonus points are fully locked in, correctly
    staying False while a match is still live or awaiting bonus).

    Used to gate auto-refresh (see app.py's My Squad tab): polling FPL's
    live API every 60s is only worth doing while a real gameweek is
    actually live -- pointless (and wasteful) the rest of the week, when
    nothing on this page is actually changing minute to minute."""
    try:
        raw = _load_bootstrap()
    except FileNotFoundError:
        return False
    for event in raw.get("events", []):
        if event.get("is_current"):
            return not event.get("finished", False)
    return False


def _fixtures_path() -> str:
    """Same fallback pattern as _latest_bootstrap_path(): data/raw/2026-27/
    fixtures.csv is gitignored (lives under data/raw/), so a fresh Streamlit
    Cloud deploy has none of it and every fixture-dependent feature (PL
    Table, fixture-difficulty strips/adjustments) would silently show
    nothing. Falls back to the deliberately-committed, non-timestamped
    data/dashboard_fixtures.csv (refreshed by
    src/collector/refresh_dashboard_fallbacks.py, same day-stale bound
    already accepted for data/dashboard_bootstrap.json). Returns None (not
    an exception) if neither exists -- these features already treat a
    missing fixtures file as "nothing collected yet," not an error."""
    live_path = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "fixtures.csv")
    if os.path.exists(live_path):
        return live_path
    fallback = os.path.join(PROJECT_DIR, "data", "dashboard_fixtures.csv")
    if os.path.exists(fallback):
        return fallback
    return None


@st.cache_data(ttl=60)
def _load_fixtures_df() -> pd.DataFrame:
    """Live-first replacement for reading _fixtures_path() as a CSV: calls
    the real FPL fixtures API directly (fpl_api.get_fixtures(), same schema
    as fixtures.csv -- checked directly, identical column names) so real
    match scores/results are live (60s ttl) rather than waiting on the
    once-a-day collector batch -- this is what makes PL Table and every
    fixture-difficulty feature genuinely live instead of a day stale.

    Falls back to _fixtures_path()'s file (local data/raw/ snapshot, then
    the committed data/dashboard_fixtures.csv) if the live call fails.
    Returns an empty DataFrame (not an exception) if neither the live call
    nor any fallback file is available -- callers already treat that as
    "nothing collected yet.\""""
    try:
        return pd.DataFrame(fpl_api.get_fixtures())
    except Exception:
        pass
    path = _fixtures_path()
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def live_price_changes() -> pd.DataFrame:
    """Real 2026-27 price movement SO FAR this season, straight from FPL's
    own bootstrap-static -- `now_cost` (current price) and `cost_change_start`
    (FPL's own real cumulative change since the season's opening prices,
    verified directly against bootstrap-static: positive = risen, negative =
    fallen). No GW-by-GW historical table needed here, unlike the Historical
    page's season_insights() -- FPL tracks this delta itself in real time, so
    this is always exactly as current as the latest collected bootstrap
    snapshot (refreshed daily by the scheduled collector workflow -- see
    Automated Collection in the README -- so it updates week by week
    automatically as the season progresses, no separate wiring needed).

    Returns every player with any real price movement so far, sorted by
    biggest riser first, with columns: name, position, team, start_cost
    (now_cost - cost_change_start, i.e. back-derived from FPL's own delta,
    not a separate lookup), cost (current), price_change. Empty (not an
    error) if literally nobody has moved yet, which is the correct state
    very early in a season before FPL's price-change algorithm has reacted
    to any real transfer activity."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        change = p["cost_change_start"]
        if change == 0:
            continue
        rows.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "cost": p["now_cost"] / 10.0,
            "price_change": change / 10.0,
            "start_cost": (p["now_cost"] - change) / 10.0,
            "net_transfers": p["transfers_in_event"] - p["transfers_out_event"],
        })
    df = pd.DataFrame(rows, columns=["name", "position", "team", "start_cost", "cost", "price_change", "net_transfers"])
    return df.sort_values("price_change", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60)
def likely_price_movers(top_n: int = 10) -> pd.DataFrame:
    """Players with real, current transfer MOMENTUM (net transfers this
    gameweek: transfers_in_event - transfers_out_event, both real FPL fields
    verified directly against bootstrap-static) who HAVEN'T had their price
    move yet -- a real, public leading indicator of who's likely to rise or
    fall soon, distinct from live_price_changes() (which shows movement that
    has ALREADY happened). Excludes anyone already in live_price_changes()'s
    output, since a player already moving doesn't need a "likely to move"
    flag -- links the two features together rather than letting them overlap
    silently.

    Returns up to top_n risers (positive net transfers) and top_n fallers
    (negative net transfers) combined, sorted by |net_transfers| descending,
    with columns: name, position, team, cost, net_transfers. Real momentum
    numbers reset each gameweek (transfers_in_event is THIS gameweek's
    activity, not cumulative), so this naturally refreshes automatically as
    the collector runs -- same "no separate wiring needed" property as
    live_price_changes()."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}
    already_moved = set(live_price_changes()["name"])

    rows = []
    for p in raw["elements"]:
        net = p["transfers_in_event"] - p["transfers_out_event"]
        if net == 0:
            continue
        name = f"{p['first_name']} {p['second_name']}"
        if name in already_moved:
            continue
        rows.append({
            "name": name,
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "cost": p["now_cost"] / 10.0,
            "net_transfers": net,
        })
    df = pd.DataFrame(rows, columns=["name", "position", "team", "cost", "net_transfers"])
    df["_abs"] = df["net_transfers"].abs()
    return df.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(top_n * 2).reset_index(drop=True)


@st.cache_data(ttl=60)
def tonight_price_projections(top_n: int = 10) -> pd.DataFrame:
    """FPL's OWN real price-change forecast for the very next update --
    distinct from likely_price_movers() (this project's own momentum-based
    guess from transfers_in_event/transfers_out_event). This is FPL's own
    published projection, read verbatim from each player's real
    `price_change_projections` list (offset 0 = the next scheduled price
    update) and `likelihood` field (a real -5..+5 scale FPL itself computes
    -- verified directly against bootstrap-static: -5 is their strongest
    "will fall" signal, +5 their strongest "will rise" signal, 0 means no
    change expected).

    Deliberately doesn't claim a specific clock time for "tonight's"
    update -- FPL's public API doesn't expose one, and this project only
    surfaces fields it can verify directly rather than assuming a schedule.

    Returns up to top_n likely risers (likelihood > 0) and top_n likely
    fallers (likelihood < 0), sorted by |likelihood| descending, with
    columns: name, position, team, cost, projected_percent, likelihood.
    Players already excluded from FPL's own price-change calculation
    (`price_change_calibrating` -- new signings/promoted-team players FPL
    hasn't built enough transfer history on yet) are skipped, since their
    projection field isn't meaningful yet."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        if p.get("price_change_calibrating"):
            continue
        projections = p.get("price_change_projections") or []
        tonight = next((proj for proj in projections if proj.get("offset") == 0), None)
        if tonight is None or tonight.get("likelihood") == 0:
            continue
        rows.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "cost": p["now_cost"] / 10.0,
            "projected_percent": float(tonight["projected_percent"]),
            "likelihood": tonight["likelihood"],
        })
    df = pd.DataFrame(rows, columns=["name", "position", "team", "cost", "projected_percent", "likelihood"])
    if df.empty:
        return df
    df["_abs"] = df["likelihood"].abs()
    return df.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(top_n * 2).reset_index(drop=True)


@st.cache_data(ttl=60)
def differential_finder(max_ownership: float = 10.0, min_ep_next: float = 2.0, top_n: int = 15) -> pd.DataFrame:
    """Real 'differential' picks -- low-owned players (FPL's own real
    selected_by_percent field) with real, meaningful upside (FPL's own
    ep_next, the same field suggest_captain()/the Chip Advisor tab already
    use), rather than a homegrown score. A genuine FPL strategy concept
    (bringing in a player few other managers have, for a rank-gaining edge
    if they do well) quantified from real, checkable numbers, not a vague
    "under the radar" guess.

    max_ownership: exclude anyone owned by more than this % of managers --
    default 10% is a common real-world differential threshold, not an
    arbitrary project-specific number.
    min_ep_next: exclude anyone FPL itself doesn't expect to score
    meaningfully next gameweek -- a differential nobody expects to actually
    perform isn't a useful pick regardless of how low-owned they are.

    Returns up to top_n players sorted by ep_next descending, with columns:
    name, position, team, cost, selected_by_percent, ep_next,
    is_penalty_taker (FPL's own penalties_order == 1 -- confirmed set-piece
    duty is a real reason a low-owned player's upside is more trustworthy
    than ep_next alone would suggest, same signal build_live_squad_df's
    penalty-taker badge already surfaces on squad cards)."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        owned = float(p["selected_by_percent"])
        ep_next = float(p["ep_next"] or 0)
        if owned > max_ownership or ep_next < min_ep_next:
            continue
        rows.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "cost": p["now_cost"] / 10.0,
            "selected_by_percent": owned,
            "ep_next": ep_next,
            "is_penalty_taker": p.get("penalties_order") == 1,
        })
    df = pd.DataFrame(rows, columns=["name", "position", "team", "cost", "selected_by_percent", "ep_next", "is_penalty_taker"])
    return df.sort_values("ep_next", ascending=False).head(top_n).reset_index(drop=True)


@st.cache_data(ttl=60)
def league_wide_status_flags() -> pd.DataFrame:
    """Every real, currently-flagged player league-wide (not just this
    manager's own squad) -- the SAME real FPL fields (status/news/
    chance_of_playing_next_round) the Transfers tab already uses to flag a
    squad member as injured/suspended/doubtful, just applied to every
    player instead of only the 15 in one manager's squad. Meant for
    scouting a potential transfer-IN too: a low-owned name with a scary
    points total might just be a player who's been out injured, not a bad
    pick -- and a fringe replacement can be checked for their OWN status
    before being suggested.

    Returns name, position, team, cost, selected_by_percent, status, news,
    chance_of_playing_next_round for every player whose real status != 'a'
    (available), sorted by ownership descending (higher-owned flags are the
    ones most managers actually need to know about first)."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        if p["status"] == "a":
            continue
        rows.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "cost": p["now_cost"] / 10.0,
            "selected_by_percent": float(p["selected_by_percent"]),
            "status": p["status"],
            "news": p.get("news") or "",
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
        })
    df = pd.DataFrame(rows, columns=[
        "name", "position", "team", "cost", "selected_by_percent",
        "status", "news", "chance_of_playing_next_round",
    ])
    return df.sort_values("selected_by_percent", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60)
def premier_league_table(through_gw: int = None) -> pd.DataFrame:
    """The real Premier League table for the season in progress -- computed
    directly from fixtures.csv's own real team_h_score/team_a_score for
    every match that has an actual result recorded, not FPL's own strength
    ratings or any derived metric. Uses team_h_score.notna() (a real score
    has been recorded) rather than the 'finished' column, since 'finished'
    verifiably stays False on a played match until bonus points are fully
    locked in -- real final scores are already known well before that, so
    waiting on 'finished' would hide results for hours after a match
    actually ends.

    through_gw, if given, restricts to fixtures with event <= through_gw --
    used to compute the REAL table as it stood after an earlier gameweek
    (e.g. to diff table movement since last week in the PL Table tab),
    rather than only ever the current, all-played-matches table. None (the
    default) includes every played match regardless of gameweek, i.e. the
    current table.

    Returns team, played, won, drawn, lost, gf, ga, gd, points, sorted by
    points then goal difference descending -- the real, standard PL
    table-ordering rule."""
    fx = _load_fixtures_df()
    if fx.empty:
        return pd.DataFrame(columns=["team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"])
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    played = fx[fx["team_h_score"].notna() & fx["team_a_score"].notna()]
    if through_gw is not None:
        played = played[played["event"] <= through_gw]

    stats = {tid: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0} for tid in team_by_id}
    for _, row in played.iterrows():
        h, a = int(row["team_h"]), int(row["team_a"])
        hs, as_ = int(row["team_h_score"]), int(row["team_a_score"])
        for tid, gf, ga in ((h, hs, as_), (a, as_, hs)):
            stats[tid]["played"] += 1
            stats[tid]["gf"] += gf
            stats[tid]["ga"] += ga
            if gf > ga:
                stats[tid]["won"] += 1
            elif gf == ga:
                stats[tid]["drawn"] += 1
            else:
                stats[tid]["lost"] += 1

    rows = []
    for tid, s in stats.items():
        points = s["won"] * 3 + s["drawn"]
        rows.append({
            "team": team_by_id[tid],
            "played": s["played"], "won": s["won"], "drawn": s["drawn"], "lost": s["lost"],
            "gf": s["gf"], "ga": s["ga"], "gd": s["gf"] - s["ga"], "points": points,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60)
def premier_league_table_with_movement() -> pd.DataFrame:
    """premier_league_table()'s current table, with a real `movement`
    column added: each team's rank NOW minus their rank as of ONE gameweek
    earlier (both computed from the exact same real fixtures data via
    premier_league_table(through_gw=...), not a stored/guessed snapshot --
    genuinely re-derivable at any time from fixtures.csv alone). Positive
    movement means the team has risen that many places since last
    gameweek's table, negative means they've fallen, 0 means unchanged.

    Uses the real current gameweek (bootstrap's own is_current) as the
    reference point -- movement is null (not 0) for GW1, since there's no
    real "last gameweek" table to compare against yet, and for any team
    that didn't exist in the through_gw-1 table for some reason (shouldn't
    normally happen with a fixed 20-team league, but guarded rather than
    assumed)."""
    current = premier_league_table()
    if current.empty:
        return current.assign(movement=pd.Series(dtype="Int64"))

    raw = _load_bootstrap()
    current_events = [e["id"] for e in raw["events"] if e.get("is_current")]
    current_gw = current_events[0] if current_events else 1
    if current_gw <= 1:
        return current.assign(movement=pd.Series([None] * len(current), dtype="Int64"))

    previous = premier_league_table(through_gw=current_gw - 1)
    current = current.reset_index(drop=True)
    previous = previous.reset_index(drop=True)
    current_rank = {row["team"]: i + 1 for i, row in current.iterrows()}
    previous_rank = {row["team"]: i + 1 for i, row in previous.iterrows()}

    movement = []
    for team in current["team"]:
        prev = previous_rank.get(team)
        now = current_rank.get(team)
        movement.append((prev - now) if prev is not None and now is not None else None)
    current["movement"] = pd.array(movement, dtype="Int64")
    return current


@st.cache_data(ttl=60)
def team_insights(top_n: int = 5) -> dict:
    """Real, current-season team-level insights derived from
    premier_league_table()'s own gf/ga (no separate fixtures.csv read --
    reuses the same real computed table rather than duplicating the
    scoring logic). Returns a dict of DataFrames:

    - "best_attack": teams sorted by real goals scored (gf) descending.
    - "best_defense": teams sorted by real goals conceded (ga) ascending
      (fewest first -- the actual real defensive record, not goal
      difference, which best_attack/best_defense would otherwise conflate).
    - "most_owned_players": each team's single most-selected player
      (FPL's own real selected_by_percent), one row per team with at least
      one played minute -- a real, checkable "who's the team's most-picked
      asset" view, not a guess.

    Empty DataFrames (not an error) if premier_league_table() has no
    results yet this season."""
    table = premier_league_table()
    if table.empty or table["played"].sum() == 0:
        empty_cols = {
            "best_attack": ["team", "played", "gf"],
            "best_defense": ["team", "played", "ga"],
            "most_owned_players": ["team", "name", "position", "selected_by_percent"],
        }
        return {k: pd.DataFrame(columns=v) for k, v in empty_cols.items()}

    played_table = table[table["played"] > 0]
    best_attack = played_table[["team", "played", "gf"]].sort_values("gf", ascending=False).head(top_n).reset_index(drop=True)
    best_defense = played_table[["team", "played", "ga"]].sort_values("ga", ascending=True).head(top_n).reset_index(drop=True)

    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        rows.append({
            "team": team_by_id.get(p["team"]),
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "selected_by_percent": float(p["selected_by_percent"]),
        })
    players_df = pd.DataFrame(rows)
    most_owned = (
        players_df.sort_values("selected_by_percent", ascending=False)
        .drop_duplicates("team", keep="first")
        .sort_values("selected_by_percent", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "best_attack": best_attack,
        "best_defense": best_defense,
        "most_owned_players": most_owned,
    }


@st.cache_data(ttl=60)
def season_leaderboards(top_n: int = 10) -> dict:
    """Real, current 2026-27 season leaderboards -- every number here is a
    single FPL bootstrap-static field read directly, no derived/invented
    metric and no minimum-appearance cutoff (early in a season that would
    just as easily exclude a real 1-game leader as a fluke). Returns a dict
    of DataFrames, each with name/position/team/value columns:

    - "golden_boot": real goals_scored, most first.
    - "assists": real assists, most first.
    - "yellow_cards": real yellow_cards, most first.
    - "red_cards": real red_cards, most first (only includes players with
      at least 1 -- a red-card "leaderboard" full of zeroes isn't useful).
    - "defensive_contribution": FPL's own real defensive_contribution stat
      (the 2025-26+ DEFCON scoring category -- tackles+interceptions+
      clearances for defenders, ball recoveries+tackles+interceptions for
      mid/fwd -- read verbatim, not recomputed).
    - "mvp": real total_points leader -- deliberately the plain, already-
      trusted FPL number rather than a model-derived score, since a trained
      model would need real in-season signal this project doesn't have yet
      this early (same caveat already shown on the Transfers tab).
    - "in_form": FPL's own real `form` field (their published rolling
      average points over recent gameweeks, not this project's own
      recompute) -- the single number that actually separates players once
      several gameweeks exist, unlike season totals which barely
      differentiate after just 1-2 games. Same "no cutoff" reasoning as
      every other board here: early on, form and total_points will look
      similar, and that's a correct, honest reflection of there being only
      a handful of real gameweeks so far, not a bug to hide."""
    raw = _load_bootstrap()
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}

    rows = []
    for p in raw["elements"]:
        rows.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": _POSITION_MAP_APP[p["element_type"]],
            "team": team_by_id.get(p["team"]),
            "goals_scored": p.get("goals_scored") or 0,
            "assists": p.get("assists") or 0,
            "yellow_cards": p.get("yellow_cards") or 0,
            "red_cards": p.get("red_cards") or 0,
            "defensive_contribution": p.get("defensive_contribution") or 0,
            "total_points": p.get("total_points") or 0,
            "form": float(p.get("form") or 0),
        })
    df = pd.DataFrame(rows)

    def _top(col, n=top_n, min_positive=True):
        d = df[df[col] > 0] if min_positive else df
        return (
            d[["name", "position", "team", col]]
            .rename(columns={col: "value"})
            .sort_values("value", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    return {
        "golden_boot": _top("goals_scored"),
        "assists": _top("assists"),
        "yellow_cards": _top("yellow_cards"),
        "red_cards": _top("red_cards"),
        "defensive_contribution": _top("defensive_contribution"),
        "mvp": _top("total_points"),
        "in_form": _top("form"),
    }


@st.cache_data(ttl=60)
def player_season_stats(player_id: int) -> dict:
    """Every real, current 2026-27 SEASON-cumulative stat for one player,
    read verbatim from bootstrap-static (no per-gameweek recomputation --
    FPL already tracks these as running season totals). Returns an empty
    dict if the player id isn't found (shouldn't normally happen for a
    real squad member, but guarded rather than assumed).

    Deliberately includes DEFCON (`defensive_contribution`) alongside the
    more commonly-shown goals/assists/bonus -- the 2025-26+ scoring
    category is easy to overlook since it's newer than the rest, but it's
    real points on the board same as anything else here. Numeric-as-string
    fields FPL publishes that way (`ict_index`, `points_per_game`,
    `selected_by_percent`, `form`, `expected_goals`/`expected_assists`/
    `expected_goal_involvements`) are cast to float here so callers never
    have to remember which fields need that."""
    raw = _load_bootstrap()
    player = next((p for p in raw["elements"] if p["id"] == player_id), None)
    if player is None:
        return {}
    team_by_id = {t["id"]: t["name"] for t in raw["teams"]}
    return {
        "name": f"{player['first_name']} {player['second_name']}",
        "team": team_by_id.get(player["team"]),
        "position": _POSITION_MAP_APP[player["element_type"]],
        "cost": player["now_cost"] / 10.0,
        "minutes": player["minutes"],
        "starts": player["starts"],
        "total_points": player["total_points"],
        "points_per_game": float(player["points_per_game"] or 0),
        "goals_scored": player["goals_scored"],
        "assists": player["assists"],
        "expected_goals": float(player["expected_goals"] or 0),
        "expected_assists": float(player["expected_assists"] or 0),
        "expected_goal_involvements": float(player["expected_goal_involvements"] or 0),
        "clean_sheets": player["clean_sheets"],
        "goals_conceded": player["goals_conceded"],
        "defensive_contribution": player["defensive_contribution"],
        "bonus": player["bonus"],
        "bps": player["bps"],
        "yellow_cards": player["yellow_cards"],
        "red_cards": player["red_cards"],
        "saves": player["saves"],
        "ict_index": float(player["ict_index"] or 0),
        "form": float(player["form"] or 0),
        "selected_by_percent": float(player["selected_by_percent"] or 0),
        "status": player["status"],
        "news": player.get("news") or "",
    }


@st.cache_data(ttl=60)
def team_upcoming_fixtures(n_gws: int = 3) -> dict:
    """Each real Premier League team's next N gameweeks' opponents and FPL's
    own published fixture-difficulty rating (1-5, verified directly against
    fixtures.csv), for the season actually in progress. Returns
    {team_name: [{"gw": int, "opponent": str, "is_home": bool, "difficulty": int}, ...]}.

    "Upcoming" is determined from the live bootstrap's own is_current/is_next
    gameweek flags, NOT the fixtures.csv `finished` column alone -- checked
    directly: GW1's own fixtures.csv rows show finished=False even for
    matches that have already kicked off and finished_provisional=True,
    since `finished` only flips once bonus points are fully locked in. Using
    the bootstrap's real current-gameweek number avoids treating an
    already-played (but not yet "finished") match as still upcoming.

    Returns an empty dict if no 2026-27 fixtures.csv has been collected yet
    in this environment (expected before the collector's first run)."""
    fx = _load_fixtures_df()
    if fx.empty:
        return {}

    raw = _load_bootstrap()
    team_id_to_name = {t["id"]: t["name"] for t in raw["teams"]}
    current_events = [e["id"] for e in raw["events"] if e.get("is_current")]
    current_gw = current_events[0] if current_events else 1

    upcoming = fx[fx["event"] >= current_gw].sort_values("event").head(n_gws * 10)

    result = {name: [] for name in team_id_to_name.values()}
    for _, row in upcoming.iterrows():
        if len(result.get(team_id_to_name.get(row["team_h"]), [])) < n_gws:
            home_name = team_id_to_name.get(row["team_h"])
            if home_name:
                result[home_name].append({
                    "gw": int(row["event"]), "opponent": team_id_to_name.get(row["team_a"]),
                    "is_home": True, "difficulty": int(row["team_h_difficulty"]),
                })
        if len(result.get(team_id_to_name.get(row["team_a"]), [])) < n_gws:
            away_name = team_id_to_name.get(row["team_a"])
            if away_name:
                result[away_name].append({
                    "gw": int(row["event"]), "opponent": team_id_to_name.get(row["team_h"]),
                    "is_home": False, "difficulty": int(row["team_a_difficulty"]),
                })
    return result


def average_fixture_difficulty(team: str, fixtures_by_team: dict) -> float:
    """Mean upcoming fixture difficulty for one team, from
    team_upcoming_fixtures()'s output -- a single number for quick sorting/
    comparison (e.g. "who has the easier run"). Returns None if that team
    has no upcoming fixtures data (shouldn't normally happen for a real
    Premier League team once fixtures.csv exists, but a promoted/relegated
    edge case or a blank gameweek could leave a team with fewer than
    expected)."""
    fixtures = fixtures_by_team.get(team, [])
    if not fixtures:
        return None
    return sum(f["difficulty"] for f in fixtures) / len(fixtures)


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
    raw = _load_bootstrap()
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
    # A live squad's bench/unused player who genuinely didn't get any
    # minutes shows a label instead of "0 pts" -- a bare 0 reads as a harsh/
    # unfair result for someone who simply wasn't selected to play. Split
    # into two real, distinct states using fixture_started (see
    # _team_fixture_started/build_live_squad_df) rather than one generic
    # label that would otherwise conflate them: "Not yet played" (their
    # team's real fixture hasn't kicked off yet this gameweek -- there's
    # still genuinely nothing to report) vs. "No game time" (their match
    # has started/finished and they simply weren't used). fixture_started
    # missing/None (unknown -- e.g. no fixtures data available) falls back
    # to the older, more conservative "No game time" wording rather than
    # guessing. Only applies to build_live_squad_df's real squads
    # (did_not_play column) -- optimizer-built squads never have this
    # column, so they keep showing a real points/predicted-points number.
    did_not_play = "did_not_play" in row.index and row["did_not_play"]
    if did_not_play:
        fixture_started = row.get("fixture_started") if "fixture_started" in row.index else None
        points_label = "Not yet played" if fixture_started is False else "No game time"
    elif is_season_pool:
        points_label = f'{row["predicted_points"]:.0f} pts'
    else:
        points_label = f'{row["predicted_points"]:.1f} pts'
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
    # Real, confirmed set-piece duty (FPL's own penalties_order == 1, see
    # build_live_squad_df) -- shares the bottom-right corner with scout_html
    # since the two never co-occur (scout_reasons only exists on the
    # now-removed Scout Picks pool; is_penalty_taker only on live squads).
    penalty_taker_html = (
        f'<div title="Club\'s #1 penalty taker" style="position: absolute; bottom: -6px; right: -6px; '
        f'background: #1a1a1a; color: #ffb300; border-radius: 50%; width: 16px; height: 16px; '
        f'font-size: 10px; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">P</div>'
        if "is_penalty_taker" in row.index and row["is_penalty_taker"] else ""
    )
    # Real captain/vice-captain marker for a manager's own live squad (see
    # build_live_squad_df) -- distinct from the MVP ⭐ badge, which marks the
    # single highest-SCORING player, not necessarily who was actually made
    # captain (a captain can score 0 and still be captain).
    captain_html = (
        f'<div title="Captain (points doubled)" style="position: absolute; bottom: -6px; left: -6px; '
        f'background: #ffb300; color: #1a1a1a; border-radius: 50%; width: 18px; height: 18px; '
        f'font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">C</div>'
        if "is_captain" in row.index and row["is_captain"] else
        f'<div title="Vice-captain" style="position: absolute; bottom: -6px; left: -6px; '
        f'background: #d8dde3; color: #1a1a1a; border-radius: 50%; width: 18px; height: 18px; '
        f'font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center; '
        f'box-shadow: 0 1px 3px rgba(0,0,0,0.4);">VC</div>'
        if "is_vice_captain" in row.index and row["is_vice_captain"] else ""
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
    # Real upcoming-fixture strip -- only present when the caller has
    # attached a `next_fixtures` column (a list of team_upcoming_fixtures()
    # entries for this player's team), which links this card into the SAME
    # real fixture-difficulty data Transfers now uses to gate suggestions
    # (see optimize_transfers' fixture-adjustment in app.py), rather than
    # being a separate, disconnected display. FPL's own 1 (easiest) - 5
    # (hardest) rating, color-coded the same way FPL's own site does
    # (green=easy, red=hard) so it reads instantly without a legend.
    DIFFICULTY_COLORS = {1: "#2a9650", 2: "#6cbf5a", 3: "#e8c547", 4: "#e0793a", 5: "#c83232"}
    fixtures_html = ""
    if "next_fixtures" in row.index and row["next_fixtures"]:
        chips = "".join(
            f'<span title="GW{f["gw"]}: {"vs" if f["is_home"] else "@"} {f["opponent"]} '
            f'(difficulty {f["difficulty"]}/5)" style="display: inline-block; width: 16px; '
            f'height: 16px; line-height: 16px; border-radius: 3px; '
            f'background: {DIFFICULTY_COLORS.get(f["difficulty"], "#999")}; color: white; '
            f'font-size: 9px; font-weight: 700; margin: 0 1px;">{f["difficulty"]}</span>'
            for f in row["next_fixtures"]
        )
        fixtures_html = f'<div style="margin-top: 3px;">{chips}</div>'
    # Real per-gameweek stat breakdown (this specific gameweek, only present
    # on a live squad's build_live_squad_df row -- an optimizer-built squad
    # has no single-gameweek minutes/goals/assists/bonus to show) shown as a
    # native browser hover tooltip (the title attribute -- same mechanism
    # every other badge on this card already uses).
    gw_stats_line = ""
    if "minutes_played" in row.index and pd.notna(row.get("minutes_played")):
        bonus = row.get("bonus")
        bonus_line = f"Bonus {int(bonus)}" if pd.notna(bonus) else "Bonus not yet calculated"
        gw_stats_line = (
            f"THIS GW -- Minutes: {int(row['minutes_played'])} | "
            f"Goals: {int(row.get('goals_scored') or 0)} | "
            f"Assists: {int(row.get('assists') or 0)} | "
            f"GW points: {int(row.get('gw_total_points') or 0)} | "
            f"{bonus_line}"
        )
    # Real SEASON-cumulative stats (player_season_stats(), 60s-ttl live
    # data) appended to the same tooltip -- requested explicitly so a
    # squad card's hover shows the full season picture, not just this one
    # gameweek's numbers. Only attempted when a real player_id is present
    # (every card-building path has this column) and the season lookup
    # actually finds the player (should always succeed for a real squad
    # member, but guarded rather than assumed).
    season_stats_line = ""
    if "player_id" in row.index and pd.notna(row.get("player_id")):
        season = player_season_stats(int(row["player_id"]))
        if season:
            season_stats_line = (
                f"SEASON -- Pts: {season['total_points']} | Mins: {season['minutes']} | "
                f"Goals: {season['goals_scored']} | Assists: {season['assists']} | "
                f"DEFCON: {season['defensive_contribution']} | Bonus: {season['bonus']} | "
                f"xG: {season['expected_goals']:.1f} | xA: {season['expected_assists']:.1f} | "
                f"Yellow/Red: {season['yellow_cards']}/{season['red_cards']}"
            )
    stats_tooltip = " || ".join(line for line in (gw_stats_line, season_stats_line) if line)
    title_attr = f' title="{stats_tooltip}"' if stats_tooltip else ""
    return (
        f'<div class="fpl-player-card"{title_attr} style="position: relative; background: rgba(255,255,255,0.94); '
        f'border-radius: 8px; padding: 6px 8px; min-width: 92px; max-width: 118px; text-align: center; '
        f'box-shadow: 0 2px 6px rgba(0,0,0,0.25); font-family: sans-serif;">'
        f'{badge_html}'
        f'{dreamteam_html}'
        f'{scout_html}'
        f'{penalty_taker_html}'
        f'{captain_html}'
        f'{img_html}'
        f'<div style="font-weight: 600; font-size: 12px; color: #1a1a1a; line-height: 1.2; '
        f'white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row["name"]}</div>'
        f'<div style="font-size: 10.5px; color: #555; margin-top: 2px;">£{row["cost"]:.1f}m</div>'
        f'<div style="font-size: 11px; color: #0a6b2f; font-weight: 700; margin-top: 1px;">'
        f'{points_label}</div>'
        f'{ppg_html}'
        f'{fixtures_html}'
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


def inject_shared_css() -> None:
    """Scoped visual styling only -- no logic here. Colors match the pitch
    view's own green (see render_pitch) so the two don't clash, plus FPL's
    own purple as an accent (their real brand color, not invented). Kept to
    CSS only, no custom components, so it degrades gracefully if Streamlit's
    internal class names shift in a future version -- worst case it looks
    like plain Streamlit.

    Called once per page (each page calls this itself, right after its own
    st.set_page_config) rather than once globally, since st.set_page_config
    must be the first Streamlit call on each page and this must run after it.

    IMPORTANT: does NOT hardcode .stApp's background -- an earlier version of
    this forced a dark background unconditionally, which broke the experience
    for anyone using Streamlit's light theme (metric-card gradients tuned for
    dark also looked washed out on light). Every color below is deliberately
    translucent (rgba with a low alpha) or white-text-on-a-solid-gradient
    (app-hero, which is a fixed-color element by design, same as the pitch
    view), so it reads correctly against BOTH themes rather than assuming one."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Sora:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: "Manrope", "Segoe UI", Roboto, sans-serif; }
    h1, h2, h3, h4 { font-family: "Sora", "Segoe UI", Roboto, sans-serif; letter-spacing: -0.01em; }

    /* ---- Metric cards: glassy, gradient-bordered, lift on hover ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(42,150,80,0.16), rgba(90,60,180,0.10));
        border: 1px solid rgba(90,60,180,0.35);
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.10);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(90,60,180,0.20);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.75; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    div[data-testid="stMetricValue"] { font-family: "Sora", sans-serif; font-weight: 800; }

    /* ---- Tabs: pill-style, colored active state ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 8px 20px;
        font-weight: 600;
        background: rgba(127,127,127,0.08);
        border: 1px solid transparent;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(90,60,180,0.12);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2a9650, #5a3cb4) !important;
        border-color: transparent !important;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    /* ---- Hero banner ---- */
    .app-hero {
        background: linear-gradient(120deg, #1f7a3f 0%, #2a9650 45%, #5a3cb4 130%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(42,150,80,0.25);
        position: relative;
        overflow: hidden;
    }
    .app-hero::after {
        content: "";
        position: absolute;
        top: -40%; right: -10%;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.16), transparent 70%);
        pointer-events: none;
    }
    .app-hero h1 {
        color: white;
        margin: 0 0 8px 0;
        font-size: 2.1rem;
        font-weight: 800;
    }
    .app-hero p {
        color: rgba(255,255,255,0.92);
        margin: 0;
        font-size: 0.98rem;
        max-width: 640px;
    }

    /* ---- Section cards / expanders ---- */
    .section-card {
        background: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.15);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: border-color 0.15s ease;
    }
    .section-card:hover { border-color: rgba(90,60,180,0.35); }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(127,127,127,0.18);
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
    }

    /* ---- Buttons ---- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px;
        font-weight: 700;
        border: none;
        background: linear-gradient(120deg, #2a9650, #5a3cb4);
        color: white;
        padding: 0.5em 1.4em;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(90,60,180,0.35);
        color: white;
    }

    /* ---- Dataframes: rounder, less spreadsheet-y ---- */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(127,127,127,0.15);
    }

    /* ---- Player cards (see _player_card_html): entrance + hover lift ---- */
    @keyframes fpl-card-in {
        from { opacity: 0; transform: translateY(10px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .fpl-player-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        animation: fpl-card-in 0.35s ease both;
    }
    .fpl-player-card:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 12px 24px rgba(90,60,180,0.30);
        z-index: 2;
    }
    /* stagger up to the first 20 cards on a row so they don't all pop at once */
    div[data-testid="column"]:nth-child(1) .fpl-player-card { animation-delay: 0.00s; }
    div[data-testid="column"]:nth-child(2) .fpl-player-card { animation-delay: 0.03s; }
    div[data-testid="column"]:nth-child(3) .fpl-player-card { animation-delay: 0.06s; }
    div[data-testid="column"]:nth-child(4) .fpl-player-card { animation-delay: 0.09s; }
    div[data-testid="column"]:nth-child(5) .fpl-player-card { animation-delay: 0.12s; }
    div[data-testid="column"]:nth-child(6) .fpl-player-card { animation-delay: 0.15s; }
    div[data-testid="column"]:nth-child(7) .fpl-player-card { animation-delay: 0.18s; }
    div[data-testid="column"]:nth-child(8) .fpl-player-card { animation-delay: 0.21s; }
    div[data-testid="column"]:nth-child(9) .fpl-player-card { animation-delay: 0.24s; }
    div[data-testid="column"]:nth-child(10) .fpl-player-card { animation-delay: 0.27s; }
    div[data-testid="column"]:nth-child(11) .fpl-player-card { animation-delay: 0.30s; }

    /* ---- Hero: subtle shimmer sweep, not distracting ---- */
    @keyframes fpl-hero-shimmer {
        0% { transform: translateX(-30%) translateY(-30%) rotate(0deg); }
        100% { transform: translateX(-30%) translateY(-30%) rotate(360deg); }
    }
    .app-hero::after { animation: fpl-hero-shimmer 18s linear infinite; }

    /* ---- Whole-page fade-in on load ---- */
    @keyframes fpl-fade-in { from { opacity: 0; } to { opacity: 1; } }
    section[data-testid="stMain"] { animation: fpl-fade-in 0.4s ease both; }

    /* ---- Tabs: smooth underline-free active swap already handled above;
       add a quick scale pulse when a tab becomes active ---- */
    @keyframes fpl-tab-pop { 0% { transform: scale(0.94); } 100% { transform: scale(1); } }
    .stTabs [aria-selected="true"] { animation: fpl-tab-pop 0.2s ease both; }

    /* ---- Metrics: gentle rise-in ---- */
    @keyframes fpl-metric-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    div[data-testid="stMetric"] { animation: fpl-metric-in 0.3s ease both; }

    /* ---- Buttons: press feedback ---- */
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(0) scale(0.97);
    }

    /* ---- Sidebar: subtle depth so it doesn't read as flat admin nav ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(127,127,127,0.15);
    }

    /* Respect users who've asked for reduced motion */
    @media (prefers-reduced-motion: reduce) {
        .fpl-player-card, .app-hero::after, section[data-testid="stMain"],
        .stTabs [aria-selected="true"], div[data-testid="stMetric"] {
            animation: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Shared sidebar content -- identical on every page, so it's here rather
    than duplicated in app.py and pages/1_Historical_and_Model.py."""
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
