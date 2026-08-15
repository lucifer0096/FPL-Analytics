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
    file is present."""
    sub = df[(df["season"] == season) & (df["GW"] == gw)].copy()
    sub["player_id"] = sub["player_code"]
    sub["cost"] = sub["value"] / 10.0

    model = _get_xp_model()
    if model is not None and set(FEATURE_COLUMNS).issubset(sub.columns):
        sub["predicted_points"] = predict_points(sub, model)
    else:
        sub["predicted_points"] = sub["total_points_avg_last_5"].fillna(0).clip(lower=0)

    return sub.drop_duplicates(subset="player_id")[
        ["player_id", "name", "position", "team", "cost", "predicted_points"]
    ]


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

    return live[["player_id", "name", "position", "team", "cost", "predicted_points"]]


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
    return (
        f'<div style="position: relative; background: rgba(255,255,255,0.94); border-radius: 8px; '
        f'padding: 6px 8px; min-width: 92px; max-width: 118px; text-align: center; '
        f'box-shadow: 0 2px 6px rgba(0,0,0,0.25); font-family: sans-serif;">'
        f'{badge_html}'
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


st.title("⚽ FPL Analytics")
st.caption(
    "Expected-points model, squad optimizer, and chip advisor for Fantasy Premier League. "
    "The 2026-27 season hasn't started yet, so every tool below runs against historical "
    "seasons as a working demo — this becomes a live tool once real gameweeks exist."
)

tab_overview, tab_squad, tab_transfers, tab_chips, tab_model, tab_history = st.tabs(
    ["Overview", "Squad Builder", "Transfers", "Chip Advisor", "Model Performance", "Manager History"]
)

# =============================================================================
# OVERVIEW
# =============================================================================
with tab_overview:
    st.header("Project overview")

    col1, col2, col3, col4 = st.columns(4)
    df = load_features()
    col1.metric("Seasons tracked", f"{df['season'].nunique()}")
    col2.metric("Player-gameweek rows", f"{len(df):,}")
    col3.metric("Model validation MAE", "0.986")
    col4.metric("Squad rules encoded", "8")

    st.markdown("""
    **Pipeline:** live FPL API collector → 10 seasons of unified historical data →
    engineered features (rolling form, team form, fixture difficulty, new-player
    baseline) → trained LightGBM model → squad/transfer optimizer (PuLP integer
    programming) → chip-timing advisor.

    All of FPL's real rules used here (squad composition, budget, transfer costs,
    the 50% sell-fee, chip-per-half structure) were verified directly against the
    live API's `bootstrap-static` `game_settings` and `chips` fields — see the
    project README for the full rules table and how each was confirmed.
    """)

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
        ["Historical gameweek", "Team of the Season", "2026-27 pre-season (live prices)", "My squad (enter manually)"],
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
        st.caption(
            f"Player pool: {len(pool)} players who appeared at least once across {window_desc}, at the window's "
            f"final-gameweek price. This is a LOOK BACK at who actually produced the most REAL points in this "
            f"window, not a prediction — these gameweeks are already complete, so there's nothing to predict. "
            f"Ranked by total points scored, not a per-game rate — a great rate over a handful of games can't "
            f"outrank someone who played most of the window and produced far more for a real squad. "
            f"Points-per-game (FPL's own metric: total points ÷ appearances, shown once a player has a few games "
            f"to make it trustworthy) is shown on each card as context, not as what drives selection. "
            f"No budget cap here — this is the best XI the window actually produced, not a squad you could have "
            f"afforded on day one."
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

    else:  # My squad (enter manually)
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
            st.success(f"Squad: £{squad['cost'].sum():.1f}m · {squad['predicted_points'].sum():.0f} total points scored ({window_label})")
            badge_label = f"MVP — most points scored in {window_label}"
        else:
            st.success(
                f"Squad: £{squad['cost'].sum():.1f}m / £{DEFAULT_BUDGET}m · "
                f"{squad['predicted_points'].sum():.1f} total predicted points"
            )
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
            free_transfers = st.slider("Free transfers available", 1, 5, 1, key="ft_slider")
            st.caption(f"Checking against {pool_label}.")

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
                            free_transfers=free_transfers,
                        )

                    if result["transfers_in"]:
                        out_names = next_pool[next_pool["player_id"].isin(result["transfers_out"])]["name"].tolist()
                        in_names = next_pool[next_pool["player_id"].isin(result["transfers_in"])]["name"].tolist()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Transfers suggested", len(result["transfers_in"]))
                        col2.metric("Hit cost", f"-{result['hit_cost']} pts")
                        col3.metric("Net points gain", f"{result['net_points_gain']:+.1f}")
                        st.write(f"**Out:** {', '.join(out_names)}")
                        st.write(f"**In:** {', '.join(in_names)}")
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

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Bench Boost")
                    for s in suggest_bench_boost(squad, future_points_by_gw, gw_range)[:3]:
                        st.write(f"**GW{s.gameweek}** — {s.detail}")
                with col2:
                    st.subheader("Triple Captain")
                    for s in suggest_triple_captain(squad, future_points_by_gw, gw_range)[:3]:
                        st.write(f"**GW{s.gameweek}** — {s.detail}")
                with col3:
                    st.subheader("Free Hit")
                    for s in suggest_free_hit_or_wildcard(squad, future_points_by_gw, optimal_points_by_gw, gw_range, chip="free_hit")[:3]:
                        st.write(f"**GW{s.gameweek}** — {s.detail}")

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
with tab_model:
    st.header("Model performance")
    st.caption(f"Validated on {VALIDATION_SEASON} (chronological split — trained only on earlier seasons). {FINAL_HOLDOUT_SEASON} is held out entirely and untouched.")

    perf = pd.DataFrame({
        "Model": ["Single-stage", "Two-stage", "Naive baseline (rolling-5 avg)", "FPL's own xP*"],
        "MAE": [0.986, 0.984, 1.052, 0.904],
        "RMSE": [1.914, 1.914, 2.069, 1.757],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption(
        "*FPL's own xP carries a caveat from the data source's maintainer: it may contain "
        "post-match information for some gameweeks (scraper runs after each gameweek ends, "
        "FPL's update cadence for the underlying field is undocumented). Treated as an "
        "informative but not fully leak-free comparison — see the README's Model Training section."
    )

    st.subheader("Where the model does well vs. FPL's xP")
    played_perf = pd.DataFrame({
        "Model": ["Single-stage (played rows only)", "Naive baseline (played only)", "FPL's own xP (played only)"],
        "MAE": [1.832, 2.053, 1.759],
    })
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

    manager_data = pd.DataFrame({
        "season": ["2016/17", "2017/18", "2018/19", "2019/20", "2020/21",
                   "2021/22", "2022/23", "2023/24", "2024/25", "2025/26"],
        "points": [2053, 2049, 2094, 1973, 2206, 2130, 2312, 2284, 2227, 2102],
        "rank": [386166, 809108, 898121, 2207538, 873796, 1688770, 1404045, 1189768, 2461833, 2240945],
    })
    manager_data["top_pct"] = [9, 14, 14, 29, 11, 18, 12, 11, 22, 17]

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
        "2025/26 figures may be provisional if pulled before that season's final gameweek "
        "was confirmed finished. Full interactive version: "
        "[my-fpl-history.html](https://lucifer0096.github.io/FPL-Analytics/my-fpl-history.html)."
    )
