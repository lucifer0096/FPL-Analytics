"""FPL Analytics dashboard.

The 2026-27 season hasn't started yet (first fixture 21 Aug 2026), so there's
no live gameweek data to predict from -- every tab below uses historical
seasons as a working demonstration of what each tool does, clearly labeled as
such. Once the collector has real 2026-27 gameweek results, the squad/transfer/
chip tools can be pointed at live predictions instead (see this repo's README,
Future Improvements).
"""

import os
import sys

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "model"))

from optimizer import optimize_squad, optimize_transfers, load_latest_prices, POSITION_REQUIREMENTS, DEFAULT_BUDGET
from chips import suggest_bench_boost, suggest_triple_captain, suggest_free_hit_or_wildcard

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES_PATH = os.path.join(PROJECT_DIR, "data", "processed", "features.parquet")
HISTORICAL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "historical_gw.parquet")
MANAGER_ENTRY_ID = 1132016

SEASON_ORDER = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
VALIDATION_SEASON = "2024-25"
FINAL_HOLDOUT_SEASON = "2025-26"


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def gw_pool(df: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Build an optimizer-ready player pool for one (season, GW), using each
    player's rolling-5 average as a predicted_points stand-in -- the same
    approach used throughout this project's test scripts, since no live
    per-gameweek model predictions exist yet for a season that hasn't
    started."""
    sub = df[(df["season"] == season) & (df["GW"] == gw)].copy()
    sub["player_id"] = sub["player_code"]
    sub["cost"] = sub["value"] / 10.0
    sub["predicted_points"] = sub["total_points_avg_last_5"].fillna(0).clip(lower=0)
    return sub.drop_duplicates(subset="player_id")[
        ["player_id", "name", "position", "team", "cost", "predicted_points"]
    ]


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

    col_a, col_b = st.columns(2)
    season = col_a.selectbox("Season", SEASON_ORDER, index=SEASON_ORDER.index("2025-26"), key="squad_season")
    df = load_features()
    max_gw = int(df[df["season"] == season]["GW"].max())
    gw = col_b.slider("Gameweek", 1, max_gw, min(20, max_gw), key="squad_gw")

    pool = gw_pool(df, season, gw)
    st.caption(f"Player pool: {len(pool)} players, using each player's rolling-5 average as a predicted-points stand-in.")

    if st.button("Build optimal squad", key="build_squad_btn"):
        with st.spinner("Solving..."):
            squad = optimize_squad(pool)
        st.session_state["built_squad"] = squad
        st.session_state["built_squad_season_gw"] = (season, gw)

    if "built_squad" in st.session_state:
        squad = st.session_state["built_squad"]
        st.success(
            f"Squad: £{squad['cost'].sum():.1f}m / £{DEFAULT_BUDGET}m · "
            f"{squad['predicted_points'].sum():.1f} total predicted points"
        )
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
        built_season, built_gw = st.session_state["built_squad_season_gw"]
        next_gw = built_gw + 1
        df = load_features()
        max_gw_for_season = int(df[df["season"] == built_season]["GW"].max())

        if next_gw > max_gw_for_season:
            st.warning(f"GW{built_gw} is the last available gameweek in {built_season} — pick an earlier gameweek in the Squad Builder tab to leave room for a following gameweek.")
        else:
            free_transfers = st.slider("Free transfers available", 1, 5, 1, key="ft_slider")

            if st.button("Find best transfer(s)", key="find_transfers_btn"):
                current_squad = st.session_state["built_squad"]
                next_pool = gw_pool(df, built_season, next_gw)

                common_ids = set(current_squad["player_id"]) & set(next_pool["player_id"])
                squad_ids = [pid for pid in current_squad["player_id"] if pid in common_ids]
                dropped = len(current_squad) - len(squad_ids)

                if dropped:
                    st.caption(f"Note: {dropped} squad player(s) not present in GW{next_gw}'s pool, excluded from this check.")

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
                    st.error("Too many squad players missing from the next gameweek's pool to run this check.")

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
