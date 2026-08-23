"""FPL Analytics dashboard -- Home page.

Live, current-season focused: this manager's real squad and points, league
standings, and transfer/chip advice for the actual team. Demo modes against
past seasons, the model's validation metrics, and season history live on the
separate "Historical & Model" page (see app/pages/) instead -- kept off this
page so it stays about "what should I do this gameweek," not a crowded
seven-tab methodology tour.
"""

import os

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import (
    PROJECT_DIR, MANAGER_ENTRY_ID,
    load_features, gw_pool, preseason_pool,
    load_manager_name, load_current_season_progress,
    load_current_squad_picks, build_live_squad_df, load_joined_leagues,
    render_pitch, inject_shared_css, render_sidebar,
    optimize_transfers, suggest_bench_boost, suggest_triple_captain, suggest_free_hit_or_wildcard,
    optimize_squad, POSITION_REQUIREMENTS,
)

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_shared_css()
render_sidebar()

manager_name = load_manager_name(MANAGER_ENTRY_ID)
st.markdown(f"""
<div class="app-hero">
<h1>⚽ FPL Analytics — {manager_name}</h1>
<p>Your real squad, points, and league standings for the season in progress — plus
transfer and chip advice for your actual team, not just an optimizer demo. See the
<b>Historical &amp; Model</b> page (sidebar) for past-season proof and methodology.</p>
</div>
""", unsafe_allow_html=True)


def _collected_gws() -> list:
    """Which gameweeks this manager's real picks have actually been
    collected for -- checks disk directly rather than re-deriving "current
    gameweek" from a live bootstrap call on every page load."""
    picks_dir = os.path.join(PROJECT_DIR, "data", "raw", "2026-27", "entry", str(MANAGER_ENTRY_ID), "picks")
    if os.path.isdir(picks_dir):
        local = sorted(
            int(f[2:-5]) for f in os.listdir(picks_dir) if f.startswith("gw") and f.endswith(".json")
        )
        if local:
            return local
    # No local data/raw/ (deploy environment) -- fall back to whatever
    # load_current_squad_picks can find via its own committed-fallback logic,
    # by checking gameweeks 1 upward until one comes back empty. The
    # fallback bundle only ever holds ONE gameweek (the latest collected --
    # see refresh_dashboard_fallbacks.py), so this converges in at most a
    # couple of tries in practice, not an unbounded scan.
    found = []
    for gw in range(1, 39):
        if load_current_squad_picks(MANAGER_ENTRY_ID, gw) is not None:
            found.append(gw)
    return found


tab_squad, tab_transfers, tab_chips, tab_leagues = st.tabs(
    ["🧠 My Squad", "🔁 Transfers", "🃏 Chip Advisor", "🏅 League Tracker"]
)

# =============================================================================
# MY SQUAD (real, live)
# =============================================================================
with tab_squad:
    st.header("My current squad")

    collected_gws = _collected_gws()

    if not collected_gws:
        st.info(
            "No picks collected yet for this manager — this section fills in automatically "
            "once the collector captures a gameweek's picks (available as soon as that "
            "gameweek's deadline passes, even before final bonus points are added). Run "
            "`python src/collector/snapshot.py` with `FPL_ENTRY_ID` set to populate this page."
        )
    else:
        latest_gw = collected_gws[-1]
        picks_data = load_current_squad_picks(MANAGER_ENTRY_ID, latest_gw)
        squad_df = build_live_squad_df(picks_data, latest_gw)
        entry_hist = picks_data["entry_history"]

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric(f"GW{latest_gw} points", entry_hist["points"])
        pcol2.metric("Total points", entry_hist["total_points"])
        pcol3.metric("Overall rank", f"{entry_hist['overall_rank']:,}")
        pcol4.metric("Points on bench", entry_hist["points_on_bench"])

        captain_row = squad_df[squad_df["is_captain"]]
        if not captain_row.empty:
            st.caption(f"Captain: **{captain_row.iloc[0]['name']}** (points doubled below)")

        render_pitch(squad_df)
        st.caption(
            f"This is your REAL GW{latest_gw} squad and points, read from FPL's own "
            f"entry/{MANAGER_ENTRY_ID}/event/{latest_gw}/picks — not the optimizer. "
            f"Points shown may still be provisional if bonus points haven't been finalized yet."
        )

        # Store the live squad in the SAME session_state keys the historical
        # Squad Builder page uses, so Transfers/Chip Advisor below (and on
        # the other page) can operate on it without a separate code path.
        # "live_squad" is a distinct sentinel from "completed_season"/None/
        # a (season, gw) tuple -- see optimize_transfers/chip usage below.
        st.session_state["built_squad"] = squad_df
        st.session_state["built_squad_season_gw"] = "live_squad"
        st.session_state["live_squad_gw"] = latest_gw

    st.divider()
    st.subheader("2026-27 progress")
    current_progress = load_current_season_progress(MANAGER_ENTRY_ID)
    if current_progress.empty:
        st.info(
            "No 2026-27 gameweeks captured yet — this section fills in automatically, "
            "gameweek by gameweek, once the collector captures real results."
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

# =============================================================================
# TRANSFERS (against the real live squad)
# =============================================================================
with tab_transfers:
    st.header("Transfer optimizer")
    st.caption(
        "Suggests the transfer(s) — if any — worth actually making from your real squad, "
        "checked against the live current player pool."
    )

    if "built_squad" not in st.session_state or st.session_state.get("built_squad_season_gw") != "live_squad":
        st.warning("Build your real squad in the **My Squad** tab first (requires at least one collected gameweek).")
    else:
        current_squad = st.session_state["built_squad"]
        df = load_features()
        try:
            next_pool = preseason_pool(df)
            pool_label = "the live current player pool (re-fetched, in case prices moved)"
        except FileNotFoundError as e:
            st.error(str(e))
            next_pool = None

        if next_pool is not None:
            unlimited = st.checkbox(
                "Playing Wildcard or Free Hit this gameweek — unlimited free transfers",
                key="unlimited_transfers_checkbox_home",
                help="Verified against the live API's game_settings: every transfer made while "
                     "Wildcard or Free Hit is active is free by chip definition, with no cap and "
                     "no -4pt hit ever applying. Check this instead of setting free transfers below.",
            )
            if not unlimited:
                free_transfers = st.slider("Free transfers available", 1, 5, 1, key="ft_slider_home")

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

            if st.button("Find best transfer(s)", key="find_transfers_btn_home"):
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
# CHIP ADVISOR (against the real live squad, historical seasons as the projection source)
# =============================================================================
with tab_chips:
    st.header("Chip-timing advisor")
    st.caption(
        "Ranks upcoming gameweeks for Bench Boost, Triple Captain, and Free Hit, "
        "given your real squad. Each FPL chip is usable once per season half — see "
        "the README for how that was verified against the live API."
    )

    if "built_squad" not in st.session_state or st.session_state.get("built_squad_season_gw") != "live_squad":
        st.warning("Build your real squad in the **My Squad** tab first (requires at least one collected gameweek).")
    else:
        st.info(
            "Chip timing needs a run of several **upcoming** gameweeks to project against, "
            "which doesn't exist yet this early in a live season — once the collector has "
            "captured enough 2026-27 gameweeks, this tool will project forward using real "
            "upcoming fixtures. For a demonstration of how this tool works against a completed "
            "run of gameweeks, see the **Historical & Model** page's Squad Builder → "
            "Historical gameweek mode, then this same Chip Advisor logic there."
        )

# =============================================================================
# LEAGUE TRACKER
# =============================================================================
with tab_leagues:
    st.header("League tracker")
    st.caption(
        "Real standings for every PRIVATE classic (points-based) mini-league this manager "
        "has joined by code — excludes FPL's own auto-generated global/region/club leagues, "
        "which aren't leagues you actually 'joined.'"
    )

    leagues = load_joined_leagues(MANAGER_ENTRY_ID)

    if not leagues:
        st.info(
            "No league standings collected yet — this fills in automatically once the "
            "collector runs with `FPL_ENTRY_ID` set (see README's 'Running the Collector'). "
            "Run `python src/collector/snapshot.py` to populate this tab."
        )
    else:
        league_names = [l["league"]["name"] for l in leagues]
        selected_name = st.selectbox("League", league_names, key="league_select")
        league = next(l for l in leagues if l["league"]["name"] == selected_name)

        results = league["standings"]["results"]
        standings_df = pd.DataFrame([
            {
                "rank": r["rank"],
                "manager": r["player_name"],
                "team": r["entry_name"],
                "gw_points": r["event_total"],
                "total_points": r["total"],
                "is_you": r["entry"] == MANAGER_ENTRY_ID,
            }
            for r in results
        ]).sort_values("rank")

        my_row = standings_df[standings_df["is_you"]]
        if not my_row.empty:
            lcol1, lcol2, lcol3 = st.columns(3)
            lcol1.metric("Your rank", f"{int(my_row.iloc[0]['rank'])} / {len(standings_df)}")
            lcol2.metric("Your total points", int(my_row.iloc[0]["total_points"]))
            lcol3.metric("Your GW points", int(my_row.iloc[0]["gw_points"]))

        display_df = standings_df.drop(columns=["is_you"]).rename(columns={
            "rank": "Rank", "manager": "Manager", "team": "Team Name",
            "gw_points": "GW Points", "total_points": "Total Points",
        })
        # Highlight this manager's own row so it's easy to find in a longer
        # league table -- st.dataframe doesn't support row-conditional
        # styling directly via column config, so pandas Styler is used here
        # specifically (the one place in this app a DataFrame is styled
        # rather than rendered as plain HTML, since this needs real per-row
        # conditional logic a static CSS class can't express).
        def _highlight_own_row(row):
            is_you = standings_df.loc[row.name, "is_you"]
            return ["background-color: rgba(42,150,80,0.25)" if is_you else "" for _ in row]

        st.dataframe(
            display_df.style.apply(_highlight_own_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"League: **{league['league']['name']}** · {len(standings_df)} managers · "
            f"read live from FPL's own `leagues-classic/{league['league']['id']}/standings` "
            f"endpoint via the collector, not hardcoded."
        )
