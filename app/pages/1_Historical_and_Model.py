"""Historical & Model page -- everything that isn't about the manager's real,
current-season team: demo Squad Builder modes (Historical gameweek, Team of
the Season, pre-season, Scout Picks, manual entry), Model Performance, the
project Overview, and past-season history. Split out of the old single-page
app so Home (app.py) can stay focused on "what should I do THIS gameweek"
without the historical/methodology material crowding it out -- see this
project's README, Dashboard section, for the reasoning."""

import os

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import (
    SEASON_ORDER, PROJECT_DIR, MANAGER_ENTRY_ID,
    load_features, season_pool, season_insights,
    load_model_metrics, load_manager_history,
    render_pitch, inject_shared_css, render_sidebar,
    optimize_squad, DEFAULT_BUDGET,
)

st.set_page_config(
    page_title="FPL Analytics — Historical & Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_shared_css()
render_sidebar()

st.markdown("""
<div class="app-hero">
<h1>📊 Historical & Model</h1>
<p>Real analysis of completed seasons, the trained model's validation metrics, and this
manager's season-by-season history — not live current-gameweek advice (see the Home page
for that; squad-BUILDING tools for an already-played season have no real purpose, so
this page is about what actually happened, not what an optimizer would have picked).</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_insights, tab_tots, tab_model, tab_past = st.tabs(
    ["📋 Overview", "🔍 Season Insights", "🏆 Team of the Season", "📈 Model Performance", "🏆 Past Seasons"]
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

    st.subheader("Why this page uses historical data, not live predictions")
    st.info(
        "This page demonstrates each tool against real historical seasons — a genuine "
        "proof of how the optimizer, model, and chip advisor behave, verified against "
        "known outcomes. For live, current-gameweek advice on your actual squad, see "
        "the **Home** page instead."
    )

# =============================================================================
# SEASON INSIGHTS
# =============================================================================
with tab_insights:
    st.header("Season insights")
    st.caption(
        "Real, verifiable facts about a completed (or completing) season — not a squad-"
        "building tool. Every number here is an actual historical total, not a prediction."
    )

    df = load_features()
    insight_season = st.selectbox("Season", SEASON_ORDER, index=SEASON_ORDER.index("2025-26"), key="insights_season")
    insights = season_insights(df, insight_season)

    st.subheader("Top scorers")
    st.dataframe(
        insights["top_scorers"][["name", "position", "team", "total_points", "cost"]]
        .rename(columns={"name": "Player", "position": "Pos", "team": "Team", "total_points": "Points", "cost": "Cost (£m)"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Best value (points per £m spent)")
    st.caption("Players with at least 450 minutes played — too small a sample otherwise (one big haul off the bench would dominate this ranking).")
    st.dataframe(
        insights["best_value"][["name", "position", "team", "total_points", "cost", "pts_per_million"]]
        .rename(columns={"name": "Player", "position": "Pos", "team": "Team", "total_points": "Points", "cost": "Cost (£m)", "pts_per_million": "Pts / £m"})
        .round({"Pts / £m": 1}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Top scorer by position")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    for col, pos in zip([pcol1, pcol2, pcol3, pcol4], ["GK", "DEF", "MID", "FWD"]):
        leaders = insights["position_leaders"][pos]
        with col:
            st.markdown(f"**{pos}**")
            for _, row in leaders.iterrows():
                st.caption(f"{row['name']} — {row['total_points']} pts")

    st.subheader("Biggest price risers")
    st.caption(
        f"Real price movement WITHIN {insight_season} only — each player's own first and last "
        f"gameweek price that season (not their current 2026-27 price), never compared across "
        f"seasons. The market's own signal of who performed above expectations that year."
    )
    st.dataframe(
        insights["biggest_price_risers"][["name", "position", "team", "start_cost", "cost", "price_rise"]]
        .rename(columns={
            "name": "Player", "position": "Pos", "team": "Team",
            "start_cost": f"Start of {insight_season} (£m)", "cost": f"End of {insight_season} (£m)",
            "price_rise": "Change (£m)",
        })
        .head(5),
        use_container_width=True, hide_index=True,
    )

# =============================================================================
# TEAM OF THE SEASON
# =============================================================================
with tab_tots:
    st.header("Team of the Season")
    st.caption(
        "A LOOK BACK at an already-completed season (or a narrower gameweek window — "
        "Team of the Week — via a slider), not a prediction: the optimal squad ranked by "
        "each player's real total points scored, at their final-gameweek price."
    )

    df = load_features()
    season = st.selectbox("Season", SEASON_ORDER, index=SEASON_ORDER.index("2025-26"), key="tots_season")
    max_gw = int(df[df["season"] == season]["GW"].max())
    gw_start, gw_end = st.slider(
        "Gameweek range (full season by default — narrow it to build a Team of the Week instead)",
        1, max_gw, (1, max_gw), key="tots_gw_range",
    )
    is_full_season = (gw_start, gw_end) == (1, max_gw)
    pool = season_pool(df, season, gw_start, gw_end)
    window_desc = f"the full {season} season" if is_full_season else f"GW{gw_start}–GW{gw_end} of {season}"
    st.caption(f"Player pool: {len(pool)} players who appeared at least once across {window_desc}. Ranked by total points scored, no budget cap.")
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
        st.session_state["tots_squad"] = squad
        st.session_state["tots_window_label"] = "this season" if is_full_season else f"GW{gw_start}–{gw_end}"

    if "tots_squad" in st.session_state:
        squad = st.session_state["tots_squad"]
        window_label = st.session_state.get("tots_window_label", "this window")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Squad cost", f"£{squad['cost'].sum():.1f}m")
        sc2.metric("Points scored", f"{squad['predicted_points'].sum():.0f}")
        sc3.metric("Window", window_label)
        render_pitch(squad, top_player_badge=f"MVP — most points scored in {window_label}")

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
# PAST SEASONS
# =============================================================================
with tab_past:
    st.header("Past seasons")
    st.caption(f"Season-by-season points and overall rank (entry {MANAGER_ENTRY_ID}).")

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
