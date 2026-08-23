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
    load_features, _get_xp_model, gw_pool, season_pool, preseason_pool, scout_picks_pool,
    load_model_metrics, load_manager_history,
    render_pitch, inject_shared_css, render_sidebar,
    _load_saved_manual_squad_ids, _save_manual_squad_ids,
    optimize_squad, select_starting_xi, POSITION_REQUIREMENTS, DEFAULT_BUDGET,
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
<p>Demo squad-building modes against past seasons, the trained model's real validation
metrics, and this manager's season-by-season history — proof of how each tool works,
not live current-gameweek advice (see the Home page for that).</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_squad, tab_model, tab_past = st.tabs(
    ["📋 Overview", "🧠 Squad Builder", "📈 Model Performance", "🏆 Past Seasons"]
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
