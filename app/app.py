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
    load_features, preseason_pool,
    load_manager_name, load_current_season_progress, calculate_free_transfers,
    load_current_squad_picks, build_live_squad_df, load_joined_leagues, live_price_changes, likely_price_movers,
    tonight_price_projections,
    differential_finder, league_wide_status_flags, premier_league_table, premier_league_table_with_movement,
    season_leaderboards, team_insights, player_season_stats,
    team_upcoming_fixtures, average_fixture_difficulty, suggest_captain, is_gameweek_live,
    ep_next_player_pool, _current_season_label, explain_transfer_suggestion_debug,
    render_pitch, inject_shared_css, render_sidebar,
    optimize_transfers, optimize_squad, POSITION_REQUIREMENTS,
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
    picks_dir = os.path.join(PROJECT_DIR, "data", "raw", _current_season_label(), "entry", str(MANAGER_ENTRY_ID), "picks")
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


def _arrow_price_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Prefixes a numeric price-movement column with a real green ▲ / red ▼
    arrow matching the row's own actual sign -- a purely visual aid (the
    real underlying number is unchanged) so a riser/faller table reads at
    a glance without checking the sign of every row individually. Positive
    values get ▲ (green), negative get ▼ (red), exactly zero gets neither
    (there's no real movement to point an arrow at)."""
    df = df.copy()

    def _format(value):
        if pd.isna(value) or value == 0:
            return f"{value}"
        arrow = "🟢▲" if value > 0 else "🔴▼"
        return f"{arrow} {value}"

    df[column] = df[column].map(_format)
    return df


tab_squad, tab_transfers, tab_chips, tab_leagues, tab_prices, tab_table, tab_insights = st.tabs(
    ["🧠 My Squad", "🔁 Transfers", "🃏 Chip Advisor", "🏅 League Tracker", "💰 Price Changes", "📊 PL Table", "🔥 Season Insights"]
)

# =============================================================================
# MY SQUAD (real, live)
# =============================================================================
# Wrapped in an st.fragment so it can auto-rerun on its own timer without
# re-running the whole page (Transfers/Price Changes/etc. below don't need
# to refetch every 60s just because My Squad does). run_every is set to a
# real number (not None) ONLY while a gameweek is genuinely live (see
# is_gameweek_live() -- FPL's own is_current AND NOT finished) -- polling
# FPL's live API every 60s the rest of the week would be pure waste, since
# nothing here actually changes minute to minute outside a live gameweek.
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_my_squad_tab():
    st.header("My current squad")
    if is_gameweek_live():
        st.caption("🔴 Live gameweek in progress — this section auto-refreshes every 60 seconds.")

    collected_gws = _collected_gws()
    live_gw_points = None  # set below when a real squad's live points are computed, reused by the progress section further down

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

        # This gameweek's points are summed directly from the same real,
        # live per-player points shown on the pitch below (squad_df's
        # predicted_points, already multiplier-applied for captaincy) rather
        # than trusted from entry_history["points"] -- verified directly
        # that FPL's own entry-summary field can lag behind its OWN
        # per-player live feed by a few points for a short window during a
        # gameweek (e.g. summary said 15 while the live per-player feed's
        # own numbers already summed to 18) -- summing the same numbers
        # already on screen keeps the headline metric consistent with the
        # cards underneath it, and matches the real live per-player source
        # sooner than the summary field catches up.
        gw_points = int(squad_df["predicted_points"].sum())
        live_gw_points = (latest_gw, gw_points)

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric(f"GW{latest_gw} points", gw_points)
        pcol2.metric("Total points", entry_hist["total_points"] + (gw_points - entry_hist["points"]))
        pcol3.metric("Overall rank", f"{entry_hist['overall_rank']:,}")
        pcol4.metric("Points on bench", entry_hist["points_on_bench"])

        captain_row = squad_df[squad_df["is_captain"]]
        if not captain_row.empty:
            st.caption(f"Captain: **{captain_row.iloc[0]['name']}** (points doubled below)")

        # Real captain suggestion, using FPL's OWN ep_next field for every
        # starter in THIS squad -- links directly into the squad just built
        # above rather than a separate tool. Only shown as a distinct
        # suggestion when it actually differs from the real choice already
        # made, so agreement doesn't produce redundant noise.
        suggested_captain = suggest_captain(squad_df)
        if suggested_captain is not None and (
            captain_row.empty or suggested_captain["player_id"] != captain_row.iloc[0]["player_id"]
        ):
            st.info(
                f"💡 Suggested captain based on FPL's own real expected-points data: "
                f"**{suggested_captain['name']}** ({suggested_captain['ep_next']:.1f} expected pts) "
                f"— currently the highest-`ep_next` starter in your squad."
            )

        render_pitch(squad_df)
        st.caption(
            f"This is your REAL GW{latest_gw} squad and points, read from FPL's own "
            f"entry/{MANAGER_ENTRY_ID}/event/{latest_gw}/picks — not the optimizer. "
            f"Points shown may still be provisional if bonus points haven't been finalized yet."
        )
        st.caption(
            f"⚠️ If you've made a transfer for GW{latest_gw + 1} since this was collected, it "
            f"won't show here yet — checked directly against FPL's API: a gameweek's picks aren't "
            f"publicly visible until THAT gameweek's own deadline passes (no public endpoint "
            f"exposes a squad mid-transfer-window). This will update automatically to your real "
            f"GW{latest_gw + 1} squad once its deadline passes and the collector picks it up."
        )

        st.caption(
            "💡 Hover any player card above for their full real 2026-27 SEASON stats "
            "(minutes, goals, assists, DEFCON, bonus, xG/xA, ICT, and more) — every number is "
            "a running season total FPL itself tracks, not a single-gameweek figure."
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
        # The latest gameweek's points/total_points here can suffer the SAME
        # entry_history-lags-behind-its-own-live-feed staleness as the My
        # Squad metric above (verified directly: FPL's summary said 15 while
        # the real per-player live feed already summed to 18) -- corrected
        # using the same real gw_points already computed above from
        # squad_df, when this row is for that same gameweek, so the chart/
        # metrics below never contradict the pitch view above them.
        if live_gw_points is not None and current_progress["gw"].iloc[-1] == live_gw_points[0]:
            corrected_points = live_gw_points[1]
            stale_points = current_progress["points"].iloc[-1]
            current_progress.loc[current_progress.index[-1], "points"] = corrected_points
            current_progress.loc[current_progress.index[-1], "total_points"] += (corrected_points - stale_points)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Overall rank by gameweek (lower is better)")
            st.line_chart(current_progress.set_index("gw")["overall_rank"])
        with col2:
            st.caption("Your points vs. the real average across ALL managers, by gameweek")
            st.line_chart(current_progress.set_index("gw")[["points", "average_entry_score"]].rename(
                columns={"points": "Your points", "average_entry_score": "Average (all managers)"}
            ))
        latest = current_progress.iloc[-1]
        total_bench_points = current_progress["points_on_bench"].sum()
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Total points", f"{latest['total_points']:.0f}")
        mcol2.metric("Overall rank", f"{latest['overall_rank']:,.0f}")
        if pd.notna(latest["overall_rank_percentage"]):
            mcol3.metric(
                "Top %", f"{latest['overall_rank_percentage']:.0f}%",
                help="FPL's own real 'you're in the top X%' figure for your current overall "
                     "rank -- not derived here, taken directly from their own reported number.",
            )
        mcol4.metric("Bank", f"£{latest['bank']:.1f}m")
        mcol5.metric(
            "Points left on bench", f"{total_bench_points:.0f}",
            help="Real, running total of points_on_bench (FPL's own field) across every "
                 "gameweek so far this season — points your bench scored that never counted "
                 "toward your total, since only your starting XI's points count.",
        )
        st.dataframe(
            current_progress.rename(columns={
                "gw": "GW", "points": "GW points", "total_points": "Total points",
                "overall_rank": "Overall rank", "bank": "Bank (£m)", "value": "Squad value (£m)",
                "points_on_bench": "Bench points", "overall_rank_percentage": "Top %",
                "average_entry_score": "Avg. (all managers)",
            }).drop(columns=["event_transfers", "event_transfers_cost"]),
            use_container_width=True,
            hide_index=True,
        )


with tab_squad:
    _render_my_squad_tab()

# =============================================================================
# TRANSFERS (against the real live squad)
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_transfers_tab():
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
            st.info(
                "⚠️ predicted_points here comes from last season's **closing 2025-26 form** "
                "(`preseason_pool()`), not real 2026-27 in-season data or the trained model's "
                "real fixture-difficulty/current-form features (see the Historical & Model "
                "page's Model Performance tab for how that model works on already-finished "
                "gameweeks). Treat suggested transfers as a rough signal, not a confident "
                "recommendation. **This is a standing limitation, not a temporary one** — "
                "wiring live 2026-27 gameweeks into the trained model's own feature pipeline "
                "(fixture_difficulty, rolling form, etc.) is a real data-engineering step that "
                "hasn't been built yet, so this pool won't silently improve just because more "
                "gameweeks pass; it needs that pipeline work first."
            )
        except FileNotFoundError as e:
            st.error(str(e))
            next_pool = None

        if next_pool is not None:
            # Real, current injury/suspension/doubt status -- FPL's own
            # status/news/chance_of_playing_next_round fields (verified live
            # against bootstrap-static), not inferred from a points gap.
            # 'a' = available; anything else (i/injured, s/suspended,
            # d/doubtful, u/unavailable-left-club, n/not-in-squad) is a real,
            # named reason a squad member might be worth transferring out --
            # shown explicitly rather than left for a stale predicted-points
            # comparison to (maybe) stumble onto.
            STATUS_LABELS = {"i": "Injured", "s": "Suspended", "d": "Doubtful", "u": "Left club", "n": "Not in squad"}
            squad_status = next_pool[next_pool["player_id"].isin(current_squad["player_id"])]
            flagged = squad_status[squad_status["status"] != "a"]
            flagged_ids = set()
            if not flagged.empty:
                st.warning("⚠️ Squad members with a real availability concern:")
                for _, row in flagged.iterrows():
                    label = STATUS_LABELS.get(row["status"], row["status"])
                    chance = row["chance_of_playing_next_round"]
                    chance_str = f" ({chance}% chance of playing)" if chance is not None else ""
                    news = f" — {row['news']}" if row["news"] else ""
                    st.caption(f"**{row['name']}**: {label}{chance_str}{news}")
                    flagged_ids.add(row["player_id"])
                st.caption(
                    "These players are treated as effectively unavailable below (predicted points "
                    "zeroed for this check) so a genuine injury/suspension — not just a form dip — "
                    "is what actually drives a transfer-out suggestion, rather than relying on a "
                    "predicted-points gap to happen to catch it."
                )
                next_pool = next_pool.copy()
                next_pool.loc[next_pool["player_id"].isin(flagged_ids), "predicted_points"] = 0.0

            # Real upcoming fixture difficulty (FPL's own 1-5 rating, same
            # data now shown on every squad card's fixture strip -- see
            # build_live_squad_df/team_upcoming_fixtures) adjusts
            # predicted_points before optimizing: an easy run of fixtures is
            # a real, checkable reason a player might be worth bringing IN,
            # and a hard run a real reason to consider moving one OUT --
            # not just last season's closing form on its own, which has no
            # notion of who a player is actually about to play. Scaled
            # modestly (+/-15% at the extremes, difficulty 3 = neutral) so
            # fixtures nudge the ranking rather than dominate it outright --
            # form still matters more than a single average-difficulty number.
            fixtures_by_team = team_upcoming_fixtures(3)
            next_pool = next_pool.copy()

            def _fixture_multiplier(team):
                avg_difficulty = average_fixture_difficulty(team, fixtures_by_team)
                if avg_difficulty is None:
                    return 1.0  # no fixture data for this team -- don't adjust, not a reason to guess
                return 1.0 - (avg_difficulty - 3.0) * 0.075

            next_pool["predicted_points"] = next_pool["predicted_points"] * next_pool["team"].map(_fixture_multiplier)
            st.caption(
                "📅 Predicted points above are adjusted (±15% at the extremes) for each player's "
                "team's real upcoming fixture difficulty (FPL's own rating, next 3 gameweeks) — "
                "the same fixture strip shown on your squad cards in the My Squad tab."
            )

            unlimited = st.checkbox(
                "Playing Wildcard or Free Hit this gameweek — unlimited free transfers",
                key="unlimited_transfers_checkbox_home",
                help="Verified against the live API's game_settings: every transfer made while "
                     "Wildcard or Free Hit is active is free by chip definition, with no cap and "
                     "no -4pt hit ever applying. Check this instead of setting free transfers below.",
            )
            if not unlimited:
                real_free_transfers = calculate_free_transfers(MANAGER_ENTRY_ID)
                free_transfers = st.slider(
                    "Free transfers available", 1, 5, real_free_transfers, key="ft_slider_home",
                    help="Pre-filled from your REAL transfer history (event_transfers each gameweek, "
                         "banked per FPL's real rule: 1 per week, up to 5 max) — not a guess. Override "
                         "if you know it's wrong (e.g. a chip changed the normal accounting).",
                )
                st.caption(f"Your real banked free transfers, computed from transfer history: **{real_free_transfers}**.")

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
                with st.expander("💎 Differential picks (low-owned, real upside)"):
                    st.caption(
                        "Real 'differential' candidates: owned by ≤10% of managers (FPL's own "
                        "`selected_by_percent`) with meaningful real upside (FPL's own `ep_next`, "
                        "the same field the captain suggestion and Chip Advisor use) — a genuine "
                        "rank-gaining edge if they perform, since few other managers have them."
                    )
                    diffs = differential_finder()
                    if diffs.empty:
                        st.caption("No qualifying differentials right now.")
                    else:
                        display_diffs = diffs.copy()
                        display_diffs["is_penalty_taker"] = display_diffs["is_penalty_taker"].map({True: "✓", False: ""})
                        st.dataframe(
                            display_diffs.rename(columns={
                                "name": "Player", "position": "Pos", "team": "Team", "cost": "Cost (£m)",
                                "selected_by_percent": "Owned (%)", "ep_next": "Expected pts (next GW)",
                                "is_penalty_taker": "Penalty taker",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                with st.expander("🚑 League-wide injury/suspension feed (scout transfer-ins too)"):
                    st.caption(
                        "The same real status/news/chance_of_playing_next_round fields used to "
                        "flag your own squad above, applied league-wide — so a potential transfer-in "
                        "can be checked for their OWN real availability before you bring them in, "
                        "not just your current squad's."
                    )
                    flags = league_wide_status_flags()
                    if flags.empty:
                        st.caption("No players currently flagged league-wide.")
                    else:
                        display_flags = flags.copy()
                        display_flags["status"] = display_flags["status"].map(STATUS_LABELS).fillna(display_flags["status"])
                        st.dataframe(
                            display_flags.rename(columns={
                                "name": "Player", "position": "Pos", "team": "Team", "cost": "Cost (£m)",
                                "selected_by_percent": "Owned (%)", "status": "Status", "news": "News",
                                "chance_of_playing_next_round": "Chance of playing (%)",
                            }),
                            use_container_width=True, hide_index=True,
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
                            # Doubled from the default 2.0 -- this pool's
                            # predicted_points is last season's closing form,
                            # not this project's trained model, so it's a
                            # noisier, less trustworthy signal than the
                            # historical-gameweek path uses. Verified against
                            # real GW1 data: at the default margin, this noise
                            # alone justified 4 hits for a real squad; at 4.0
                            # it settles to 1 sensible free transfer with no
                            # hit, while a genuinely obvious case (a starter's
                            # predicted points zeroed, simulating an injury)
                            # still correctly clears the bar and takes the hit.
                            min_net_gain_per_hit=4.0,
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

                        # Optional LLM narration of the transfer ABOVE --
                        # every fact handed to it (out_reasons/in_notes) is
                        # already real data computed above (real injury
                        # status, real ep_next) -- the model only narrates,
                        # it never invents a number. Silently absent (no
                        # error shown) if OPENROUTER_API_KEY isn't set or
                        # the free-tier call fails for any reason -- this
                        # is a nice-to-have on top of the real OUT/IN
                        # display above, never something the page needs.
                        out_reasons = {
                            row["name"]: f"{STATUS_LABELS.get(row['status'], row['status'])}"
                            + (f" — {row['news']}" if row["news"] else "")
                            for _, row in flagged.iterrows()
                        } if not flagged.empty else {}
                        in_notes = {
                            row["name"]: f"Real expected points next gameweek (ep_next): {row['ep_next']:.1f}"
                            for _, row in next_pool[next_pool["player_id"].isin(result["transfers_in"])].iterrows()
                            if "ep_next" in row.index and pd.notna(row.get("ep_next"))
                        }
                        explanation, explanation_error = explain_transfer_suggestion_debug(
                            out_names, in_names, result["hit_cost"], result["net_points_gain"],
                            out_reasons=out_reasons, in_notes=in_notes,
                        )
                        if explanation:
                            st.caption("🤖 AI summary (free model, narrates the real numbers above — never a separate source of truth):")
                            st.markdown(f"> {explanation}")
                        elif explanation_error:
                            # A real, honest diagnostic (never the API key itself) for why
                            # narration didn't appear -- e.g. "OPENROUTER_API_KEY is not set"
                            # vs. "OpenRouter returned HTTP 401" vs. rate-limited. A silent
                            # None here was genuinely undebuggable once the key WAS set on
                            # Streamlit Cloud but narration still didn't show up.
                            st.caption(f"🤖 AI summary unavailable: {explanation_error}")
                    else:
                        st.info("No transfer improves on the current squad enough to be worth it — holding is optimal here.")
                else:
                    st.error("Too many squad players missing from this pool to run this check.")


with tab_transfers:
    _render_transfers_tab()

# =============================================================================
# CHIP ADVISOR (against the real live squad, historical seasons as the projection source)
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_chip_advisor_tab():
    st.header("Chip-timing advisor")
    st.caption(
        "A genuine multi-gameweek chip-timing projection (ranking several upcoming gameweeks "
        "against each other) needs real per-gameweek predictions for MULTIPLE future "
        "gameweeks — checked directly: FPL's own API only ever publishes `ep_next` (expected "
        "points for the SINGLE next gameweek), never anything further out. So this can't "
        "honestly rank \"which of the next 5 gameweeks is best\" yet. What it CAN do "
        "honestly, right now, is check whether each chip is worth it for the ONE real next "
        "gameweek, using that same real `ep_next` field — see below."
    )

    if "built_squad" not in st.session_state or st.session_state.get("built_squad_season_gw") != "live_squad":
        st.warning("Build your real squad in the **My Squad** tab first (requires at least one collected gameweek).")
    else:
        squad_df = st.session_state["built_squad"]
        bench = squad_df[~squad_df["in_starting_xi"]]
        starters = squad_df[squad_df["in_starting_xi"]]

        st.subheader("🪑 Bench Boost — next gameweek only")
        bench_total = bench["ep_next"].sum()
        st.metric("Your bench's real expected points next gameweek", f"{bench_total:.1f}")
        st.caption(
            "Sum of FPL's own `ep_next` across your 4 bench players. Worth using Bench Boost "
            "this specific gameweek if that number looks high relative to a normal week for "
            "your bench — there's no multi-week comparison to rank it against yet (see above), "
            "so this is a single data point, not a \"best gameweek\" recommendation."
        )

        st.subheader("👑 Triple Captain — next gameweek only")
        best_captain = suggest_captain(squad_df)
        if best_captain is not None:
            extra_value = best_captain["ep_next"]
            st.metric(
                f"Best starter: {best_captain['name']}",
                f"+{extra_value:.1f} pts extra vs normal captaincy",
            )
            st.caption(
                "Same player suggest_captain() recommends above (FPL's own `ep_next`, "
                "starters only) — Triple Captain's real extra value is exactly one more "
                "multiple of that same real number."
            )
        else:
            st.caption("No starter with a positive `ep_next` right now — nothing to suggest.")

        st.subheader("🔄 Free Hit / Wildcard — next gameweek only")
        st.caption(
            "Real gap between your current squad's real `ep_next` total and a freshly "
            "optimized 15-man squad's `ep_next` total, both scored on the SAME real field "
            "(unlike Transfers, which uses last season's closing form for pre-season "
            "predicted_points — mixing the two here would compare apples to oranges). "
            "**Wildcard's real value is a multi-gameweek strategic call, not just this one "
            "gap** — treat this as a weaker signal than Free Hit's, which genuinely only "
            "ever affects a single gameweek by design."
        )
        ep_pool = ep_next_player_pool()
        current_total_ep = squad_df["ep_next"].sum() if "ep_next" in squad_df.columns else 0.0
        try:
            optimal_squad = optimize_squad(ep_pool)
            optimal_total_ep = optimal_squad["predicted_points"].sum()
            gap = optimal_total_ep - current_total_ep
            fcol1, fcol2, fcol3 = st.columns(3)
            fcol1.metric("Your squad's real ep_next", f"{current_total_ep:.1f}")
            fcol2.metric("Freshly optimized squad's real ep_next", f"{optimal_total_ep:.1f}")
            fcol3.metric("Gap", f"{gap:.1f}", help="A big gap means a Free Hit/Wildcard rebuild would score meaningfully more real expected points next gameweek than your current squad.")
            if gap > 15:
                st.info("💡 A real, meaningful gap — worth genuinely considering a Free Hit here, or a Wildcard if you're also thinking multi-gameweek.")
            else:
                st.caption("Gap isn't large enough to clearly justify a chip on ep_next alone — holding is a reasonable call.")
        except Exception as e:
            st.caption(f"Couldn't run the optimizer against the live ep_next pool right now ({e}).")


with tab_chips:
    _render_chip_advisor_tab()

# =============================================================================
# LEAGUE TRACKER
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_league_tracker_tab():
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


with tab_leagues:
    _render_league_tracker_tab()

# =============================================================================
# PRICE CHANGES (live, auto-refreshes while a gameweek is live)
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_price_changes_tab():
    st.header("Price changes")
    st.caption(
        "Real 2026-27 price movement so far this season — straight from FPL's own "
        "cost_change_start field (verified directly against the live API), the same real "
        "number FPL itself uses to track price rises/falls. Fetched live on every refresh "
        "(60s cache) — no separate action needed to keep this current."
    )

    price_changes = live_price_changes()

    if price_changes.empty:
        st.info(
            "No real price movement yet this season — this is the correct, expected state "
            "very early on, before FPL's own price-change algorithm has reacted to enough "
            "real transfer activity. This will fill in automatically, and keep updating "
            "week by week, as prices actually start moving."
        )
    else:
        risers = price_changes[price_changes["price_change"] > 0]
        fallers = price_changes[price_changes["price_change"] < 0].sort_values("price_change")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.subheader(f"📈 Risers ({len(risers)})")
            if risers.empty:
                st.caption("No risers yet.")
            else:
                st.dataframe(
                    _arrow_price_column(
                        risers[["name", "position", "team", "start_cost", "cost", "price_change"]],
                        "price_change",
                    ).rename(columns={
                        "name": "Player", "position": "Pos", "team": "Team",
                        "start_cost": "Start (£m)", "cost": "Now (£m)", "price_change": "Change (£m)",
                    }),
                    use_container_width=True, hide_index=True,
                )
        with rcol2:
            st.subheader(f"📉 Fallers ({len(fallers)})")
            if fallers.empty:
                st.caption("No fallers yet.")
            else:
                st.dataframe(
                    _arrow_price_column(
                        fallers[["name", "position", "team", "start_cost", "cost", "price_change"]],
                        "price_change",
                    ).rename(columns={
                        "name": "Player", "position": "Pos", "team": "Team",
                        "start_cost": "Start (£m)", "cost": "Now (£m)", "price_change": "Change (£m)",
                    }),
                    use_container_width=True, hide_index=True,
                )

    st.divider()
    st.subheader("🔮 Likely to move next")
    st.caption(
        "Players with real, current transfer MOMENTUM (net transfers this gameweek — a real, "
        "public leading indicator FPL's price-change algorithm reacts to) who haven't had "
        "their price move yet — distinct from the risers/fallers above, which already happened."
    )
    movers = likely_price_movers()
    if movers.empty:
        st.caption("No significant transfer momentum on unmoved players right now.")
    else:
        mcol1, mcol2 = st.columns(2)
        likely_risers = movers[movers["net_transfers"] > 0].head(5)
        likely_fallers = movers[movers["net_transfers"] < 0].sort_values("net_transfers").head(5)
        with mcol1:
            st.caption("Likely risers (heavy net transfers IN)")
            st.dataframe(
                _arrow_price_column(
                    likely_risers[["name", "position", "team", "cost", "net_transfers"]], "net_transfers",
                ).rename(columns={"name": "Player", "position": "Pos", "team": "Team", "cost": "Now (£m)", "net_transfers": "Net transfers in"}),
                use_container_width=True, hide_index=True,
            )
        with mcol2:
            st.caption("Likely fallers (heavy net transfers OUT)")
            st.dataframe(
                _arrow_price_column(
                    likely_fallers[["name", "position", "team", "cost", "net_transfers"]], "net_transfers",
                ).rename(columns={"name": "Player", "position": "Pos", "team": "Team", "cost": "Now (£m)", "net_transfers": "Net transfers out"}),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("🌙 Tonight's projected price changes")
    st.caption(
        "FPL's OWN real forecast for the very next price update — distinct from the momentum-"
        "based 'likely to move' above (this project's own guess from transfer activity), this is "
        "FPL's own published projection (`price_change_projections`/`likelihood`, verified "
        "directly against the live API). A real -5..+5 likelihood scale: further from 0 means a "
        "stronger real signal from FPL's own algorithm. No specific clock time is shown for when "
        "the change actually lands — FPL's public API doesn't expose one, so this project only "
        "surfaces what it can verify, not an assumed schedule."
    )
    projections = tonight_price_projections()
    if projections.empty:
        st.caption("No real price-change forecast available right now.")
    else:
        jcol1, jcol2 = st.columns(2)
        likely_rise = projections[projections["likelihood"] > 0].sort_values("likelihood", ascending=False).head(5)
        likely_fall = projections[projections["likelihood"] < 0].sort_values("likelihood").head(5)
        with jcol1:
            st.caption("Projected to rise tonight")
            st.dataframe(
                _arrow_price_column(
                    likely_rise[["name", "position", "team", "cost", "projected_percent", "likelihood"]], "likelihood",
                ).rename(columns={
                    "name": "Player", "position": "Pos", "team": "Team", "cost": "Now (£m)",
                    "projected_percent": "Projected %", "likelihood": "Likelihood (-5..+5)",
                }),
                use_container_width=True, hide_index=True,
            )
        with jcol2:
            st.caption("Projected to fall tonight")
            st.dataframe(
                _arrow_price_column(
                    likely_fall[["name", "position", "team", "cost", "projected_percent", "likelihood"]], "likelihood",
                ).rename(columns={
                    "name": "Player", "position": "Pos", "team": "Team", "cost": "Now (£m)",
                    "projected_percent": "Projected %", "likelihood": "Likelihood (-5..+5)",
                }),
                use_container_width=True, hide_index=True,
            )


with tab_prices:
    _render_price_changes_tab()

# =============================================================================
# PL TABLE (real league standings, computed from fixtures.csv's own scores)
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_pl_table_tab():
    st.header("Premier League table")
    st.caption(
        "The real, current 2026-27 table — computed directly from fixtures.csv's own recorded "
        "match scores, not FPL's strength ratings or any derived metric. A match's real score is "
        "counted as soon as it's recorded, without waiting on the 'finished' flag (verified "
        "directly: 'finished' stays False for hours after a match ends, until bonus points lock "
        "in) so results appear here as soon as they're actually known. Fetched live on every "
        "refresh (60s cache)."
    )
    table = premier_league_table_with_movement()
    if table.empty or table["played"].sum() == 0:
        st.info("No results recorded yet this season.")
    else:
        display_table = table.copy()
        display_table["movement"] = display_table["movement"].map(
            lambda m: "—" if pd.isna(m) else ("🟢▲" if m > 0 else ("🔴▼" if m < 0 else "▬")) + f" {abs(m) if pd.notna(m) else ''}"
        )
        st.dataframe(
            display_table.rename(columns={
                "team": "Team", "played": "P", "won": "W", "drawn": "D", "lost": "L",
                "gf": "GF", "ga": "GA", "gd": "GD", "points": "Pts", "movement": "Since last GW",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "**Since last GW** — real table-position movement vs. one gameweek ago, computed "
            "from the exact same fixtures data at both points in time (not a guess or a stored "
            "snapshot) — a genuine re-derivable comparison. Shows '—' for the season's opening "
            "gameweek, since there's no real earlier table to compare against yet."
        )


with tab_table:
    _render_pl_table_tab()

# =============================================================================
# SEASON INSIGHTS (real, live 2026-27 leaderboards)
# =============================================================================
@st.fragment(run_every=60 if is_gameweek_live() else None)
def _render_season_insights_tab():
    st.header("2026-27 season insights")
    st.caption(
        "Real, current-season leaderboards — every number below is a single FPL field read "
        "directly (goals_scored, assists, yellow_cards, red_cards, defensive_contribution, "
        "total_points, form), not a derived or invented score, and not gated behind a "
        "minimum-games cutoff this early in the season. Fetched live on every refresh (60s "
        "cache) — no separate wiring needed."
    )
    boards = season_leaderboards()

    def _show_board(title, key, unit, icon, col):
        with col:
            st.subheader(f"{icon} {title}")
            board = boards[key]
            if board.empty:
                st.caption("Nothing recorded yet.")
            else:
                st.dataframe(
                    board.rename(columns={
                        "name": "Player", "position": "Pos", "team": "Team", "value": unit,
                    }),
                    use_container_width=True, hide_index=True,
                )

    icol1, icol2 = st.columns(2)
    _show_board("Golden Boot (goals)", "golden_boot", "Goals", "⚽", icol1)
    _show_board("Assists", "assists", "Assists", "🎯", icol2)

    icol3, icol4 = st.columns(2)
    _show_board("Yellow cards", "yellow_cards", "Yellow", "🟨", icol3)
    _show_board("Red cards", "red_cards", "Red", "🟥", icol4)

    st.divider()
    icol5, icol6 = st.columns(2)
    with icol5:
        st.subheader("🛡️ DEFCON leaders")
        st.caption(
            "FPL's own real `defensive_contribution` stat (tackles + interceptions + clearances "
            "for defenders; ball recoveries + tackles + interceptions for midfielders/forwards) — "
            "the same real number that earns DEFCON bonus points in FPL's actual scoring rules."
        )
        board = boards["defensive_contribution"]
        if board.empty:
            st.caption("Nothing recorded yet.")
        else:
            st.dataframe(
                board.rename(columns={"name": "Player", "position": "Pos", "team": "Team", "value": "DefCon"}),
                use_container_width=True, hide_index=True,
            )
    with icol6:
        st.subheader("👑 MVP so far")
        st.caption(
            "The real, current total_points leader — FPL's own already-trusted number, not a "
            "model-derived score. A trained-model MVP estimate isn't used here for the same "
            "reason Transfers flags its own predicted_points as unreliable this early: there "
            "aren't enough real 2026-27 gameweeks yet for this project's model to have a genuine "
            "signal beyond last season's closing form."
        )
        board = boards["mvp"]
        if board.empty:
            st.caption("Nothing recorded yet.")
        else:
            st.dataframe(
                board.rename(columns={"name": "Player", "position": "Pos", "team": "Team", "value": "Points"}),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("🔥 In-form right now")
    st.caption(
        "FPL's own real `form` field — their published rolling average points over recent "
        "gameweeks, the same real number Transfers/differential/captain logic elsewhere in this "
        "app treats as the closest thing to genuine current-season signal. Early in a season "
        "this will look similar to the MVP board above (only a handful of real gameweeks exist "
        "so far) — that's a correct reflection of the data, not a bug. As more real gameweeks "
        "are played, this is what actually separates a hot streak from a season-long total."
    )
    form_board = boards["in_form"]
    if form_board.empty:
        st.caption("Nothing recorded yet.")
    else:
        st.dataframe(
            form_board.rename(columns={"name": "Player", "position": "Pos", "team": "Team", "value": "Form"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("🏟️ Team-level insights")
    st.caption(
        "Real, current-season team form — best attack (most real goals scored) and best defense "
        "(fewest real goals conceded), both derived from the same PL Table data above, not a "
        "separate recompute. Also shows each real Premier League team's single most-selected "
        "asset (FPL's own real `selected_by_percent`) — a quick 'who's the team's most-trusted "
        "pick right now' view."
    )
    tinsights = team_insights()
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.caption("⚔️ Best attack (goals scored)")
        board = tinsights["best_attack"]
        if board.empty:
            st.caption("No results recorded yet.")
        else:
            st.dataframe(
                board.rename(columns={"team": "Team", "played": "P", "gf": "Goals"}),
                use_container_width=True, hide_index=True,
            )
    with tcol2:
        st.caption("🧱 Best defense (goals conceded)")
        board = tinsights["best_defense"]
        if board.empty:
            st.caption("No results recorded yet.")
        else:
            st.dataframe(
                board.rename(columns={"team": "Team", "played": "P", "ga": "Conceded"}),
                use_container_width=True, hide_index=True,
            )

    st.caption("Most-owned player per team")
    owned_board = tinsights["most_owned_players"]
    if owned_board.empty:
        st.caption("Nothing recorded yet.")
    else:
        st.dataframe(
            owned_board.rename(columns={
                "team": "Team", "name": "Player", "position": "Pos", "selected_by_percent": "Owned (%)",
            }),
            use_container_width=True, hide_index=True,
        )


with tab_insights:
    _render_season_insights_tab()
