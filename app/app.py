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
    load_current_squad_picks, build_live_squad_df, load_joined_leagues, live_price_changes,
    team_upcoming_fixtures, average_fixture_difficulty, suggest_captain,
    render_pitch, inject_shared_css, render_sidebar,
    optimize_transfers, POSITION_REQUIREMENTS,
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


tab_squad, tab_transfers, tab_chips, tab_leagues, tab_prices = st.tabs(
    ["🧠 My Squad", "🔁 Transfers", "🃏 Chip Advisor", "🏅 League Tracker", "💰 Price Changes"]
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
            st.info(
                "⚠️ This early in the season, predicted_points still comes from last season's "
                "**closing 2025-26 form**, not real 2026-27 in-season data — there aren't enough "
                "finished 2026-27 gameweeks yet for the trained model to use real fixture "
                "difficulty/current form (see the Historical & Model page's Model Performance tab "
                "for how that model works once it can be used here). Treat suggested transfers as "
                "a rough signal, not a confident recommendation, until this improves — a real "
                "current-season model will automatically start being used here once enough live "
                "gameweeks are collected."
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

# =============================================================================
# PRICE CHANGES (live, updates automatically week by week)
# =============================================================================
with tab_prices:
    st.header("Price changes")
    st.caption(
        "Real 2026-27 price movement so far this season — straight from FPL's own "
        "cost_change_start field (verified directly against the live API), the same real "
        "number FPL itself uses to track price rises/falls. Updates automatically as the "
        "collector runs each day; no separate action needed to keep this current."
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
                    risers[["name", "position", "team", "start_cost", "cost", "price_change"]]
                    .rename(columns={
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
                    fallers[["name", "position", "team", "start_cost", "cost", "price_change"]]
                    .rename(columns={
                        "name": "Player", "position": "Pos", "team": "Team",
                        "start_cost": "Start (£m)", "cost": "Now (£m)", "price_change": "Change (£m)",
                    }),
                    use_container_width=True, hide_index=True,
                )
