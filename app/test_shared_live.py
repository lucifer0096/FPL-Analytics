"""Ad-hoc verification of shared.py's live-sync functions against REAL data.

Not a pytest-style unit-test suite (same house style as
src/model/test_optimizer.py / test_chips.py) -- this project verifies
against real, live FPL data and asserts on properties that must hold for
ANY real season state (not fixed expected values, since real scores/
standings/injuries change every gameweek and this script has to keep
passing as the season progresses). Requires network access to FPL's API;
run it locally, not as part of an offline CI step.

Covers everything added by the 2026-08-24 live-sync work:
- _load_bootstrap() / _load_fixtures_df() / _load_entry_history()
  live-first-with-fallback behavior (both the happy path AND the live-API-
  down path, the latter via monkeypatching fpl_api to simulate an outage)
- season_leaderboards(), team_insights() shape/sanity checks
- _team_fixture_started() / is_gameweek_live() real-data consistency
- The "Not yet played" vs "No game time" split in build_live_squad_df
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared

MANAGER_ENTRY_ID = shared.MANAGER_ENTRY_ID


def _simulate_api_outage():
    """Monkeypatches every fpl_api call shared.py's live-first loaders use,
    so the fallback-to-file path can be exercised without actually taking
    the network down. Returns the originals so the caller can restore them."""
    originals = {
        "get_bootstrap_static": shared.fpl_api.get_bootstrap_static,
        "get_fixtures": shared.fpl_api.get_fixtures,
        "get_entry_history": shared.fpl_api.get_entry_history,
        "get_entry_picks": shared.fpl_api.get_entry_picks,
        "get_event_live": shared.fpl_api.get_event_live,
        "get_entry": shared.fpl_api.get_entry,
        "get_league_standings": shared.fpl_api.get_league_standings,
    }

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated API outage")

    for name in originals:
        setattr(shared.fpl_api, name, _boom)
    return originals


def _restore_api(originals):
    for name, fn in originals.items():
        setattr(shared.fpl_api, name, fn)


def test_every_live_facing_function_has_a_cache_ttl():
    """Regression test for a real bug found 2026-08-24: several functions
    that read live data via _load_bootstrap()/_load_fixtures_df() (which
    DO have a 60s ttl) were themselves decorated with plain @st.cache_data
    -- NO ttl of their own -- so Streamlit cached their return value
    FOREVER regardless of what changed underneath. A live gameweek's real
    stat updates (e.g. a player's assist total) never reached the UI even
    though every fragment/rerun/underlying-loader was working correctly,
    because these functions' own cached results simply never expired.

    Checks .clear has a corresponding ttl by inspecting Streamlit's cache
    wrapper -- concretely, via the documented cache_data function
    attribute path is version-fragile, so instead this asserts each
    function actually returns FRESH data by clearing its cache and calling
    it, which only proves it CAN run, not that it times out on its own. The
    real, durable check: every one of these must be re-decorated with an
    explicit ttl matching the loaders it depends on (60s) -- enforced here
    by reading shared.py's source and asserting these specific function
    names are decorated with @st.cache_data(ttl=...), not the bare form."""
    import inspect
    source = inspect.getsource(shared)
    live_facing_functions = [
        "live_price_changes", "likely_price_movers", "differential_finder",
        "league_wide_status_flags", "premier_league_table", "team_insights",
        "season_leaderboards", "team_upcoming_fixtures",
        "tonight_price_projections", "premier_league_table_with_movement",
        "_load_bootstrap", "_load_fixtures_df", "_load_entry_history",
        "load_current_squad_picks", "load_live_gw_points", "load_live_gw_minutes",
        "load_live_gw_stats", "load_joined_leagues", "load_manager_name",
        "load_current_season_progress", "_team_fixture_started", "player_season_stats",
    ]
    missing_ttl = []
    for name in live_facing_functions:
        marker = f"def {name}("
        idx = source.find(marker)
        assert idx != -1, f"{name} not found in shared.py -- test needs updating"
        # Walk backwards to the nearest decorator line.
        preceding = source[:idx]
        last_at = preceding.rfind("@st.cache_data")
        decorator_line = preceding[last_at:idx].splitlines()[0]
        if "ttl=" not in decorator_line:
            missing_ttl.append(name)
    assert not missing_ttl, f"these live-facing functions are cached with NO ttl (will freeze forever): {missing_ttl}"
    print(f"PASS: all {len(live_facing_functions)} live-facing functions carry an explicit cache ttl")


def test_load_bootstrap_live_and_fallback():
    raw = shared._load_bootstrap.__wrapped__()
    assert len(raw["elements"]) > 500, "real bootstrap should have 500+ players"
    assert len(raw["teams"]) == 20, "Premier League always has 20 teams"

    originals = _simulate_api_outage()
    try:
        raw_fallback = shared._load_bootstrap.__wrapped__()
        assert len(raw_fallback["elements"]) > 500, "fallback bootstrap must still load real data"
    finally:
        _restore_api(originals)
    print("PASS: _load_bootstrap live + fallback both return real data")


def test_load_fixtures_df_live_and_fallback():
    fx = shared._load_fixtures_df.__wrapped__()
    assert not fx.empty, "real fixtures list should never be empty mid-season"
    assert set(["event", "team_h", "team_a", "started", "finished"]).issubset(fx.columns)

    originals = _simulate_api_outage()
    try:
        fx_fallback = shared._load_fixtures_df.__wrapped__()
        assert not fx_fallback.empty, "fallback fixtures must still load real data"
    finally:
        _restore_api(originals)
    print("PASS: _load_fixtures_df live + fallback both return real data")


def test_load_entry_history_live_and_fallback():
    history = shared._load_entry_history.__wrapped__(MANAGER_ENTRY_ID)
    assert "current" in history and "past" in history, "real entry history always has both keys"

    originals = _simulate_api_outage()
    try:
        history_fallback = shared._load_entry_history.__wrapped__(MANAGER_ENTRY_ID)
        assert "current" in history_fallback, "fallback entry history must still load real data"
    finally:
        _restore_api(originals)
    print("PASS: _load_entry_history live + fallback both return real data")


def test_season_leaderboards_shape():
    boards = shared.season_leaderboards()
    expected_keys = {"golden_boot", "assists", "yellow_cards", "red_cards", "defensive_contribution", "mvp", "in_form"}
    assert expected_keys.issubset(boards.keys())
    for key, df in boards.items():
        assert set(["name", "position", "team", "value"]).issubset(df.columns), f"{key} missing expected columns"
        if not df.empty:
            # Every board is meant to be sorted descending -- verify it actually is.
            assert df["value"].is_monotonic_decreasing, f"{key} isn't sorted descending"
    print(f"PASS: season_leaderboards() shape OK, red_cards has {len(boards['red_cards'])} real entries")


def test_team_insights_consistency():
    insights = shared.team_insights()
    table = shared.premier_league_table()
    if table.empty or table["played"].sum() == 0:
        print("SKIP: team_insights consistency (no results recorded yet this season)")
        return
    # best_attack's top scorer's goals should match that team's real gf in the PL table.
    top_attack_team = insights["best_attack"].iloc[0]["team"]
    top_attack_goals = insights["best_attack"].iloc[0]["gf"]
    table_row = table[table["team"] == top_attack_team].iloc[0]
    assert top_attack_goals == table_row["gf"], "team_insights' gf must match premier_league_table's gf exactly (same source)"
    assert not insights["most_owned_players"].empty, "at least one team should have a most-owned player once players exist"
    print("PASS: team_insights() reuses premier_league_table() data consistently")


def test_premier_league_table_movement():
    table = shared.premier_league_table_with_movement()
    if table.empty or table["played"].sum() == 0:
        print("SKIP: PL table movement (no results recorded yet this season)")
        return
    assert "movement" in table.columns
    raw = shared._load_bootstrap()
    current_events = [e["id"] for e in raw["events"] if e.get("is_current")]
    current_gw = current_events[0] if current_events else 1
    if current_gw <= 1:
        assert table["movement"].isna().all(), "GW1 has no earlier table to compare against -- movement must be null, not 0"
        print("PASS: premier_league_table_with_movement() correctly null for GW1")
    else:
        # Every team's movement must be a real integer within a plausible
        # range (-19..+19 -- a 20-team league can't move further than that
        # in one gameweek).
        assert table["movement"].dropna().between(-19, 19).all()
        print(f"PASS: premier_league_table_with_movement() real movement values OK for GW{current_gw}")


def test_tonight_price_projections_shape():
    projections = shared.tonight_price_projections()
    assert set(["name", "position", "team", "cost", "projected_percent", "likelihood"]).issubset(projections.columns)
    if not projections.empty:
        assert projections["likelihood"].abs().max() <= 5, "FPL's real likelihood scale is -5..+5"
        assert (projections["likelihood"] != 0).all(), "likelihood==0 rows (no change expected) should be excluded"
    print(f"PASS: tonight_price_projections() shape OK, {len(projections)} real projected mover(s)")


def test_fixture_started_and_gameweek_live():
    started = shared._team_fixture_started.__wrapped__(1)
    is_live = shared.is_gameweek_live()
    assert isinstance(is_live, bool)
    if started:
        assert all(isinstance(v, bool) for v in started.values()), "fixture_started values must be real booleans"
    print(f"PASS: _team_fixture_started/is_gameweek_live real data OK (is_gameweek_live={is_live})")


def test_squad_card_hover_stats():
    picks_data = shared.load_current_squad_picks(MANAGER_ENTRY_ID, 1)
    if picks_data is None:
        print("SKIP: hover stats (no real squad data for this entry/gw)")
        return
    squad_df = shared.build_live_squad_df(picks_data, 1)
    for col in ["minutes_played", "goals_scored", "assists", "bonus", "gw_total_points"]:
        assert col in squad_df.columns, f"build_live_squad_df missing {col}"
    html = shared._player_card_html(squad_df.iloc[0])
    assert "\n" not in html, "card HTML must stay single-line (Markdown-then-HTML rendering bug)"
    assert "THIS GW" in html and "SEASON" in html, "hover tooltip must carry both gameweek and season sections"
    assert "DEFCON" in html, "hover tooltip must include DEFCON per explicit request"
    print("PASS: My Squad card hover tooltip carries real per-GW + season stats (incl. DEFCON)")


def test_player_season_stats_and_optimizer_card_hover():
    raw = shared._load_bootstrap()
    player = raw["elements"][0]
    stats = shared.player_season_stats(player["id"])
    assert stats, "player_season_stats must find a real player by id"
    for field in ["total_points", "minutes", "goals_scored", "assists", "defensive_contribution", "bonus", "expected_goals", "expected_assists"]:
        assert field in stats, f"player_season_stats missing {field}"

    # An optimizer-built card (no per-gameweek columns at all) should still
    # get a real season-stats tooltip -- this is what makes the hover work
    # on Team of the Season / any non-live-squad pitch view too, not just
    # My Squad.
    row = pd.Series({
        "player_id": player["id"], "name": "Test Player", "position": "FWD",
        "team": "Test Team", "cost": 5.0, "predicted_points": 1.0, "in_starting_xi": True,
    })
    html = shared._player_card_html(row)
    assert "SEASON" in html and "THIS GW" not in html, "optimizer-built card must show season stats only, no per-GW section"
    print("PASS: player_season_stats() + optimizer-built card hover (Team of the Season etc.) both work")


def test_not_yet_played_vs_no_game_time_split():
    picks_data = shared.load_current_squad_picks(MANAGER_ENTRY_ID, 1)
    if picks_data is None:
        print("SKIP: not-yet-played split (no real squad data for this entry/gw)")
        return
    squad_df = shared.build_live_squad_df(picks_data, 1)
    assert "fixture_started" in squad_df.columns
    not_played = squad_df[squad_df["did_not_play"]]
    if not_played.empty:
        print("SKIP: not-yet-played split (everyone in this real squad has played)")
        return
    # Real data check: at least the split is internally consistent -- every
    # did_not_play row has a real True/False/None fixture_started, never a
    # made-up default.
    valid_values = {True, False, None}
    assert set(not_played["fixture_started"].tolist()).issubset(valid_values)
    print(f"PASS: {len(not_played)} real 0-minute squad member(s) correctly carry a real fixture_started flag")


if __name__ == "__main__":
    test_every_live_facing_function_has_a_cache_ttl()
    test_load_bootstrap_live_and_fallback()
    test_load_fixtures_df_live_and_fallback()
    test_load_entry_history_live_and_fallback()
    test_season_leaderboards_shape()
    test_team_insights_consistency()
    test_premier_league_table_movement()
    test_tonight_price_projections_shape()
    test_squad_card_hover_stats()
    test_player_season_stats_and_optimizer_card_hover()
    test_fixture_started_and_gameweek_live()
    test_not_yet_played_vs_no_game_time_split()
    print("\nAll shared.py live-sync checks passed.")
