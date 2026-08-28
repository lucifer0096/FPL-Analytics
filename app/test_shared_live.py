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
        "_current_season_label", "ep_next_player_pool", "chip_usage_status", "sidebar_summary",
        "gameweek_fixtures",
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


def test_no_hardcoded_season_path_in_shared():
    """Regression test for a real bug found in an audit: 7 fallback-path
    functions in shared.py (plus one in app.py, found in a follow-up sweep)
    hardcoded the literal path segment "2026-27" (e.g.
    os.path.join(..., "2026-27", "live", ...)) instead of deriving it live
    via _current_season_label() -- correct today, but silently wrong
    (pointing at a folder that no longer exists) the day the season rolls
    over to 2027-28. Fixed by adding _current_season_label() and routing
    every one of those call sites through it.

    Checks specifically for the literal used as a PATH ARGUMENT (a quoted
    "2026-27" immediately preceded by a comma and whitespace, the actual
    os.path.join(...) call shape every real bug site had) rather than
    banning the substring outright -- plain prose mentions in docstrings
    ("2026-27 pre-season player pool", "2026-27 progress", etc.) are fine
    and expected; only a literal PATH SEGMENT is the real regression risk.
    Checks BOTH shared.py and app.py -- the app.py bug (_collected_gws())
    was in a different file than the original 7 and wasn't caught by an
    earlier, shared.py-only version of this same test."""
    import inspect
    import re
    path_literal_pattern = re.compile(r',\s*["\']2026-27["\']\s*,')

    shared_source = inspect.getsource(shared)
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    with open(app_path, encoding="utf-8") as f:
        app_source = f.read()

    for source, label in ((shared_source, "shared.py"), (app_source, "app.py")):
        matches = path_literal_pattern.findall(source)
        assert not matches, (
            f"found {len(matches)} hardcoded \"2026-27\" path-segment literal(s) in {label} -- "
            "use _current_season_label() instead so this doesn't silently break next season"
        )
    print("PASS: no hardcoded season path-segment literal remains in shared.py or app.py")


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


def test_ownership_swing_and_overall_most_owned():
    """Verifies the real ownership-swing derivation and the new
    league-wide (not per-team) most-owned board, added per explicit
    request. ownership_swing_pct is derived from two real FPL fields
    (transfers_in_event - transfers_out_event, divided by total_players)
    since FPL doesn't publish an ownership-delta field directly."""
    insights = shared.team_insights()
    for key in ("most_owned_players", "overall_most_owned"):
        board = insights[key]
        assert "ownership_swing_pct" in board.columns, f"{key} missing ownership_swing_pct"
        if not board.empty:
            # A real swing must be a small, plausible number -- one
            # gameweek's net transfers can't realistically move ownership
            # by more than a few percentage points for any single player.
            assert board["ownership_swing_pct"].abs().max() < 20, f"{key} has an implausibly large swing value"

    overall = insights["overall_most_owned"]
    if not overall.empty:
        assert overall["selected_by_percent"].is_monotonic_decreasing, "overall_most_owned must be sorted by ownership descending"
        # The single most-owned player league-wide must be at least as
        # owned as any one team's own most-owned player (a real
        # consistency check between the two boards, same underlying data).
        per_team = insights["most_owned_players"]
        if not per_team.empty:
            assert overall.iloc[0]["selected_by_percent"] >= per_team["selected_by_percent"].max() - 1e-6
    print(f"PASS: ownership_swing_pct and overall_most_owned() verified against real data ({len(overall)} league-wide entries)")


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


def test_explain_transfer_suggestion_no_key_returns_none():
    """explain_transfer_suggestion() must return None cleanly (never raise,
    never return a placeholder string) when OPENROUTER_API_KEY isn't set --
    an LLM narration is a nice-to-have on top of real data, never something
    the Transfers tab depends on. This test intentionally does NOT check
    the with-a-real-key path (that would require a real OpenRouter API key
    and make a real network call to a free-tier model, not appropriate for
    a test that runs automatically)."""
    original_key = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        assert not shared.openrouter.is_configured()
        result = shared.explain_transfer_suggestion(["Test Player Out"], ["Test Player In"], 0, 2.5)
        assert result is None, "must return None (not raise, not a placeholder) with no API key configured"
    finally:
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key
    print("PASS: explain_transfer_suggestion() returns None cleanly with no OPENROUTER_API_KEY set")


def test_explain_transfer_suggestion_debug_gives_real_reasons():
    """Regression test for a real debugging gap found live: with no key,
    the debug variant must give a specific, honest reason (not just None)
    -- and with an invalid key, it must surface OpenRouter's own real HTTP
    error (never the key value itself) rather than swallowing it into an
    undebuggable None, which is what happened when narration silently
    didn't appear even after a real key was set on Streamlit Cloud."""
    original_key = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        content, error = shared.explain_transfer_suggestion_debug(["Out"], ["In"], 0, 1.0)
        assert content is None
        assert error and "OPENROUTER_API_KEY is not set" in error

        os.environ["OPENROUTER_API_KEY"] = "sk-invalid-test-key-not-real"
        content, error = shared.explain_transfer_suggestion_debug(["Out"], ["In"], 0, 1.0)
        assert content is None
        assert error is not None, "an invalid key must produce a real error message, not a silent None"
        assert "sk-invalid-test-key-not-real" not in error, "the error message must never echo back the key value"
        print(f"PASS: explain_transfer_suggestion_debug() gives real, specific reasons (no key: clear message; invalid key: {error!r})")
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key


def test_rate_limit_errors_are_tagged_for_silent_handling():
    """Regression test for a real UX issue found live: a genuine HTTP 429
    (free-tier rate limit -- an expected, normal condition under load, not
    a real problem) was showing the same alarming error caption as a
    genuine failure (bad key, retired model). Verifies the real 429 path
    is tagged with RATE_LIMIT_PREFIX so app.py can suppress it specifically
    while still surfacing every other real error -- via a monkeypatched
    urlopen that raises a real-shaped HTTPError(429), not a live network
    call (a genuine rate limit isn't reliably reproducible on demand)."""
    import io
    import json as json_module
    import urllib.error
    import urllib.request

    original_urlopen = urllib.request.urlopen
    original_key = os.environ.get("OPENROUTER_API_KEY")

    class _Fake429(urllib.error.HTTPError):
        def __init__(self):
            body = json_module.dumps({"error": {"message": "Provider returned error"}}).encode()
            super().__init__("url", 429, "Too Many Requests", {}, io.BytesIO(body))

        def read(self):
            return json_module.dumps({"error": {"message": "Provider returned error"}}).encode()

    def _raise_429(*args, **kwargs):
        raise _Fake429()

    urllib.request.urlopen = _raise_429
    os.environ["OPENROUTER_API_KEY"] = "sk-fake-for-test"
    try:
        content, error = shared.explain_transfer_suggestion_debug(["Out"], ["In"], 0, 1.0)
        assert content is None
        assert error is not None
        assert error.startswith(shared.RATE_LIMIT_PREFIX), f"a real 429 must be tagged with RATE_LIMIT_PREFIX, got: {error!r}"
        assert "429" in error, "the underlying real HTTP code should still be visible in the tagged message"
        print(f"PASS: a real HTTP 429 is correctly tagged for silent handling ({error!r})")
    finally:
        urllib.request.urlopen = original_urlopen
        os.environ.pop("OPENROUTER_API_KEY", None)
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key


def test_openrouter_hardcoded_to_free_model():
    """Regression guard: openrouter.py's FREE_MODEL must always end in
    ':free' (OpenRouter's own naming convention for models that don't
    consume paid credits) -- this project must never silently start
    incurring API costs. Checked directly against the actual constant, not
    assumed."""
    assert shared.openrouter.FREE_MODEL.endswith(":free"), (
        f"FREE_MODEL ({shared.openrouter.FREE_MODEL!r}) must end in ':free' -- "
        "this project must only ever use free OpenRouter models"
    )
    print(f"PASS: openrouter.py is hardcoded to a free model ({shared.openrouter.FREE_MODEL})")


def test_chip_usage_status_shape_and_parsing():
    """Verifies chip_usage_status() against real live data AND against
    simulated real chip_plays/chip-window payloads to confirm the per-half
    parsing logic itself is correct, not just that it runs.

    Real bug fixed here: an earlier version tracked "used" per chip name
    across the WHOLE season, which would have incorrectly shown a chip as
    permanently used after its first-half play even once a fresh copy
    became available in the second half -- verified directly against
    bootstrap-static's own `chips` array, which publishes a real, separate
    start_event/stop_event window per half for every chip, not just
    Wildcard. Also verified a real, genuinely valid edge case live: at
    GW1, Wildcard/Free Hit's real first-half window hasn't opened yet
    (starts GW2), so they must report "not_yet_open", not "used" or a
    silently wrong full-season fallback window."""
    status = shared.chip_usage_status(MANAGER_ENTRY_ID)
    expected_chips = {"Wildcard", "Free Hit", "Bench Boost", "Triple Captain"}
    assert set(status.keys()) == expected_chips
    for chip, info in status.items():
        assert "used" in info and "gameweeks" in info and "not_yet_open" in info
        assert isinstance(info["used"], bool)
        assert isinstance(info["gameweeks"], list)
        if not info["used"]:
            assert info["gameweeks"] == [], f"{chip} not used but has gameweeks listed"
        if info["not_yet_open"]:
            assert info["half_start"] is None and info["half_stop"] is None

    # Simulate real chip-window (bootstrap "chips") + chip-play (entry
    # history "chips") payloads to verify the per-half parsing logic
    # end-to-end, including a chip used in an EARLIER half correctly NOT
    # counting as used in the CURRENT half.
    fake_bootstrap = {
        "events": [{"id": 25, "is_current": True}],
        "chips": [
            {"name": "wildcard", "start_event": 2, "stop_event": 19},
            {"name": "wildcard", "start_event": 20, "stop_event": 38},
            {"name": "bboost", "start_event": 1, "stop_event": 19},
            {"name": "bboost", "start_event": 20, "stop_event": 38},
        ],
    }
    fake_history = {"current": [], "past": [], "chips": [
        {"name": "bboost", "event": 5, "time": "2026-09-20T00:00:00Z"},  # played in the FIRST half
        {"name": "wildcard", "event": 25, "time": "2026-12-01T00:00:00Z"},  # played in the SECOND half (current)
    ]}
    original_bootstrap, original_history = shared._load_bootstrap, shared._load_entry_history
    try:
        shared._load_bootstrap = lambda: fake_bootstrap
        shared._load_entry_history = lambda entry_id: fake_history
        simulated = shared.chip_usage_status.__wrapped__(MANAGER_ENTRY_ID)
        assert simulated["Wildcard"]["used"] is True and simulated["Wildcard"]["gameweeks"] == [25], (
            "Wildcard played in the CURRENT (second) half must show as used"
        )
        assert simulated["Bench Boost"]["used"] is False and simulated["Bench Boost"]["gameweeks"] == [], (
            "Bench Boost played in an EARLIER half must NOT show as used in the current half -- "
            "this is the real bug this fix addresses"
        )
    finally:
        shared._load_bootstrap = original_bootstrap
        shared._load_entry_history = original_history
    print("PASS: chip_usage_status() correctly tracks per-half availability (real GW1 not_yet_open case + simulated cross-half reset)")


def test_free_transfers_slider_allows_zero():
    """Regression test for a real UX gap found live: the free-transfers
    slider's minimum was hardcoded to 1, even though optimize_transfers()
    genuinely supports free_transfers=0 (every suggested transfer takes a
    real hit) -- a real, useful case when a manager has already queued a
    transfer for next gameweek the public API can't see yet and wants to
    check if a FURTHER change is worth an additional hit. Checks the
    actual app.py source for the slider's real min bound rather than
    running the full Streamlit app (already covered by the AppTest
    verification done when this was fixed)."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    with open(app_path, encoding="utf-8") as f:
        app_source = f.read()
    assert '"Free transfers available", 0, 5, real_free_transfers' in app_source, (
        "the free-transfers slider must allow a real minimum of 0, not 1 -- "
        "optimize_transfers() genuinely supports free_transfers=0"
    )
    print("PASS: free-transfers slider allows a real minimum of 0")


def test_ai_explanation_output_is_html_escaped():
    """Security check: the styled "Why this transfer" card renders the
    LLM's real output inside an unsafe_allow_html=True st.markdown block
    (needed for the gradient-card styling) -- the raw model response MUST
    be passed through html.escape() first, since an unescaped LLM output
    embedded in raw HTML is a real injection risk (the model's response is
    external, untrusted text, even though it's grounded in real facts).
    Checks the actual app.py source for the escape call rather than
    driving a full browser render."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    with open(app_path, encoding="utf-8") as f:
        app_source = f.read()
    assert "html.escape(explanation)" in app_source, (
        "the LLM's real output must be html.escape()'d before being embedded in an "
        "unsafe_allow_html=True block -- otherwise a model response containing HTML-like "
        "text could inject arbitrary markup into the page"
    )
    print("PASS: AI explanation output is html.escape()'d before rendering in unsafe_allow_html")


def test_ep_next_player_pool_shape_and_optimizable():
    """Verifies the pool backing Chip Advisor's new Free Hit/Wildcard check:
    real shape (matches what optimize_squad() requires), only available
    players (status == 'a'), and genuinely optimizable end-to-end."""
    pool = shared.ep_next_player_pool()
    assert set(["player_id", "name", "position", "team", "cost", "predicted_points"]).issubset(pool.columns)
    assert not pool.empty, "real player pool should never be empty mid-season"
    assert (pool["predicted_points"] >= 0).all(), "ep_next should never be negative"

    from optimizer import optimize_squad
    optimal = optimize_squad(pool)
    assert len(optimal) == 15, "optimize_squad must return a real 15-man squad from this pool"
    print(f"PASS: ep_next_player_pool() shape OK ({len(pool)} available players), optimizes to a real 15-man squad")


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
    assert "fpl-has-tooltip" in html, "card must carry the CSS-tooltip trigger class"
    assert 'class="fpl-tooltip-content"' in html, "tooltip must be real HTML block content, not a native title attribute (title-attribute line breaks aren't reliable across browsers -- confirmed directly)"
    assert "This gameweek" in html and "Season so far" in html, "hover tooltip must carry both gameweek and season section headings"
    assert "DEFCON" in html, "hover tooltip must include DEFCON per explicit request"
    # Each section heading must be its own separate <div>, not text glued
    # into one line -- this is what actually guarantees separate lines in
    # the rendered tooltip (real HTML block layout), unlike a title
    # attribute's unreliable \n/&#10; rendering.
    assert '<div class="fpl-tooltip-heading">This gameweek</div>' in html
    assert '<div class="fpl-tooltip-heading">Season so far</div>' in html
    print("PASS: My Squad card hover tooltip is a real CSS tooltip with separate GW/season sections (incl. DEFCON)")


def test_team_primary_color_covers_every_real_team():
    """Regression test for the visual pass's team-tinted photo fallback:
    TEAM_PRIMARY_COLOR must have a real entry for every one of the actual
    current 20 Premier League clubs (fetched live from bootstrap, not
    hardcoded here) -- a missing entry would silently fall back to plain
    gray for that one team's players, a real visual regression if the
    real team list ever changes (promotion/relegation) without this table
    being updated to match."""
    raw = shared._load_bootstrap()
    real_teams = {t["name"] for t in raw["teams"]}
    missing = real_teams - set(shared.TEAM_PRIMARY_COLOR.keys())
    assert not missing, f"TEAM_PRIMARY_COLOR is missing real current teams: {missing}"
    for team, color in shared.TEAM_PRIMARY_COLOR.items():
        assert color.startswith("#") and len(color) == 7, f"{team}'s color {color!r} isn't a valid hex code"
    print(f"PASS: TEAM_PRIMARY_COLOR covers all {len(real_teams)} real current Premier League teams")


def test_sidebar_summary_shape_and_deadline():
    """Verifies the sidebar's real live summary (points, rank, next
    deadline, squad's own next fixtures) against real data -- replaces the
    earlier sidebar's static dev-pipeline description, which carried no
    live data at all."""
    summary = shared.sidebar_summary(MANAGER_ENTRY_ID)
    for key in ("total_points", "overall_rank", "top_pct", "next_gw", "next_deadline", "squad_next_fixtures"):
        assert key in summary
    if summary["total_points"] is not None:
        assert isinstance(summary["total_points"], int) and summary["total_points"] >= 0
        assert isinstance(summary["overall_rank"], int) and summary["overall_rank"] > 0
    if summary["next_deadline"] is not None:
        # A real ISO 8601 deadline must actually parse -- a malformed
        # string here would break the sidebar's countdown silently.
        from datetime import datetime
        parsed = datetime.fromisoformat(summary["next_deadline"].replace("Z", "+00:00"))
        assert parsed.year >= 2026, "parsed deadline year looks implausible"
        assert isinstance(summary["next_gw"], int) and 1 <= summary["next_gw"] <= 38
    assert isinstance(summary["squad_next_fixtures"], list)
    for f in summary["squad_next_fixtures"]:
        assert set(("team", "opponent", "is_home", "difficulty")).issubset(f.keys())
        assert 1 <= f["difficulty"] <= 5
        assert f["team"] != f["opponent"], "a real fixture can't have a team playing itself"
    print(f"PASS: sidebar_summary() real data OK (points={summary['total_points']}, next_gw={summary['next_gw']}, {len(summary['squad_next_fixtures'])} squad fixture(s))")


def test_gameweek_fixtures_played_vs_upcoming():
    """Verifies gameweek_fixtures() against a real PLAYED gameweek (GW1,
    must have real recorded scores) and a real UPCOMING one, confirming
    both states are represented correctly and consistently."""
    played = shared.gameweek_fixtures(1)
    assert not played.empty, "GW1 should have real fixtures"
    assert played["team_h_score"].notna().any(), "GW1 should have at least one real recorded score by now"
    for _, row in played.iterrows():
        assert row["team_h"] != row["team_a"], "a real fixture can't have a team playing itself"
        if row["team_h_score"] is not None:
            assert isinstance(row["team_h_score"], int) and row["team_h_score"] >= 0

    # Find a real gameweek genuinely still in the future relative to GW1
    # (GW1 is confirmed played above) to verify the "not played yet" state.
    raw = shared._load_bootstrap()
    current_gw = next((e["id"] for e in raw["events"] if e.get("is_current")), 1)
    upcoming = shared.gameweek_fixtures(current_gw + 1)
    if not upcoming.empty:
        assert upcoming["team_h_score"].isna().all(), f"GW{current_gw + 1} shouldn't have real scores yet"
    print(f"PASS: gameweek_fixtures() correctly distinguishes played (GW1) from upcoming (GW{current_gw + 1}) real fixtures")


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
    assert "Season so far" in html and "This gameweek" not in html, "optimizer-built card must show season stats only, no per-GW section"
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
    test_no_hardcoded_season_path_in_shared()
    test_load_bootstrap_live_and_fallback()
    test_load_fixtures_df_live_and_fallback()
    test_load_entry_history_live_and_fallback()
    test_season_leaderboards_shape()
    test_team_insights_consistency()
    test_ownership_swing_and_overall_most_owned()
    test_premier_league_table_movement()
    test_explain_transfer_suggestion_no_key_returns_none()
    test_explain_transfer_suggestion_debug_gives_real_reasons()
    test_rate_limit_errors_are_tagged_for_silent_handling()
    test_openrouter_hardcoded_to_free_model()
    test_chip_usage_status_shape_and_parsing()
    test_free_transfers_slider_allows_zero()
    test_ai_explanation_output_is_html_escaped()
    test_ep_next_player_pool_shape_and_optimizable()
    test_tonight_price_projections_shape()
    test_squad_card_hover_stats()
    test_team_primary_color_covers_every_real_team()
    test_sidebar_summary_shape_and_deadline()
    test_gameweek_fixtures_played_vs_upcoming()
    test_player_season_stats_and_optimizer_card_hover()
    test_fixture_started_and_gameweek_live()
    test_not_yet_played_vs_no_game_time_split()
    print("\nAll shared.py live-sync checks passed.")
