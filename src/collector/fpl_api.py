"""Thin client for the official (free, public, unauthenticated) FPL API.

Endpoints used:
- bootstrap-static: all players, teams, gameweek metadata for the current season
- element-summary/{player_id}: per-gameweek history for one player (current season)
  + summarized history_past for prior seasons
- fixtures: full season fixture list with difficulty ratings
- entry/{entry_id}: a manager's team info and league memberships
- entry/{entry_id}/history: a manager's season-by-season totals (past seasons) and
  gameweek-by-gameweek record for the *current* season only — the public API does not
  expose gameweek-level picks for prior, already-finished seasons
- entry/{entry_id}/event/{gw}/picks: a manager's squad/picks for one gameweek of the
  current season (404s for gameweeks that haven't been played yet)
- leagues-classic/{league_id}/standings: full leaderboard for one classic
  (points-based) mini-league -- the league ids to query come from
  get_entry()'s own leagues.classic list, not guessed
"""

import time
import urllib.error
import urllib.request
import json

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "Mozilla/5.0 (FPL-Analytics data collector; personal project)"


def _get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get_bootstrap_static() -> dict:
    """All players, teams, positions, and gameweek metadata for the current season."""
    return _get_json(f"{BASE_URL}/bootstrap-static/")


def get_player_summary(player_id: int) -> dict:
    """Per-gameweek history for one player: current-season history + prior-season summaries."""
    return _get_json(f"{BASE_URL}/element-summary/{player_id}/")


def get_fixtures() -> list:
    """Full season fixture list, including difficulty ratings."""
    return _get_json(f"{BASE_URL}/fixtures/")


def get_entry(entry_id: int) -> dict:
    """A manager's team info: name, current rank/points, league memberships."""
    return _get_json(f"{BASE_URL}/entry/{entry_id}/")


def get_entry_history(entry_id: int) -> dict:
    """A manager's season-by-season totals (past seasons) and gameweek-by-gameweek
    record for the current season. Prior seasons are season totals only — the public
    API doesn't expose old picks once a season ends."""
    return _get_json(f"{BASE_URL}/entry/{entry_id}/history/")


def get_entry_picks(entry_id: int, gameweek: int) -> dict | None:
    """A manager's squad/picks for one gameweek of the *current* season.
    Returns None (rather than raising) if that gameweek hasn't been played yet."""
    try:
        return _get_json(f"{BASE_URL}/entry/{entry_id}/event/{gameweek}/picks/")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def get_event_live(gameweek: int) -> dict:
    """Every player's stats/points for ONE gameweek, in a single call -- the
    same figures element-summary's per-player `history` rows eventually
    settle into, but available immediately (if provisional) as soon as that
    gameweek's matches are underway, and without N separate per-player
    requests. Used for showing a manager's real squad's per-player point
    breakdown for the CURRENT gameweek without waiting on
    get_all_player_summaries' full per-player fetch."""
    return _get_json(f"{BASE_URL}/event/{gameweek}/live/")


def get_league_standings(league_id: int, page: int = 1) -> dict:
    """One page of a classic (points-based) mini-league's leaderboard --
    league name/metadata plus up to 50 entries per page, ranked by total
    points. Pagination via `page` (FPL's own standings.has_next flag tells
    the caller whether another page exists) -- most private mini-leagues fit
    on one page, so callers checking a specific manager's own small leagues
    typically don't need more than page 1."""
    return _get_json(f"{BASE_URL}/leagues-classic/{league_id}/standings/?page_standings={page}")


def get_all_player_summaries(player_ids: list, delay_seconds: float = 0.3) -> dict:
    """Fetch element-summary for many players, with a small delay between requests
    to be a reasonable API citizen. Returns {player_id: summary_dict}."""
    summaries = {}
    for i, pid in enumerate(player_ids):
        summaries[pid] = get_player_summary(pid)
        if delay_seconds and i < len(player_ids) - 1:
            time.sleep(delay_seconds)
    return summaries
