"""FPL chip-timing advisor: given a squad and per-gameweek player/team
projections, suggest good windows for each chip.

Chip rules verified against the live FPL API (bootstrap-static's `chips` list,
2026-27 season), not assumed -- each of the four chips is usable ONCE PER HALF
of the season (gameweeks 1-19 and 20-38 this season; the exact boundary gameweek
is fixed by FPL each year via each chip's start_event/stop_event, not a literal
calendar date like "December"). This module doesn't enforce chip availability
itself (that's a season-long bookkeeping concern for the caller, same as
optimize_transfers' free_transfers parameter) -- it scores candidate gameweeks
so a manager can decide when to spend a chip they still have.

Chip logic:
- Bench Boost: best when the BENCH (non-starters) has an unusually strong
  upcoming gameweek, since bench points normally don't count.
- Triple Captain: best on the single highest-projected player's easiest,
  highest-ceiling fixture (captain already doubles points; this triples them).
- Free Hit: best for a gameweek where the current squad's total projection is
  unusually low relative to a fresh optimal squad for that gameweek -- a one-week
  full rebuild recovers the gap. Requires a per-gameweek optimal-squad
  projection to compare against, not just the current squad's own fixture run.
- Wildcard: no fully automated suggestion here -- wildcard timing is a
  longer-horizon strategic call (before a good run of fixtures begins across
  MULTIPLE weeks) that this module doesn't have enough forward-looking
  structure to resolve confidently. Returns the same "your squad's projection
  is below what's achievable" signal as Free Hit for now, flagged as such.
"""

from dataclasses import dataclass

import pandas as pd

CHIP_HALVES = {
    "first_half": (1, 19),
    "second_half": (20, 38),
}


@dataclass
class ChipSuggestion:
    chip: str
    gameweek: int
    score: float
    detail: str


def suggest_bench_boost(
    squad: pd.DataFrame,
    future_points_by_gw: dict,
    gw_range: range,
) -> list:
    """Rank candidate gameweeks by total predicted points on the BENCH (the 4
    players not in in_starting_xi), since that's the value Bench Boost unlocks.

    `future_points_by_gw`: {gw: {player_id: predicted_points}} for each gw in
    gw_range -- the caller supplies these (e.g. from re-running the trained
    model's predictions for each future fixture), since projecting multiple
    gameweeks ahead is a modeling concern outside this optimizer module."""
    bench_ids = squad.loc[~squad["in_starting_xi"], "player_id"].tolist()
    if len(bench_ids) != 4:
        raise ValueError(f"Expected 4 bench players, got {len(bench_ids)} -- is 'squad' a full 15-man squad with in_starting_xi set?")

    suggestions = []
    for gw in gw_range:
        gw_points = future_points_by_gw.get(gw, {})
        bench_total = sum(gw_points.get(pid, 0.0) for pid in bench_ids)
        suggestions.append(ChipSuggestion(
            chip="bench_boost",
            gameweek=gw,
            score=bench_total,
            detail=f"Bench predicted points: {bench_total:.1f}",
        ))
    return sorted(suggestions, key=lambda s: s.score, reverse=True)


def suggest_triple_captain(
    squad: pd.DataFrame,
    future_points_by_gw: dict,
    gw_range: range,
) -> list:
    """Rank candidate gameweeks by the single highest predicted-points STARTER
    that gameweek -- Triple Captain multiplies one player's score by 3 instead
    of the normal captaincy's 2, so the best use is your single best player's
    single best upcoming fixture, not necessarily who's captain-eligible today
    (form/fixtures change week to week)."""
    starter_ids = squad.loc[squad["in_starting_xi"], "player_id"].tolist()

    suggestions = []
    for gw in gw_range:
        gw_points = future_points_by_gw.get(gw, {})
        starter_scores = {pid: gw_points.get(pid, 0.0) for pid in starter_ids}
        if not starter_scores:
            continue
        best_id = max(starter_scores, key=starter_scores.get)
        best_score = starter_scores[best_id]
        # Extra value over a normal (2x) captaincy of the same player.
        extra_value = best_score  # the 3rd multiple, i.e. (3x - 2x) * points
        name = squad.loc[squad["player_id"] == best_id, "name"].iloc[0] if "name" in squad.columns else best_id
        suggestions.append(ChipSuggestion(
            chip="triple_captain",
            gameweek=gw,
            score=extra_value,
            detail=f"Best starter: {name} ({best_score:.1f} pts projected, "
                    f"+{extra_value:.1f} extra vs normal captaincy)",
        ))
    return sorted(suggestions, key=lambda s: s.score, reverse=True)


def suggest_free_hit_or_wildcard(
    squad: pd.DataFrame,
    future_points_by_gw: dict,
    optimal_points_by_gw: dict,
    gw_range: range,
    chip: str = "free_hit",
) -> list:
    """Rank candidate gameweeks by the gap between the current squad's total
    predicted points and a freshly optimized squad's predicted points for that
    same gameweek -- the bigger the gap, the more a one-week (Free Hit) or
    permanent (Wildcard) rebuild is worth.

    `optimal_points_by_gw`: {gw: total_points} from re-running optimize_squad()
    against that gameweek's player pool -- the caller supplies this, since it
    requires running the solver per candidate gameweek, which this function
    doesn't do itself to avoid silently making N solver calls per suggestion.

    NOTE: for Wildcard specifically, this only captures a single gameweek's
    gap, not the multi-week strategic value a permanent squad change unlocks --
    treat a Wildcard suggestion from this function as a weaker signal than the
    Free Hit one, and prefer a real multi-gameweek lookahead once one exists
    (see this module's docstring)."""
    if chip not in ("free_hit", "wildcard"):
        raise ValueError(f"chip must be 'free_hit' or 'wildcard', got {chip!r}")

    squad_ids = squad["player_id"].tolist()

    suggestions = []
    for gw in gw_range:
        gw_points = future_points_by_gw.get(gw, {})
        current_total = sum(gw_points.get(pid, 0.0) for pid in squad_ids)
        optimal_total = optimal_points_by_gw.get(gw)
        if optimal_total is None:
            continue
        gap = optimal_total - current_total
        suggestions.append(ChipSuggestion(
            chip=chip,
            gameweek=gw,
            score=gap,
            detail=f"Current squad: {current_total:.1f} pts, optimal squad: "
                    f"{optimal_total:.1f} pts, gap: {gap:.1f}",
        ))
    return sorted(suggestions, key=lambda s: s.score, reverse=True)
