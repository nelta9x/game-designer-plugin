#!/usr/bin/env python3
"""Matchmaking simulator for balance iteration.

Usage
-----
# Default sample (markdown):
  python3 matchmaking_simulator.py

# JSON output:
  python3 matchmaking_simulator.py --format json

# Custom input:
  python3 matchmaking_simulator.py --input params.json

Input JSON schema
-----------------
{
  "mode": "all" | "elo_update" | "match_quality" | "simulate_season" | "rank_distribution",

  // elo_update
  "matches": [
    {"player": 1500, "opponent": 1400, "result": 1, "k": 32}
  ],

  // match_quality
  "pairs": [
    {"r1": 1500, "r2": 1400, "max_gap": 400}
  ],

  // simulate_season
  "season": {
    "num_players": 200,
    "num_matches": 2000,
    "k": 24,
    "start_rating": 1500,
    "start_spread": 300,
    "match_strategy": "random" | "skill_based",
    "match_range": 200
  },

  // rank_distribution
  "distribution": {
    "mean": 1500,
    "std": 300,
    "tiers": [
      {"name": "Bronze",   "percentile_upper": 10},
      {"name": "Silver",   "percentile_upper": 35},
      {"name": "Gold",     "percentile_upper": 65},
      {"name": "Platinum", "percentile_upper": 85},
      {"name": "Diamond",  "percentile_upper": 95},
      {"name": "Master",   "percentile_upper": 99},
      {"name": "GM",       "percentile_upper": 100}
    ]
  }
}
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS = 1e-9


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def elo_expected(r_me: float, r_opp: float) -> float:
    """Expected score (0-1) for player with rating r_me vs r_opp."""
    return 1.0 / (1.0 + math.pow(10, (r_opp - r_me) / 400.0))


def elo_update(r: float, opponent_r: float, result: float, k: float = 32.0) -> Dict[str, float]:
    """Return new rating and delta after a single match."""
    expected = elo_expected(r, opponent_r)
    delta = k * (result - expected)
    return {
        "old_rating": r,
        "opponent": opponent_r,
        "expected": expected,
        "result": result,
        "k": k,
        "delta": delta,
        "new_rating": r + delta,
    }


def match_quality(r1: float, r2: float, max_gap: float = 400.0) -> Dict[str, float]:
    """Quality metric (0-1) for a match between two ratings."""
    gap = abs(r1 - r2)
    quality = max(0.0, 1.0 - gap / max(max_gap, EPS))
    expected_r1 = elo_expected(r1, r2)
    return {
        "r1": r1,
        "r2": r2,
        "gap": gap,
        "quality": quality,
        "expected_winrate_r1": expected_r1,
    }


def simulate_season(
    num_players: int = 200,
    num_matches: int = 2000,
    k: float = 24.0,
    start_rating: float = 1500.0,
    start_spread: float = 300.0,
    match_strategy: str = "random",
    match_range: float = 200.0,
) -> Dict[str, Any]:
    """Simulate a season: assign true skill, run matches, report convergence.

    match_strategy:
        "random"      — uniform random pairing (convergence test)
        "skill_based" — prefer opponents within ±match_range of rating
    """
    if num_players < 2:
        raise ValueError(f"num_players must be >= 2, got {num_players}")
    if match_strategy not in ("random", "skill_based"):
        raise ValueError(
            f"match_strategy must be 'random' or 'skill_based', got '{match_strategy}'"
        )
    if match_range < 0:
        raise ValueError(f"match_range must be >= 0, got {match_range}")

    true_skill = [random.gauss(start_rating, start_spread) for _ in range(num_players)]
    ratings = [start_rating] * num_players
    games_played = [0] * num_players

    total_quality = 0.0

    for _ in range(num_matches):
        if match_strategy == "skill_based":
            i = random.randrange(num_players)
            best_j = -1
            best_dist = float("inf")
            in_range: List[int] = []
            for c in range(num_players):
                if c == i:
                    continue
                dist = abs(ratings[c] - ratings[i])
                if dist <= match_range:
                    in_range.append(c)
                if dist < best_dist:
                    best_dist = dist
                    best_j = c
            j = random.choice(in_range) if in_range else best_j
        else:
            i, j = random.sample(range(num_players), 2)

        # determine winner by true skill difference
        p_i_wins = elo_expected(true_skill[i], true_skill[j])
        result_i = 1.0 if random.random() < p_i_wins else 0.0

        gap = abs(ratings[i] - ratings[j])
        total_quality += max(0.0, 1.0 - gap / 400.0)

        upd_i = elo_update(ratings[i], ratings[j], result_i, k)
        upd_j = elo_update(ratings[j], ratings[i], 1.0 - result_i, k)
        ratings[i] = upd_i["new_rating"]
        ratings[j] = upd_j["new_rating"]
        games_played[i] += 1
        games_played[j] += 1

    # convergence: mean absolute error between true skill and final rating
    errors = [abs(true_skill[p] - ratings[p]) for p in range(num_players)]
    avg_error = sum(errors) / num_players

    sorted_ratings = sorted(ratings)
    p10 = sorted_ratings[int(num_players * 0.10)]
    p50 = sorted_ratings[int(num_players * 0.50)]
    p90 = sorted_ratings[int(num_players * 0.90)]
    avg_games = sum(games_played) / num_players

    return {
        "num_players": num_players,
        "num_matches": num_matches,
        "k": k,
        "match_strategy": match_strategy,
        "avg_games_per_player": avg_games,
        "avg_rating_error": avg_error,
        "avg_match_quality": total_quality / max(num_matches, 1),
        "rating_p10": p10,
        "rating_p50": p50,
        "rating_p90": p90,
    }


def rank_distribution(
    mean: float = 1500.0,
    std: float = 300.0,
    tiers: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Compute rating boundaries for each tier from percentile targets."""
    if tiers is None:
        tiers = [
            {"name": "Bronze", "percentile_upper": 10},
            {"name": "Silver", "percentile_upper": 35},
            {"name": "Gold", "percentile_upper": 65},
            {"name": "Platinum", "percentile_upper": 85},
            {"name": "Diamond", "percentile_upper": 95},
            {"name": "Master", "percentile_upper": 99},
            {"name": "GM", "percentile_upper": 100},
        ]

    prev_pct = 0.0
    for idx, tier in enumerate(tiers):
        name = tier.get("name")
        if not name:
            raise ValueError(f"tiers[{idx}].name is required")
        pct = to_float(tier.get("percentile_upper", 100))
        if not (0 < pct <= 100):
            raise ValueError(
                f"percentile_upper must be in (0, 100]: "
                f"'{name}' has {pct}"
            )
        if pct <= prev_pct:
            raise ValueError(
                f"percentile_upper must be strictly ascending: "
                f"'{name}' ({pct}) <= previous ({prev_pct})"
            )
        prev_pct = pct

    if prev_pct != 100:
        raise ValueError(f"last percentile_upper must be 100 (got {prev_pct})")

    results: List[Dict[str, Any]] = []
    prev_upper = 0
    prev_boundary = float("-inf")

    for tier in tiers:
        pct_upper = to_float(tier.get("percentile_upper", 100))
        population_pct = pct_upper - prev_upper

        if pct_upper >= 100:
            boundary = float("inf")
        else:
            z = _norm_ppf(pct_upper / 100.0)
            boundary = mean + z * std

        results.append({
            "tier": tier["name"],
            "population_pct": population_pct,
            "rating_lower": prev_boundary if prev_boundary != float("-inf") else None,
            "rating_upper": boundary if boundary != float("inf") else None,
        })
        prev_upper = pct_upper
        prev_boundary = boundary

    return results


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_norm_ppf_inner(1.0 - p)
    return _norm_ppf_inner(p)


def _norm_ppf_inner(p: float) -> float:
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_elo_md(rows: List[Dict[str, float]]) -> str:
    lines = ["## Elo Update Results", ""]
    lines.append("| Player | Opponent | Expected | Result | K | Delta | New Rating |")
    lines.append("|--------|----------|----------|--------|---|-------|------------|")
    for r in rows:
        lines.append(
            f"| {_fmt(r['old_rating'],0)} | {_fmt(r['opponent'],0)} "
            f"| {_fmt(r['expected'],3)} | {_fmt(r['result'],1)} "
            f"| {_fmt(r['k'],0)} | {_fmt(r['delta'],1)} "
            f"| {_fmt(r['new_rating'],1)} |"
        )
    return "\n".join(lines)


def render_quality_md(rows: List[Dict[str, float]]) -> str:
    lines = ["## Match Quality", ""]
    lines.append("| R1 | R2 | Gap | Quality | Expected WR (R1) |")
    lines.append("|----|----|----|---------|-------------------|")
    for r in rows:
        lines.append(
            f"| {_fmt(r['r1'],0)} | {_fmt(r['r2'],0)} "
            f"| {_fmt(r['gap'],0)} | {_fmt(r['quality'],3)} "
            f"| {_fmt(r['expected_winrate_r1'],3)} |"
        )
    return "\n".join(lines)


def render_season_md(result: Dict[str, Any]) -> str:
    lines = ["## Season Simulation", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Players | {result['num_players']} |")
    lines.append(f"| Total Matches | {result['num_matches']} |")
    lines.append(f"| K-factor | {result['k']} |")
    lines.append(f"| Match Strategy | {result['match_strategy']} |")
    lines.append(f"| Avg Games/Player | {_fmt(result['avg_games_per_player'],1)} |")
    lines.append(f"| Avg Rating Error | {_fmt(result['avg_rating_error'],1)} |")
    lines.append(f"| Avg Match Quality | {_fmt(result['avg_match_quality'],3)} |")
    lines.append(f"| Rating p10 | {_fmt(result['rating_p10'],0)} |")
    lines.append(f"| Rating p50 | {_fmt(result['rating_p50'],0)} |")
    lines.append(f"| Rating p90 | {_fmt(result['rating_p90'],0)} |")
    return "\n".join(lines)


def render_rank_md(rows: List[Dict[str, Any]]) -> str:
    lines = ["## Rank Distribution", ""]
    lines.append("| Tier | Population % | Rating Lower | Rating Upper |")
    lines.append("|------|-------------|-------------|-------------|")
    for r in rows:
        lower = _fmt(r["rating_lower"], 0) if r["rating_lower"] is not None else "-"
        upper = _fmt(r["rating_upper"], 0) if r["rating_upper"] is not None else "-"
        lines.append(f"| {r['tier']} | {_fmt(r['population_pct'],1)}% | {lower} | {upper} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default input
# ---------------------------------------------------------------------------

DEFAULT_INPUT: Dict[str, Any] = {
    "mode": "all",
    "matches": [
        {"player": 1500, "opponent": 1400, "result": 1, "k": 32},
        {"player": 1500, "opponent": 1600, "result": 0, "k": 32},
        {"player": 1200, "opponent": 1800, "result": 1, "k": 32},
    ],
    "pairs": [
        {"r1": 1500, "r2": 1500, "max_gap": 400},
        {"r1": 1500, "r2": 1400, "max_gap": 400},
        {"r1": 1500, "r2": 1300, "max_gap": 400},
        {"r1": 1500, "r2": 1100, "max_gap": 400},
    ],
    "season": {
        "num_players": 200,
        "num_matches": 2000,
        "k": 24,
        "start_rating": 1500,
        "start_spread": 300,
    },
    "distribution": {
        "mean": 1500,
        "std": 300,
        "tiers": [
            {"name": "Bronze", "percentile_upper": 10},
            {"name": "Silver", "percentile_upper": 35},
            {"name": "Gold", "percentile_upper": 65},
            {"name": "Platinum", "percentile_upper": 85},
            {"name": "Diamond", "percentile_upper": 95},
            {"name": "Master", "percentile_upper": 99},
            {"name": "GM", "percentile_upper": 100},
        ],
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Matchmaking simulator")
    parser.add_argument("--input", type=str, help="Input JSON file path")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = DEFAULT_INPUT

    mode = data.get("mode", "all")
    valid_modes = {"all", "elo_update", "match_quality", "simulate_season", "rank_distribution"}
    if mode not in valid_modes:
        parser.error(f"unknown mode '{mode}' in input. Valid: {', '.join(sorted(valid_modes))}")

    outputs: Dict[str, Any] = {}
    md_parts: List[str] = []

    # elo_update
    if mode in ("all", "elo_update"):
        matches = data.get("matches", [])
        rows = [
            elo_update(
                to_float(m.get("player"), 1500),
                to_float(m.get("opponent"), 1500),
                to_float(m.get("result"), 0.5),
                to_float(m.get("k"), 32),
            )
            for m in matches
        ]
        outputs["elo_update"] = rows
        md_parts.append(render_elo_md(rows))

    # match_quality
    if mode in ("all", "match_quality"):
        pairs = data.get("pairs", [])
        rows = [
            match_quality(
                to_float(p.get("r1"), 1500),
                to_float(p.get("r2"), 1500),
                to_float(p.get("max_gap"), 400),
            )
            for p in pairs
        ]
        outputs["match_quality"] = rows
        md_parts.append(render_quality_md(rows))

    # simulate_season
    if mode in ("all", "simulate_season"):
        season = data.get("season", {})
        result = simulate_season(
            num_players=to_int(season.get("num_players"), 200),
            num_matches=to_int(season.get("num_matches"), 2000),
            k=to_float(season.get("k"), 24),
            start_rating=to_float(season.get("start_rating"), 1500),
            start_spread=to_float(season.get("start_spread"), 300),
            match_strategy=season.get("match_strategy", "random"),
            match_range=to_float(season.get("match_range"), 200),
        )
        outputs["simulate_season"] = result
        md_parts.append(render_season_md(result))

    # rank_distribution
    if mode in ("all", "rank_distribution"):
        dist = data.get("distribution", {})
        rows = rank_distribution(
            mean=to_float(dist.get("mean"), 1500),
            std=to_float(dist.get("std"), 300),
            tiers=dist.get("tiers"),
        )
        outputs["rank_distribution"] = rows
        md_parts.append(render_rank_md(rows))

    if args.format == "json":
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
    else:
        print("\n\n".join(md_parts))


if __name__ == "__main__":
    main()
