from __future__ import annotations

import unittest

from tests.unit.helpers import load_script_module, run_script_json


class EloFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script_module("matchmaking_simulator.py")

    def test_elo_expected_equal_ratings(self) -> None:
        self.assertAlmostEqual(self.mod.elo_expected(1500, 1500), 0.5, places=6)

    def test_elo_expected_400_gap(self) -> None:
        e = self.mod.elo_expected(1900, 1500)
        self.assertAlmostEqual(e, 10 / 11, places=4)

    def test_elo_update_win(self) -> None:
        result = self.mod.elo_update(1500, 1400, 1.0, 32)
        self.assertGreater(result["delta"], 0)
        self.assertAlmostEqual(result["new_rating"], 1500 + result["delta"], places=6)

    def test_elo_update_loss(self) -> None:
        result = self.mod.elo_update(1500, 1400, 0.0, 32)
        self.assertLess(result["delta"], 0)

    def test_elo_update_zero_sum(self) -> None:
        a = self.mod.elo_update(1500, 1400, 1.0, 32)
        b = self.mod.elo_update(1400, 1500, 0.0, 32)
        self.assertAlmostEqual(a["delta"] + b["delta"], 0.0, places=6)


class MatchQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script_module("matchmaking_simulator.py")

    def test_perfect_quality(self) -> None:
        r = self.mod.match_quality(1500, 1500)
        self.assertAlmostEqual(r["quality"], 1.0, places=6)

    def test_zero_quality(self) -> None:
        r = self.mod.match_quality(1500, 1100, max_gap=400)
        self.assertAlmostEqual(r["quality"], 0.0, places=6)

    def test_quality_linear(self) -> None:
        r = self.mod.match_quality(1500, 1300, max_gap=400)
        self.assertAlmostEqual(r["quality"], 0.5, places=6)


class SimulateSeasonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script_module("matchmaking_simulator.py")

    def test_num_players_too_small(self) -> None:
        with self.assertRaisesRegex(ValueError, r"num_players must be >= 2"):
            self.mod.simulate_season(num_players=1)

    def test_invalid_match_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, r"match_strategy must be"):
            self.mod.simulate_season(match_strategy="invalid")

    def test_negative_match_range(self) -> None:
        with self.assertRaisesRegex(ValueError, r"match_range must be >= 0"):
            self.mod.simulate_season(match_range=-1)

    def test_random_strategy_output_keys(self) -> None:
        r = self.mod.simulate_season(
            num_players=20, num_matches=100, match_strategy="random"
        )
        for key in (
            "num_players", "num_matches", "k", "match_strategy",
            "avg_games_per_player", "avg_rating_error", "avg_match_quality",
            "rating_p10", "rating_p50", "rating_p90",
        ):
            self.assertIn(key, r)
        self.assertEqual(r["match_strategy"], "random")

    def test_skill_based_strategy_runs(self) -> None:
        r = self.mod.simulate_season(
            num_players=20, num_matches=100, match_strategy="skill_based", match_range=200
        )
        self.assertEqual(r["match_strategy"], "skill_based")
        self.assertGreater(r["avg_match_quality"], 0)


class RankDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_script_module("matchmaking_simulator.py")

    def test_default_tiers_population_sums_to_100(self) -> None:
        rows = self.mod.rank_distribution()
        total = sum(r["population_pct"] for r in rows)
        self.assertAlmostEqual(total, 100.0, places=6)

    def test_boundaries_are_ascending(self) -> None:
        rows = self.mod.rank_distribution()
        boundaries = [r["rating_upper"] for r in rows if r["rating_upper"] is not None]
        for i in range(1, len(boundaries)):
            self.assertGreater(boundaries[i], boundaries[i - 1])

    def test_non_ascending_percentile_raises(self) -> None:
        tiers = [
            {"name": "A", "percentile_upper": 60},
            {"name": "B", "percentile_upper": 30},
        ]
        with self.assertRaisesRegex(ValueError, r"strictly ascending"):
            self.mod.rank_distribution(tiers=tiers)

    def test_duplicate_percentile_raises(self) -> None:
        tiers = [
            {"name": "A", "percentile_upper": 50},
            {"name": "B", "percentile_upper": 50},
            {"name": "C", "percentile_upper": 100},
        ]
        with self.assertRaisesRegex(ValueError, r"strictly ascending"):
            self.mod.rank_distribution(tiers=tiers)

    def test_last_percentile_not_100_raises(self) -> None:
        tiers = [
            {"name": "A", "percentile_upper": 30},
            {"name": "B", "percentile_upper": 80},
        ]
        with self.assertRaisesRegex(ValueError, r"last percentile_upper must be 100"):
            self.mod.rank_distribution(tiers=tiers)

    def test_percentile_out_of_range_raises(self) -> None:
        tiers = [{"name": "A", "percentile_upper": 120}]
        with self.assertRaisesRegex(ValueError, r"\(0, 100\]"):
            self.mod.rank_distribution(tiers=tiers)

    def test_percentile_zero_raises(self) -> None:
        tiers = [{"name": "A", "percentile_upper": 0}]
        with self.assertRaisesRegex(ValueError, r"\(0, 100\]"):
            self.mod.rank_distribution(tiers=tiers)

    def test_tier_missing_name_raises(self) -> None:
        tiers = [{"percentile_upper": 100}]
        with self.assertRaisesRegex(ValueError, r"tiers\[0\]\.name is required"):
            self.mod.rank_distribution(tiers=tiers)


class ModeValidationTests(unittest.TestCase):
    def test_unknown_mode_exits_nonzero(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"failed"):
            run_script_json("matchmaking_simulator.py", payload={"mode": "typo"})

    def test_all_mode_returns_all_sections(self) -> None:
        out = run_script_json("matchmaking_simulator.py")
        for key in ("elo_update", "match_quality", "simulate_season", "rank_distribution"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
