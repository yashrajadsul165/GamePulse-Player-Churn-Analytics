import tempfile
import unittest
from pathlib import Path

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_player_data, load_player_data


class PlayerDataTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        first = generate_player_data(n_players=250, seed=7)
        second = generate_player_data(n_players=250, seed=7)
        self.assertTrue(first.equals(second))

    def test_schema_and_ranges(self):
        data = generate_player_data(n_players=500, seed=11)
        self.assertEqual(len(data), 500)
        self.assertEqual(data["player_id"].nunique(), 500)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(data.columns))
        self.assertTrue(data[TARGET_COLUMN].isin([0, 1]).all())
        self.assertTrue(data["churned_30d"].mean() > 0.15)
        self.assertTrue(data["churned_30d"].mean() < 0.65)
        self.assertTrue(data["engagement_score"].between(0, 100).all())
        self.assertTrue(data["win_rate"].between(0, 1).all())

    def test_loader_rejects_missing_columns(self):
        data = generate_player_data(n_players=120, seed=5).drop(columns=["sessions_30d"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.csv"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "sessions_30d"):
                load_player_data(path)


if __name__ == "__main__":
    unittest.main()
