import unittest

from src.analytics import assign_retention_segments, filter_players, segment_summary
from src.data import generate_player_data
from src.modeling import confusion_at_threshold, score_players, train_churn_model


class ChurnModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A sufficiently sized deterministic sample keeps the holdout metrics
        # stable while still making the test suite quick to run in CI.
        cls.data = generate_player_data(n_players=1_800, seed=42)
        cls.training = train_churn_model(cls.data, random_state=42)
        cls.scored = score_players(cls.training.pipeline, cls.data)

    def test_model_has_useful_but_nonperfect_holdout_performance(self):
        auc = self.training.metrics["roc_auc"]
        self.assertGreater(auc, 0.72)
        self.assertLess(auc, 0.99)
        self.assertGreater(self.training.metrics["f1"], 0.55)

    def test_scoring_ranges_and_tiers(self):
        self.assertTrue(self.scored["churn_risk"].between(0, 1).all())
        self.assertTrue(set(self.scored["risk_tier"]).issubset({"Low", "Medium", "High"}))

    def test_confusion_matrix_shape(self):
        confusion = confusion_at_threshold(
            self.training.test_labels,
            self.training.test_probabilities,
            threshold=0.5,
        )
        self.assertEqual(confusion.shape, (2, 2))
        self.assertEqual(int(confusion.to_numpy().sum()), len(self.training.test_labels))

    def test_segments_cover_each_player_once(self):
        segmented = assign_retention_segments(self.scored)
        self.assertEqual(len(segmented), len(self.scored))
        self.assertFalse(segmented["retention_segment"].isna().any())
        summary = segment_summary(segmented)
        self.assertEqual(int(summary["players"].sum()), len(segmented))

    def test_filters_are_applied_together(self):
        filtered = filter_players(
            self.scored,
            platforms=["PC"],
            modes=["Ranked"],
            regions=["APAC"],
            minimum_risk=0.40,
        )
        self.assertTrue((filtered["platform"] == "PC").all())
        self.assertTrue((filtered["preferred_mode"] == "Ranked").all())
        self.assertTrue((filtered["region"] == "APAC").all())
        self.assertTrue((filtered["churn_risk"] >= 0.40).all())


if __name__ == "__main__":
    unittest.main()
