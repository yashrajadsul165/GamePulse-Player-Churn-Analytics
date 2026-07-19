import unittest
from pathlib import Path


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - handled by conditional skip
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_starts_without_exception(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=120)
        self.assertEqual(len(app.exception), 0)
        metric_labels = {metric.label for metric in app.metric}
        self.assertIn("Players", metric_labels)
        self.assertIn("Observed 30-day churn", metric_labels)
        self.assertIn("High-risk players", metric_labels)


if __name__ == "__main__":
    unittest.main()
