"""Regenerate the committed synthetic GamePulse dataset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import generate_player_data  # noqa: E402


def main() -> None:
    output = ROOT / "data" / "gamepulse_players.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_player_data(n_players=3_000, seed=42)
    data.to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Wrote {len(data):,} synthetic players to {output}")
    print(f"Observed 30-day churn rate: {data['churned_30d'].mean():.1%}")


if __name__ == "__main__":
    main()
