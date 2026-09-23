import json
from pathlib import Path


def save_metrics(metrics: dict, output_path: str):
    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)