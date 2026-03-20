"""Local run-state persistence for live rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunStore:
    """Filesystem layout for one active round."""

    root: Path

    @classmethod
    def for_round(cls, workspace_root: Path, round_id: str) -> "RunStore":
        runs_root = workspace_root / "runs"
        plain_root = runs_root / round_id
        if plain_root.exists():
            return cls(plain_root)

        labeled_matches = sorted(runs_root.glob(f"{round_id} (*)"))
        if labeled_matches:
            return cls(labeled_matches[0])

        return cls(plain_root)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        self.ensure()
        return self.root / name

    def save_json(self, name: str, payload: Any) -> Path:
        path = self._path(name)
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        return path

    def load_json(self, name: str, default: Any = None) -> Any:
        path = self._path(name)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def append_observation(self, payload: dict[str, Any]) -> Path:
        path = self._path("observations.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return path

    def load_observations(self) -> list[dict[str, Any]]:
        path = self._path("observations.jsonl")
        if not path.exists():
            return []
        observations: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    observations.append(json.loads(line))
        return observations

    def save_prediction(self, seed_index: int, payload: list[list[list[float]]]) -> Path:
        return self.save_json(f"prediction_seed_{seed_index}.json", payload)

    def load_prediction(self, seed_index: int) -> list[list[list[float]]] | None:
        return self.load_json(f"prediction_seed_{seed_index}.json")
