"""Minimal standard-library client for the Astar Island API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ApiError(RuntimeError):
    """Raised when the remote API returns an error."""


@dataclass
class AstarIslandClient:
    """Small wrapper around the REST API."""

    token: str | None = None
    base_url: str = "https://api.ainm.no/astar-island"
    timeout: float = 30.0
    retries: int = 3
    backoff_seconds: float = 0.75

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("AINM_TOKEN")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_exception: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < self.retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    last_exception = exc
                    continue
                try:
                    detail = json.loads(body)
                except json.JSONDecodeError:
                    detail = body
                raise ApiError(f"{method.upper()} {path} failed with {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    last_exception = exc
                    continue
                raise ApiError(f"{method.upper()} {path} failed: {exc.reason}") from exc
        else:
            raise ApiError(f"{method.upper()} {path} failed after retries: {last_exception}")

        if not raw.strip():
            return None
        return json.loads(raw)

    def get_rounds(self) -> list[dict[str, Any]]:
        return self._request("GET", "rounds")

    def get_active_round(self) -> dict[str, Any]:
        rounds = self.get_rounds()
        for round_item in rounds:
            if round_item.get("status") == "active":
                return round_item
        raise ApiError("No active round found.")

    def get_round_detail(self, round_id: str) -> dict[str, Any]:
        return self._request("GET", f"rounds/{round_id}")

    def get_budget(self) -> dict[str, Any]:
        return self._request("GET", "budget")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> dict[str, Any]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        return self._request("POST", "simulate", payload)

    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict[str, Any]:
        payload = {"round_id": round_id, "seed_index": seed_index, "prediction": prediction}
        return self._request("POST", "submit", payload)

    def get_my_rounds(self) -> list[dict[str, Any]]:
        return self._request("GET", "my-rounds")

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        return self._request("GET", f"analysis/{round_id}/{seed_index}")
