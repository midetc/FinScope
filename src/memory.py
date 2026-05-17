"""Пам'ять агента: збереження контексту аналізу та попередніх результатів."""

import json
from datetime import datetime
from pathlib import Path


class AgentMemory:
    def __init__(self, storage_path: str = "output/memory.json"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"sessions": [], "preferences": {}, "last_analysis": None}
        return {"sessions": [], "preferences": {}, "last_analysis": None}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add_session(self, plan: list[str], results: dict) -> None:
        self._data["sessions"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "plan": plan,
                "results": results,
            }
        )
        self._data["last_analysis"] = results
        self.save()

    def get_last_analysis(self) -> dict | None:
        return self._data.get("last_analysis")

    def set_preference(self, key: str, value) -> None:
        self._data["preferences"][key] = value
        self.save()

    def get_preference(self, key: str, default=None):
        return self._data["preferences"].get(key, default)

    def session_count(self) -> int:
        return len(self._data["sessions"])
