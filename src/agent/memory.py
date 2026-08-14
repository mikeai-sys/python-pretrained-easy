# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Memory:
    """Append-only store of (task, solution, score) examples as JSONL.

    This is the dataset the trainer learns from. Keep it simple and durable —
    swap for a vector DB or SQLite if you outgrow JSONL.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, task: str, solution: str, score: float, meta: dict | None = None) -> None:
        record = {
            "task": task,
            "solution": solution,
            "score": score,
            "meta": meta or {},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def good_examples(self, threshold: float) -> list[dict[str, Any]]:
        return [r for r in self.load() if r.get("score", 0) >= threshold]

    def count(self) -> int:
        return len(self.load())
