# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f) or {})

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def api_key(self, env_name: str) -> str | None:
        return os.environ.get(env_name)


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (if present)."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config.yaml") -> Config:
    _load_dotenv()
    return Config.load(path)
