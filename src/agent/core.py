# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.llm import LocalModel


class ToolRegistry:
    """Holds the tools the actor may call. Stubbed offline; wire real APIs here."""

    def __init__(self, config) -> None:
        self.config = config
        # TODO: import and register real tools from src/agent/tools
        self.tools: dict[str, object] = {}

    def describe(self) -> list[dict[str, str]]:
        return [{"name": n, "description": getattr(t, "__doc__", "")} for n, t in self.tools.items()]

    def run(self, name: str, **kwargs):
        tool = self.tools.get(name)
        if tool is None:
            return f"[tool '{name}' not available]"
        return tool(**kwargs)  # type: ignore[operator]


ACT_SYSTEM = (
    "You are a capable problem-solving agent. Use the provided tools when needed "
    "to find current information, then answer the task precisely."
)


class Actor:
    """Solves a task using a local model, optionally augmented with tools."""

    def __init__(self, model: LocalModel, tools: ToolRegistry) -> None:
        self.model = model
        self.tools = tools

    def solve(self, task: str) -> str:
        messages = [
            {"role": "system", "content": ACT_SYSTEM},
            {"role": "user", "content": task},
        ]
        return self.model.chat(messages)
