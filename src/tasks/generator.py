# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import random
from src.config import Config


class TaskGenerator:
    """Produces tasks for the agent to solve.

    Start from a seed pool; later, generate new tasks from web searches or a
    curriculum. Swap ``next()`` for a web-search-driven generator when ready.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.seed_pool = [
            "Explain the difference between fine-tuning and RAG in one paragraph.",
            "Write a Python function that flattens a nested list of arbitrary depth.",
            "Summarize the key steps of an autonomous AI data-flywheel loop.",
            "Given the numbers 3, 1, 4, 1, 5, 9, compute the mean and median.",
            "Describe one failure mode of self-improving agents and how to mitigate it.",
        ]

    def next(self) -> str:
        # TODO: optionally pull a real web-search-derived task here.
        return random.choice(self.seed_pool)
