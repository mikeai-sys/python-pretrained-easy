# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import re
from src.agent.llm import LocalModel

SCORING_RUBRIC = """
You are a strict grader. Score the candidate solution to the task on a 0.0 to 1.0
scale. Reward correctness, completeness, and clarity. Penalize wrong answers,
hallucinations, and missing steps.

Return ONLY a JSON object: {"score": <float 0-1>, "reason": "<short note>"}
"""


class Evaluator:
    """Scores a (task, solution) pair with a local judge model (LLM-as-judge).

    For domains with reliable ground truth, swap ``score`` for hard test cases
    or unit tests instead of the judge model.
    """

    def __init__(self, model: LocalModel) -> None:
        self.model = model

    def score(self, task: str, solution: str) -> float:
        messages = [
            {"role": "system", "content": SCORING_RUBRIC},
            {"role": "user", "content": f"TASK:\n{task}\n\nSOLUTION:\n{solution}"},
        ]
        raw = self.model.chat(messages, temperature=0.0, max_new_tokens=128)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> float:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0.0
        try:
            obj = __import__("json").loads(match.group(0))
            return float(obj.get("score", 0.0))
        except (ValueError, TypeError):
            return 0.0
