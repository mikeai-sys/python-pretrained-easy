# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

from src.config import Config
from src.agent.memory import Memory
from src.agent.core import Actor, ToolRegistry
from src.agent.data_collector import DataCollector
from src.tasks.generator import TaskGenerator


class SelfImprovingAgent:
    """Drives the closed loop: task -> act -> evaluate -> remember -> train.

    All inference and training happen locally on your GPU/CPU. The local model is
    loaded lazily — image/video crawls and data cleanup never touch it.

    Optional web data collection (search + crawl + synthesize dataset) can feed
    the loop, and the crawled data is deleted from disk once training has consumed
    it (via ``--delete``).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.memory = Memory(config.get("paths", "memory", default="data/memory.jsonl"))
        self.tools = ToolRegistry(config)
        self.tasks = TaskGenerator(config)
        # Collector is created without a model; attached only when text synthesis
        # (which needs the model) is actually requested.
        self.collector = DataCollector(config, None)
        self.threshold = config.get("loop", "eval_threshold", default=0.7)
        self.max_retries = config.get("loop", "max_retries", default=2)
        self._model_ready = False

    def _ensure_model(self) -> None:
        """Load the local model + dependent components on first use."""
        if self._model_ready:
            return
        try:
            from src.agent.llm import LocalModel
            from src.agent.evaluator import Evaluator
            from src.agent.trainer import Trainer
        except ModuleNotFoundError as exc:  # noqa: BLE001
            print(
                "[ppte] This command needs the ML toolchain (torch), which is not installed. "
                "Run `ppte install torch` (or `ppte install txt` for the full LLM stack)."
            )
            raise SystemExit(1) from exc

        self.actor_model = LocalModel(self.config, "actor")
        judge_cfg = self.config.raw.get("judge", {})
        self.judge_model = (
            self.actor_model if not judge_cfg.get("model_id") else LocalModel(self.config, "judge")
        )
        self.evaluator = Evaluator(self.judge_model)
        self.actor = Actor(self.actor_model, self.tools)
        self.trainer = Trainer(self.config, self.memory, self.actor_model)
        self.collector.model = self.actor_model
        self._model_ready = True

    def run(self, iterations: int | None = None) -> None:
        iterations = iterations or self.config.get("loop", "iterations", default=10)
        dc = self.config.raw.get("data_collection", {})
        crawl_runs: list[str] = []

        if dc.get("enabled") and dc.get("queries"):
            self._ensure_model()
            for q in dc["queries"]:
                examples, run_id = self.collector.collect(q, dataset_type=dc.get("dataset_type", "text"))
                for ex in examples:
                    self.memory.add(ex["task"], ex["solution"], 1.0, ex.get("meta"))
                crawl_runs.append(run_id)
            self.trainer.maybe_train(force=True)

        self._ensure_model()
        for i in range(iterations):
            task = self.tasks.next()
            solution, score = self._attempt(task)
            if score >= self.threshold:
                self.memory.add(task, solution, score)
                print(f"[iter {i}] kept   score={score:.2f}  mem={self.memory.count()}")
            else:
                print(f"[iter {i}] dropped score={score:.2f}")
            self.trainer.maybe_train()

        if dc.get("auto_cleanup") and crawl_runs:
            for run_id in crawl_runs:
                self.collector.cleanup(run_id)

    def crawl_train(self, query: str, train: bool = True, dataset_type: str | None = None, raw: bool = False, parsed: bool = True, limit: int | None = None) -> None:
        """One-shot: search + crawl + build a dataset.

        If ``dataset_type`` is ``"text"`` (and ``train`` is True, no ``raw``), the
        model synthesizes (instruction, response) pairs, they're added to memory,
        and a fine-tune pass runs. ``--raw`` collects + structures page data only
        (no model). ``image``/``video`` (or ``--no-train``) just save the dataset.
        ``limit`` caps how many URLs/articles end up in the dataset.
        """
        dataset_type = dataset_type or self.config.raw.get("data_collection", {}).get(
            "dataset_type", "text"
        )
        if dataset_type == "text" and not raw:
            self._ensure_model()

        examples, run_id = self.collector.collect(query, dataset_type=dataset_type, raw=raw, parsed=parsed, limit=limit)

        if dataset_type == "text" and train and not raw:
            for ex in examples:
                self.memory.add(ex["task"], ex["solution"], 1.0, ex.get("meta"))
            self.trainer.maybe_train(force=True)
        else:
            print(
                f"[crawl] dataset '{dataset_type}' saved @ runs/{run_id} "
                f"({len(examples)} items); training skipped."
            )

    def fetch_train(self, urls: list[str], train: bool = True, dataset_type: str | None = None, raw: bool = False, parsed: bool = True, limit: int | None = None) -> None:
        """Pure webfetch: crawl the given URLs directly (no search) -> dataset.

        With ``raw=True`` it collects + structures page data (no model needed).
        With ``dataset_type="text"`` and ``train=True`` it synthesizes
        (instruction, response) pairs and fine-tunes. Skips search entirely.
        ``limit`` caps how many of the given URLs are processed.
        """
        dataset_type = dataset_type or self.config.raw.get("data_collection", {}).get(
            "dataset_type", "text"
        )
        if dataset_type == "text" and not raw:
            self._ensure_model()

        examples, run_id = self.collector.fetch(urls, dataset_type=dataset_type, raw=raw, parsed=parsed, limit=limit)

        if dataset_type == "text" and train and not raw:
            for ex in examples:
                self.memory.add(ex["task"], ex["solution"], 1.0, ex.get("meta"))
            self.trainer.maybe_train(force=True)
        else:
            print(
                f"[fetch] dataset '{dataset_type}' saved @ runs/{run_id} "
                f"({len(examples)} items); training skipped."
            )

    def _attempt(self, task: str):
        score = -1.0
        solution = ""
        for _ in range(self.max_retries + 1):
            solution = self.actor.solve(task)
            score = self.evaluator.score(task, solution)
            if score >= self.threshold:
                break
        return solution, score
