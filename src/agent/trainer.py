# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import gc
import json
import time
import torch
from pathlib import Path

from src.config import Config
from src.agent.memory import Memory


class Trainer:
    """Fine-tunes the actor locally with PEFT/LoRA (or 4-bit QLoRA).

    Uses HuggingFace TRL SFTTrainer. After training, the actor model is reloaded
    with the new LoRA adapter so the loop keeps improving in-process.
    """

    def __init__(self, config: Config, memory: Memory, actor_model=None) -> None:
        self.config = config
        self.memory = memory
        self.actor_model = actor_model

    def maybe_train(self, force: bool = False) -> str | None:
        examples = self.memory.good_examples(
            self.config.get("loop", "eval_threshold", default=0.7)
        )
        train_every = self.config.get("loop", "train_every", default=8)
        if not force and len(examples) < train_every:
            return None

        step = int(time.time())
        adapter_dir = Path(self.config.get("paths", "model_checkpoint", default="data/checkpoint"))
        adapter_dir = adapter_dir / f"adapter-{step}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        self._train(examples, adapter_dir)
        self.actor_model.load_adapter(str(adapter_dir))
        return str(adapter_dir)

    def _train(self, examples: list[dict], out_dir: Path) -> None:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

        ft = self.config.raw.get("finetune", {})
        tokenizer = AutoTokenizer.from_pretrained(self.actor_model.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def fmt(ex):
            return {
                "text": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": ex["task"]},
                        {"role": "assistant", "content": ex["solution"]},
                    ],
                    tokenize=False,
                )
            }

        dataset = Dataset.from_list([fmt(e) for e in examples])

        model_kwargs: dict = {"device_map": self.actor_model.device}
        if ft.get("method", "qlora") == "qlora":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.actor_model.model_id, **model_kwargs
        )

        peft_config = LoraConfig(
            r=ft.get("lora_r", 16),
            lora_alpha=ft.get("lora_alpha", 32),
            lora_dropout=ft.get("lora_dropout", 0.05),
            target_modules=ft.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=ft.get("epochs", 1),
            per_device_train_batch_size=ft.get("per_device_batch_size", 4),
            gradient_accumulation_steps=ft.get("gradient_accumulation_steps", 4),
            learning_rate=float(ft.get("learning_rate", 2e-4)),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(str(out_dir))

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    def export_dataset(self, path: str | Path) -> None:
        """Dump good examples into a supervised-fine-tuning JSONL file."""
        examples = self.memory.good_examples(
            self.config.get("loop", "eval_threshold", default=0.7)
        )
        with open(path, "w") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": ex["task"]},
                                {"role": "assistant", "content": ex["solution"]},
                            ]
                        }
                    )
                    + "\n"
                )
