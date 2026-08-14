# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config


class LocalModel:
    """Runs a HuggingFace causal LM locally on GPU via PyTorch.

    One instance per model. Loads in 4-bit (QLoRA) by default to fit a single
    consumer GPU. After fine-tuning, call ``load_adapter`` to pick up new LoRA
    weights without restarting the process.
    """

    def __init__(self, config: Config, section: str = "actor") -> None:
        self.config = config
        sec = config.raw.get(section, {})
        self.model_id: str = sec.get("model_id") or config.get("actor", "model_id")
        self.device: str = sec.get("device", "auto")
        self.load_in_4bit: bool = sec.get("load_in_4bit", True)
        self.max_new_tokens: int = sec.get("max_new_tokens", 512)
        self.temperature: float = sec.get("temperature", 0.7)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = self._load()

    def _load(self):
        kwargs: dict = {"device_map": self.device}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=kwargs.get("temperature", self.temperature) > 0,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def load_adapter(self, adapter_path: str) -> None:
        """Swap in freshly trained LoRA weights."""
        from peft import PeftModel

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self._load()
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload() if False else self.model
