# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

from src.config import Config


class ImageTrainer:
    """Fine-tunes a Stable-Diffusion-style model (LoRA) on collected images.

    Consumes the dataset written by ``ppte crawl --type image``
    (``images.jsonl`` with ``caption`` + ``image_path``/``image_url``). Runs a
    Dreambooth-style LoRA fine-tune with the HuggingFace ``diffusers`` +
    ``peft`` stacks. All heavy imports are lazy so the base tool works without
    them; install with::

        pip install diffusers accelerate peft safetensors
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg = config.raw.get("image", {})

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def load_dataset(self, images_jsonl: str | Path) -> list[dict]:
        """Read records from an ``images.jsonl`` file."""
        rows: list[dict] = []
        with open(images_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _resolve_local_paths(self, records: list[dict]) -> list[tuple[str, str]]:
        """Return (local_image_path, caption) pairs, downloading URLs as needed."""
        pairs: list[tuple[str, str]] = []
        tmp_dir = Path(self.config.get("paths", "crawl_dir", default="data/crawl")) / "img_dl"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(records):
            cap = (r.get("caption") or "").strip()
            if not cap:
                continue
            path = r.get("image_path")
            if not path:
                url = r.get("image_url")
                if not url:
                    continue
                import requests

                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    dest = tmp_dir / f"{i:05d}.jpg"
                    dest.write_bytes(resp.content)
                    path = str(dest)
                except Exception as exc:  # noqa: BLE001
                    print(f"[img-train] skip {url}: {exc}")
                    continue
            if Path(path).exists():
                pairs.append((str(path), cap))
        return pairs

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        images_jsonl: str | Path,
        out_dir: str | Path,
        base_model: str | None = None,
        device: str | None = None,
        max_train_steps: int | None = None,
    ) -> str:
        """Run a Dreambooth-style LoRA fine-tune on an image dataset."""
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from diffusers.utils import check_min_version
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig, get_peft_model

        check_min_version("0.10.0")

        base = base_model or self.cfg.get("base_model", "runwayml/stable-diffusion-v1-5")
        device = device or self.cfg.get("device", "auto")
        steps = max_train_steps or self.cfg.get("max_train_steps", 500)
        lr = float(self.cfg.get("learning_rate", 1e-4))
        resolution = int(self.cfg.get("resolution", 512))

        records = self.load_dataset(images_jsonl)
        pairs = self._resolve_local_paths(records)
        if not pairs:
            raise RuntimeError(f"no usable image+caption pairs in {images_jsonl}")
        print(f"[img-train] {len(pairs)} image+caption pairs; base={base}; steps={steps}; device={device}")

        # Real device resolution: "auto" -> cuda if available.
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        import torch

        tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

        # Freeze VAE + text encoder; LoRA the UNet attention projections.
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=int(self.cfg.get("lora_r", 8)),
            lora_alpha=int(self.cfg.get("lora_alpha", 16)),
            target_modules=self.cfg.get(
                "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]
            ),
        )
        unet = get_peft_model(unet, lora_cfg)
        unet.train()

        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image

        class ImgCap(Dataset):
            def __init__(self, pairs_, tok, res):
                self.pairs = pairs_
                self.tok = tok
                self.res = res
                import torchvision.transforms as T

                self.tf = T.Compose(
                    [
                        T.Resize(res, interpolation=T.InterpolationMode.BILINEAR),
                        T.CenterCrop(res),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ]
                )

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, i):
                path, cap = self.pairs[i]
                img = Image.open(path).convert("RGB")
                pix = self.tf(img)
                ids = self.tok(
                    cap,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0]
                return {"pixel_values": pix, "input_ids": ids}

        train_ds = ImgCap(pairs, tokenizer, resolution)
        batch_size = int(self.cfg.get("batch_size", 2))
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda b: {
                "pixel_values": torch.stack([x["pixel_values"] for x in b]),
                "input_ids": torch.stack([x["input_ids"] for x in b]),
            },
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()), lr=lr
        )
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=steps,
        )
        unet.to(device)
        text_encoder.to(device)
        vae.to(device)

        global_step = 0
        vae_dtype = next(vae.parameters()).dtype
        from accelerate import Accelerator

        accelerator = Accelerator()
        unet, text_encoder, vae, optimizer, loader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, vae, optimizer, loader, lr_scheduler
        )

        while global_step < steps:
            for batch in loader:
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(device).to(vae_dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                    ).long()
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                    ids = batch["input_ids"].to(device)
                    enc = text_encoder(ids)[0]

                with accelerator.accumulate(unet):
                    pred = unet(noisy, timesteps, enc).sample
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"[img-train] step {global_step}/{steps} loss={loss.item():.4f}")
                if global_step >= steps:
                    break

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            unet.save_pretrained(out_dir)
            (out_dir / "base_model.txt").write_text(base)
            print(f"[img-train] LoRA saved @ {out_dir}")

        del unet, text_encoder, vae, optimizer, loader
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return str(out_dir)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        adapter_dir: str | Path,
        out_path: str | Path,
        seed: int | None = None,
    ) -> str:
        """Generate an image from a prompt with a trained LoRA adapter."""
        from diffusers import DiffusionPipeline, AutoencoderKL
        import torch

        base = (Path(adapter_dir) / "base_model.txt").read_text().strip()
        device = self.cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = DiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe.to(device)
        try:
            pipe.load_lora_weights(str(adapter_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[img-run] could not load LoRA {adapter_dir}: {exc}")
        if seed is not None:
            torch.manual_seed(seed)
        img = pipe(prompt, num_inference_steps=int(self.cfg.get("num_inference_steps", 30))).images[0]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"[img-run] saved {out_path}")
        return str(out_path)
