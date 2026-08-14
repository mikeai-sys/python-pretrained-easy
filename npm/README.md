# @eliseart/ppte

**Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.**

**Python Pretraining Easy** — a local, GPU/CPU-friendly self-improving AI toolkit, distributed as an npm package.

The npm package is a thin wrapper: on first install it bootstraps an isolated Python venv with platform- and GPU-aware dependencies, then runs the real Python CLI (`ppte`) underneath. Works on **Linux, macOS, and Windows**, on **x64 and arm64**, with **NVIDIA GPU (CUDA), Apple Silicon (MPS), or plain CPU**. No GPU required — small models train on CPU.

---

## Table of contents

- [Install](#install)
- [Environment setup](#environment-setup)
- [Global options](#global-options)
- [Commands](#commands)
  - [`ppte install`](#ppte-install)
  - [`ppte crawl`](#ppte-crawl)
  - [`ppte fetch`](#ppte-fetch)
  - [`ppte links`](#ppte-links)
  - [`ppte run`](#ppte-run)
  - [`ppte train`](#ppte-train)
  - [`ppte export`](#ppte-export)
  - [`ppte delete`](#ppte-delete)
  - [`ppte config`](#ppte-config)
  - [`ppte render`](#ppte-render)
  - [`ppte rag`](#ppte-rag)
  - [`ppte img`](#ppte-img)
- [Configuration reference](#configuration-reference)
- [Hardware detection](#hardware-detection)
- [Publishing (for maintainers)](#publishing-for-maintainers)
- [License](#license)

---

## Install

```bash
npm install -g @eliseart/ppte
```

First run (postinstall) downloads the Python ML stack, so it can take a few minutes. It will use an existing **Python 3.10+** on your system, or attempt to install one automatically. If you need to skip the automatic setup:

```bash
npm install -g @eliseart/ppte --ignore-scripts
# then run `ppte` once — it will bootstrap lazily on first use
```

After installing, verify it works:

```bash
ppte --help
```

> **Note on npm scripts warnings.** Newer npm versions block postinstall scripts by
> default. You may see `warn install-scripts ... blocked`. This is fine — the first
> `ppte` command will auto-bootstrap the environment. To run the setup during install
> instead, allow the script once:
>
> ```bash
> npm install -g --allow-scripts=@eliseart/ppte
> # or permanently:
> npm config set allow-scripts=@eliseart/ppte --location=user
> ```

---

## Environment setup

`ppte` auto-creates its Python environment on first install. You can also manage
the toolchains explicitly (see [`ppte install`](#ppte-install)).

Set `PPTE_VENV` to place the venv somewhere other than the default, and
`PPTE_PYTHON` to force a specific Python interpreter.

---

## Global options

These options work with **every** command:

| Option | Description | Example |
|---|---|---|
| `--config PATH` | Path to the config file (default `config.yaml`) | `ppte run --config my-config.yaml` |

All examples below omit `--config`; prefix any command with it to use a custom config.

---

## Commands

### `ppte install`

Set up a Python environment and install the toolchains. `ppte` installs the
small base deps automatically on first run (works for `--help`, `crawl --raw`,
`render`, `links`, etc.). The heavy ML toolchain (torch, transformers — ~3-4 GB)
is installed when there's enough free disk space, or on demand here:

| Option | Description |
|---|---|
| `torch` | Install just the ML toolchain (torch, transformers, trl, peft, datasets, accelerate) |
| `txt` | Same as `torch` (full text/LLM toolchain) |
| `img` | Install image tools (diffusers, torchvision, peft, safetensors, accelerate) |
| `python` | Attempt to install Python 3 via the OS package manager (apt/brew/winget) if missing |
| `--dry-run` | Print what would run without actually installing (works with `txt` and `img`) |

Examples:

```bash
ppte install torch                   # ML toolchain (all LLM commands work)
ppte install txt --dry-run           # preview only, don't install
ppte install img                     # image training toolchain
ppte install img --dry-run           # preview image deps
ppte install python                  # install Python 3 if missing
```

> **Low disk space?** If your disk has under ~4 GB free, the first-run bootstrap
> installs only the small base deps and prints `SKIPPED ML toolchain`, then asks
> whether to install it anyway. `ppte --help` and the web tools (`links`, `render`,
> `crawl --raw`, `config`, `delete`) work without torch — only ML commands
> (`ppte run`, `ppte train`, `ppte rag --summarize`, `ppte img`) need it. Run
> `ppte install torch` once you have space.

---

### `ppte crawl`

Search the web, crawl results, build a dataset, and optionally fine-tune.

```
ppte crawl QUERY [--type text|image|video] [--limit N] [--no-train] [--raw] [--no-parse]
```

| Option | Description | Example |
|---|---|---|
| `QUERY` | Search query / topic (positional) | `ppte crawl "machine learning"` |
| `--type` | Dataset type: `text`, `image`, or `video` (default: from config) | `--type image` |
| `--limit` | Max items: articles for text, images/videos for media | `--limit 50` |
| `--no-train` | Build dataset but skip fine-tuning | `--no-train` |
| `--raw` | Collect + structure page data only, no model (fast) | `--raw` |
| `--no-parse` | With `--raw`: dump flat text instead of structured parse | `--no-parse` |

Examples:

```bash
# Crawl articles about transformers and fine-tune on them
ppte crawl "how do transformers work"

# Same, but only collect the dataset (no fine-tune)
ppte crawl "how do transformers work" --no-train

# Collect exactly 50 images of red foxes
ppte crawl "red fox" --type image --limit 50

# Collect 20 video thumbnails
ppte crawl "surfing" --type video --limit 20

# Fast webfetch only, no model
ppte crawl "stable diffusion tutorials" --type text --raw
```

---

### `ppte fetch`

Crawl given URLs directly (no search). Same dataset-building pipeline as `crawl`.

```
ppte fetch URL1 URL2 ... [--file urls.txt] [--type text|image|video] [--limit N] [--raw] [--no-parse] [--no-train]
```

| Option | Description | Example |
|---|---|---|
| `URL1 URL2 ...` | URLs to crawl (positional) | `ppte fetch https://a.com https://b.com` |
| `-f, --file` | Read URLs from a file, one per line | `-f urls.txt` |
| `--type` | Dataset type: `text`, `image`, or `video` | `--type text` |
| `--limit` | Max URLs to process from the input | `--limit 50` |
| `--raw` | Collect + structure page data only, no model | `--raw` |
| `--no-parse` | With `--raw`: dump flat text | `--no-parse` |
| `--no-train` | Skip fine-tuning | `--no-train` |

Examples:

```bash
ppte fetch https://en.wikipedia.org/wiki/Machine_learning
ppte fetch -f urls.txt --type text --raw          # collect only, no model
ppte fetch -f urls.txt --limit 50 --no-train      # process first 50 URLs
ppte fetch https://a.com --type text --no-train
```

---

### `ppte links`

Extract article URLs from a page or sitemap into a file.

```
ppte links SOURCE [--out urls.txt] [--match REGEX] [--same-domain]
```

| Option | Description | Example |
|---|---|---|
| `SOURCE` | Page or sitemap URL to scrape links from | `ppte links https://blog.com/sitemap.xml` |
| `--out` | Output file, one URL per line (default `urls.txt`) | `--out articles.txt` |
| `--match` | Keep only URLs matching this regex | `--match "science"` |
| `--same-domain` | Keep only links to the same domain | `--same-domain` |

Examples:

```bash
ppte links "https://en.wikipedia.org/wiki/Transformer_(machine_learning)" --out urls.txt
ppte links "https://blog.example.com/sitemap.xml" --match "python" --out python-urls.txt
ppte links "https://news.example.com" --same-domain --out same-site.txt
```

---

### `ppte run`

Run the autonomous self-improve loop: generate tasks → solve with the local model → evaluate → remember good examples → fine-tune.

```
ppte run [--iterations N]
```

| Option | Description | Example |
|---|---|---|
| `--iterations` | Number of task/solve/eval cycles (default: from config `loop.iterations`) | `--iterations 10` |

Examples:

```bash
ppte run
ppte run --iterations 20
ppte run --iterations 5 --config my-config.yaml
```

The loop automatically fine-tunes once `loop.train_every` good examples have been accumulated in `memory.jsonl`.

---

### `ppte train`

Force a LoRA fine-tune pass on the good examples currently in memory.

```
ppte train [--config PATH]
```

Examples:

```bash
ppte train
ppte train --config my-config.yaml
```

Output: prints the path of the saved LoRA adapter, e.g. `data/checkpoint/adapter-1786710364/`.

---

### `ppte export`

Export the good examples from memory into a chat-format JSONL dataset.

```
ppte export [--out data/sft.jsonl]
```

| Option | Description | Example |
|---|---|---|
| `--out` | Output JSONL path (default `data/sft.jsonl`) | `--out dataset.jsonl` |

Examples:

```bash
ppte export
ppte export --out training-data.jsonl
```

Each line is a `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` object — compatible with SFT training tools.

---

### `ppte delete`

Delete crawled data (one run or all runs).

```
ppte delete [RUN_ID]
```

| Option | Description | Example |
|---|---|---|
| `RUN_ID` | Run id to delete; omit to delete all runs | `ppte delete run_1786641914` |

Examples:

```bash
ppte delete                  # remove all crawl runs
ppte delete run_1786641914   # remove one run
```

---

### `ppte config`

Show or interactively set the search-provider / API-key configuration.

```
ppte config [--show]
```

| Option | Description | Example |
|---|---|---|
| `--show` | Show current provider/key status instead of the interactive wizard | `ppte config --show` |

Examples:

```bash
ppte config            # interactive setup (pick provider, paste keys)
ppte config --show     # print current provider + key status
```

Supported providers: `duckduckgo` (no key needed), `brave`, `serpapi`, `google`, `searlo`. Keys are saved to `.env` as `PPTE_SEARCH_PROVIDER` + the provider's key vars.

#### How to set API keys

**Step 1 — run the interactive wizard:**

```bash
ppte config
```

It will:
1. Prompt you to pick a search provider (type the number, e.g. `2` for Brave).
2. Ask you to paste each required API key (blank = skip).
3. Save everything to a `.env` file in the current directory.

**Step 2 — verify:**

```bash
ppte config --show
```

Output shows the active provider and whether each key is `set` or `MISSING`.

**What each provider needs:**

| Provider | Key needed | Env var saved | How to get the key |
|---|---|---|---|
| `duckduckgo` | none | — | built-in, no key |
| `brave` | Brave Search API key | `BRAVE_API_KEY` | https://api.search.brave.com |
| `serpapi` | SerpAPI key | `SERPAPI_KEY` | https://serpapi.com |
| `google` | Google API key + CSE ID | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | https://developers.google.com/custom-search |
| `searlo` | Searlo API key | `SEARLO_API_KEY` | https://searlo.tech |

**Without using the wizard** — set the env vars directly (or add them to `.env`):

```bash
export PPTE_SEARCH_PROVIDER=brave
export BRAVE_API_KEY=your_key_here
ppte config --show   # confirm both are set
```

> **No key needed for basic use.** The default provider `duckduckgo` works out of the
> box with zero configuration — `ppte config` is only required for Brave/SerpAPI/Google/Searlo.

---

### `ppte render`

Render a `parsed.jsonl` file or a single URL into clean Markdown.

```
ppte render [--file parsed.jsonl] [--url URL] [--section NAME] [--out FILE]
```

| Option | Description | Example |
|---|---|---|
| `--file` | `parsed.jsonl` file to render | `--file data/crawl/<run>/parsed.jsonl` |
| `--url` | Fetch + parse + render a single URL | `--url "https://en.wikipedia.org/wiki/Machine_learning"` |
| `--section` | Render only the section under this heading (substring match) | `--section "Supervised learning"` |
| `--out` | Write Markdown to a file instead of stdout | `--out articles.md` |

Examples:

```bash
ppte render --file data/crawl/run_1234/parsed.jsonl
ppte render --file parsed.jsonl --out articles.md
ppte render --file parsed.jsonl --section "Supervised learning"
ppte render --url "https://en.wikipedia.org/wiki/Machine_learning"
```

Provide `--url` **or** `--file`, not both.

---

### `ppte rag`

Build/query a RAG (retrieval-augmented generation) index over `parsed.jsonl`. GPU-friendly (falls back to CPU).

```
ppte rag build --file parsed.jsonl [--out rag_index] [--model ID] [--device cuda|cpu|auto]
ppte rag query --index rag_index --question "..." [--k N] [--model ID] [--device ...] [--summarize]
```

**`build` options:**

| Option | Description | Example |
|---|---|---|
| `--file` | `parsed.jsonl` to embed (required) | `--file data/crawl/<run>/parsed.jsonl` |
| `--out` | Output index directory (default `rag_index`) | `--out my_index` |
| `--model` | Embedding model id (default from config) | `--model sentence-transformers/all-MiniLM-L6-v2` |
| `--device` | `cuda`, `cpu`, or `auto` | `--device cuda` |

**`query` options:**

| Option | Description | Example |
|---|---|---|
| `--index` | Index dir from `build` (default `rag_index`) | `--index my_index` |
| `--question` | Question to retrieve for (required) | `--question "How do plants make food?"` |
| `--k` | Number of chunks to retrieve (default from config `rag.top_k`) | `--k 5` |
| `--model` | Embedding model id (must match the build) | `--model ...` |
| `--device` | `cuda`, `cpu`, or `auto` | `--device cpu` |
| `--summarize` | Also run the local model to answer from the retrieved context | `--summarize` |

Examples:

```bash
ppte rag build --file data/crawl/run_1234/parsed.jsonl --out rag_index
ppte rag query --index rag_index --question "How do plants make food from light?" --k 3
ppte rag query --index rag_index --question "..." --summarize
```

Requires `pip install sentence-transformers` (optional; the tool tells you if it's missing).

---

### `ppte img`

Fine-tune a Stable-Diffusion-style model (LoRA) on collected images, or generate images with a trained adapter.

```
ppte img train [--file images.jsonl] [--out data/img_lora] [--base MODEL] [--device cuda|cpu|auto] [--steps N]
ppte img run --prompt "..." [--adapter data/img_lora] [--out img_out.png] [--seed N]
```

**`train` options:**

| Option | Description | Example |
|---|---|---|
| `--file` | `images.jsonl` — or a crawl dir (uses the newest run) | `--file data/crawl/run_1234/images.jsonl` |
| `--out` | Output LoRA adapter dir (default from config) | `--out my_lora` |
| `--base` | Base diffusion model id (default from config) | `--base runwayml/stable-diffusion-v1-5` |
| `--device` | `cuda`, `cpu`, or `auto` | `--device cuda` |
| `--steps` | Max training steps | `--steps 1000` |

**`run` options:**

| Option | Description | Example |
|---|---|---|
| `--prompt` | Prompt to generate (required) | `--prompt "a red fox in the snow"` |
| `--adapter` | LoRA adapter dir from `img train` | `--adapter my_lora` |
| `--out` | Output image path (default `img_out.png`) | `--out fox.png` |
| `--seed` | RNG seed for reproducibility | `--seed 42` |

Examples:

```bash
# Collect 50 images first
ppte crawl "red fox" --type image --limit 50

# Train a LoRA on them
ppte img train --file data/crawl/run_1234/images.jsonl --out redfox_lora --steps 800

# Generate with the trained LoRA
ppte img run --prompt "a red fox in the snow at night" --adapter redfox_lora --out fox.png --seed 42
```

Requires `ppte install img` (or `pip install diffusers ...`). A GPU makes training practical.

---

## Configuration reference

`ppte` reads `config.yaml` (in your working directory, or the bundled default). All settings have defaults, so an empty config works.

### `actor:` — the main model

| Key | Default | Description |
|---|---|---|
| `model_id` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Any HuggingFace chat model id or local path (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `load_in_4bit` | `false` | `true` = 4-bit QLoRA (needs CUDA GPU + bitsandbytes) |
| `max_new_tokens` | `512` | Max tokens generated per answer |
| `temperature` | `0.7` | Sampling temperature |

### `judge:` — the evaluator model

| Key | Default | Description |
|---|---|---|
| `model_id` | `null` | `null` = reuse the actor model; or a stronger fixed model id |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `load_in_4bit` | `false` | 4-bit QLoRA |

### `loop:` — the self-improve loop

| Key | Default | Description |
|---|---|---|
| `iterations` | `10` | Task/solve/eval cycles per `ppte run` |
| `eval_threshold` | `0.7` | Keep examples scoring ≥ this |
| `train_every` | `8` | Fine-tune after this many good examples |
| `max_retries` | `2` | Retries per task before discard |

### `finetune:` — LoRA/QLoRA training

| Key | Default | Description |
|---|---|---|
| `method` | `lora` | `lora` (GPU/CPU) or `qlora` (4-bit, needs CUDA GPU) |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |
| `target_modules` | `["q_proj", "v_proj"]` | Attention layers to adapt (must match your model; Qwen2: `q_proj, k_proj, v_proj, o_proj`) |
| `epochs` | `1` | Training epochs |
| `learning_rate` | `2.0e-4` | Learning rate |
| `per_device_batch_size` | `4` | Batch size |
| `gradient_accumulation_steps` | `4` | Gradient accumulation |

### `tools.web_search:` — search provider

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable web search |
| `provider` | `duckduckgo` | `duckduckgo`, `brave`, `serpapi`, `google`, `searlo` |
| `api_key_env` | `SEARCH_API_KEY` | Generic fallback key env var |
| `brave_api_key_env` | `BRAVE_API_KEY` | Brave key env var |
| `serpapi_api_key_env` | `SERPAPI_KEY` | SerpAPI key env var |
| `max_results` | `10` | Results per query |
| `retries` | `3` | Backoff retries |
| `retry_delay` | `5` | Seconds between retries |
| `backend` | `html` | `html` avoids DDG throttling; `auto` uses JSON |
| `google.api_key_env` | `GOOGLE_API_KEY` | Google API key env var |
| `google.cse_id_env` | `GOOGLE_CSE_ID` | Google CSE id env var |
| `searlo.api_url` | `https://api.searlo.tech/api/v1` | Searlo base URL |
| `searlo.api_key_env` | `SEARLO_API_KEY` | Searlo key env var |

### `tools.crawl:` — web crawler

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable crawling |
| `max_pages` | `5` | Crawl cap |
| `respect_robots` | `true` | Respect `robots.txt` |
| `delay` | `1.0` | Polite pause between requests (seconds) |
| `timeout` | `10` | Request timeout (seconds) |
| `user_agent` | `SelfImprovingAgent/0.1 (...)` | Crawler user agent |
| `browser.enabled` | `false` | Use Playwright for JS-rendered pages |
| `browser.headless` | `true` | Headless browser |
| `browser.timeout` | `20000` | Browser timeout (ms) |

### `data_collection:` — autonomous web data collection

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable collection at `ppte run` start |
| `queries` | `[]` | Topics to search, e.g. `["how transformers work"]` |
| `dataset_type` | `text` | `text`, `image`, or `video` |
| `max_results_per_query` | `10` | Search results per query |
| `examples_per_doc` | `2` | Synthesized (instruction, response) pairs per doc |
| `max_chars_per_doc` | `4000` | Max chars kept per doc |
| `max_media` | `20` | Images/videos collected per query (default for `--limit`) |
| `image_overfetch` | `1.3` | Over-fetch candidates to absorb download failures |
| `download_media` | `true` | Download images / video thumbnails |
| `caption_from_source` | `true` | Use source-page text for captions (no model) |
| `max_caption_chars` | `300` | Truncate captions to this length |

### `paths:` — where things live

| Key | Default | Description |
|---|---|---|
| `memory` | `data/memory.jsonl` | Good-examples memory |
| `model_checkpoint` | `data/checkpoint` | LoRA adapter output dir |
| `sft_dataset` | `data/sft.jsonl` | Export dataset path |
| `crawl_dir` | `data/crawl` | Crawl run output dir |

### `rag:` — RAG index

| Key | Default | Description |
|---|---|---|
| `model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model id |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `chunk_chars` | `800` | Chunk size for splitting pages |
| `chunk_overlap` | `100` | Overlap between chunks |
| `top_k` | `5` | Default retrieval count |

### `image:` — image-model fine-tuning

| Key | Default | Description |
|---|---|---|
| `base_model` | `runwayml/stable-diffusion-v1-5` | Base diffusion model |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `resolution` | `512` | Train image resolution (px) |
| `batch_size` | `2` | Batch size |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `target_modules` | `["to_q", "to_k", "to_v", "to_out.0"]` | UNet attention projections |
| `learning_rate` | `1.0e-4` | Learning rate |
| `max_train_steps` | `500` | LoRA training steps |
| `num_inference_steps` | `30` | Sampling steps for `img run` |
| `adapter_dir` | `data/img_lora` | Default LoRA output/read dir |

---

## Hardware detection

The bootstrap picks dependencies automatically:

| Platform | Arch | Detected | torch |
|---|---|---|---|
| Linux / Windows | x64 | NVIDIA GPU (`nvidia-smi`) | CUDA 12.1 build |
| Linux / Windows | x64 | no GPU | CPU build |
| macOS | arm64 / x64 | Apple Silicon or Intel | default (MPS/CPU) |

Override detection with environment variables: `PPTE_PYTHON` (path to a specific Python interpreter), `PPTE_VENV` (venv location).

> **GPU is optional.** No GPU is required — small AI models (e.g. Qwen2.5-0.5B / 1.5B, TinyLlama) train fine on CPU. The tool auto-detects whatever you have (NVIDIA GPU → CUDA, Apple Silicon → MPS, otherwise CPU) and runs on all of them. A GPU just makes larger models and training faster.

---

## Publishing (for maintainers)

```bash
cd npm
npm login
npm publish --access public    # `prepare` script stages the Python source automatically
```

The package name is `@eliseart/ppte` (the bare `ppte` name is taken on the npm registry).

---

## License

Apache-2.0

---

## Copyright

**Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.**

This software and its documentation are the property of Mike Koci-Eliseart
Company ("Mike Koci" / "Eliseart Company"). Unauthorized copying, distribution,
or use of this software, in whole or in part, without the prior written consent
of the copyright holder is prohibited.
