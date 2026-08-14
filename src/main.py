# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from src.config import load_config


def _load(config_path: str):
    # Resolve relative to cwd first; if missing, fall back to the config shipped
    # next to the package so `ppte` works when run from any directory.
    p = Path(config_path)
    if not p.exists():
        bundled = Path(__file__).resolve().parent.parent / config_path
        if bundled.exists():
            config_path = str(bundled)
    return load_config(config_path)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_crawl(args) -> None:
    from src.agent import SelfImprovingAgent

    agent = SelfImprovingAgent(_load(args.config))
    agent.crawl_train(
        args.query, train=not args.no_train, dataset_type=args.type,
        raw=args.raw, parsed=not args.no_parse, limit=args.limit,
    )


def cmd_fetch(args) -> None:
    from src.agent import SelfImprovingAgent

    urls = list(args.urls)
    if args.file:
        with open(args.file) as f:
            urls += [line.strip() for line in f if line.strip()]
    if not urls:
        print("error: provide URLs or --file urls.txt")
        return
    agent = SelfImprovingAgent(_load(args.config))
    agent.fetch_train(
        urls, train=not args.no_train, dataset_type=args.type,
        raw=args.raw, parsed=not args.no_parse, limit=args.limit,
    )


def cmd_links(args) -> None:
    from src.agent.tools.crawl import Crawler

    crawler = Crawler(_load(args.config))
    links = crawler.extract_links(args.source, match=args.match, same_domain=args.same_domain)
    with open(args.out, "w") as f:
        for link in links:
            f.write(link + "\n")
    print(f"[links] wrote {len(links)} URLs to {args.out}")


def cmd_run(args) -> None:
    from src.agent import SelfImprovingAgent

    agent = SelfImprovingAgent(_load(args.config))
    agent.run(iterations=args.iterations)

def cmd_train(args) -> None:
    from src.agent import SelfImprovingAgent

    agent = SelfImprovingAgent(_load(args.config))
    agent._ensure_model()
    path = agent.trainer.maybe_train(force=True)
    print(f"Fine-tuned adapter at: {path}")


def cmd_export(args) -> None:
    from src.agent.memory import Memory
    from src.agent.trainer import Trainer

    config = _load(args.config)
    mem = Memory(config.get("paths", "memory", default="data/memory.jsonl"))
    Trainer(config, mem).export_dataset(args.out)
    print(f"Exported to {args.out}")


def cmd_delete(args) -> None:
    config = _load(args.config)
    crawl_dir = Path(config.get("paths", "crawl_dir", default="data/crawl"))
    if args.run_id:
        target = crawl_dir / args.run_id
        if target.exists():
            shutil.rmtree(target)
            print(f"[delete] removed {target}")
        else:
            print(f"[delete] no such run: {target}")
        return
    if crawl_dir.exists():
        removed = 0
        for child in crawl_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                removed += 1
        print(f"[delete] removed {removed} crawl run(s) in {crawl_dir}")
    else:
        print(f"[delete] nothing to remove at {crawl_dir}")


def cmd_img(args) -> None:
    from src.agent.image_trainer import ImageTrainer

    config = _load(args.config)
    trainer = ImageTrainer(config)
    if args.img_cmd == "train":
        data = args.file or config.get("paths", "crawl_dir", default="data/crawl")
        if Path(data).is_dir():
            candidates = sorted(Path(data).glob("*/images.jsonl"))
            if not candidates:
                print(f"[img] no images.jsonl found under {data}")
                return
            data = str(candidates[-1])
            print(f"[img] using dataset {data}")
        trainer.train(
            data,
            args.out or config.get("image", "adapter_dir", default="data/img_lora"),
            base_model=args.base,
            device=args.device,
            max_train_steps=args.steps,
        )
    elif args.img_cmd == "run":
        if not args.prompt:
            print("[img] --prompt is required for `ppte img run`")
            return
        trainer.generate(
            args.prompt,
            args.adapter or config.get("image", "adapter_dir", default="data/img_lora"),
            args.out or "img_out.png",
            seed=args.seed,
        )


def cmd_install(args) -> None:
    # Load the installer without triggering `src.agent.__init__` (which imports
    # torch and friends) — `install` must work even before they're installed.
    import importlib.util

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ppte_installer", here / "agent" / "installer.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)

    if args.install_cmd == "txt":
        sys.exit(installer.install_txt(dry_run=args.dry_run))
    elif args.install_cmd == "img":
        sys.exit(installer.install_img(dry_run=args.dry_run))
    elif args.install_cmd == "torch":
        sys.exit(installer.install_torch(dry_run=args.dry_run))
    elif args.install_cmd == "python":
        sys.exit(0 if installer.install_python() else 1)
    else:
        print("usage: ppte install {txt|img|torch|python} [--dry-run]")
        sys.exit(2)


def cmd_help(args) -> None:
    print(_HELP_EPILOG.strip())


# Per-provider env vars the interactive setup will ask for.
_PROVIDER_KEYS = {
    "duckduckgo": [],
    "brave": ["BRAVE_API_KEY"],
    "serpapi": ["SERPAPI_KEY"],
    "google": ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"],
    "searlo": ["SEARLO_API_KEY"],
}


def _write_env(new_vals: dict[str, str]) -> None:
    """Merge KEY=VALUE pairs into the project .env file."""
    from pathlib import Path

    path = Path(".env")
    data: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
    data.update(new_vals)
    path.write_text("\n".join(f"{k}={v}" for k, v in data.items()) + "\n")


def _print_status(args) -> None:
    config = _load(args.config)
    sec = config.raw.get("tools", {}).get("web_search", {})
    provider = os.environ.get("PPTE_SEARCH_PROVIDER") or sec.get("provider", "duckduckgo")
    g = sec.get("google", {})
    sl = sec.get("searlo", {})
    env_map = {
        "duckduckgo": [],
        "brave": [sec.get("brave_api_key_env", "BRAVE_API_KEY")],
        "serpapi": [sec.get("serpapi_api_key_env", "SERPAPI_KEY")],
        "google": [g.get("api_key_env", "GOOGLE_API_KEY"), g.get("cse_id_env", "GOOGLE_CSE_ID")],
        "searlo": [sl.get("api_key_env", "SEARLO_API_KEY")],
    }
    print("== ppte API configuration ==")
    print(f"config file : {args.config}")
    print(f"provider    : {provider}")
    print("API keys   :")
    for var in env_map.get(provider, []):
        status = "set" if os.environ.get(var) else "MISSING"
        print(f"  - {var}: {status}")
    print("\nRun `ppte config` (no flag) to set these interactively.")


def cmd_render(args) -> None:
    from src.agent.parser import (
        ContentParser,
        render_markdown,
        render_section,
    )

    if args.url:
        from src.agent.tools.crawl import Crawler

        crawler = Crawler(_load(args.config))
        html = crawler.fetch_html(args.url)
        record = ContentParser().parse(html, args.url)
        md = (
            render_section(record, args.section)
            if args.section
            else render_markdown(record)
        )
        if md is None:
            print(f"[render] no section matching {args.section!r}")
            return
        print(md)
        return

    if not args.file:
        print("error: provide --url URL or --file parsed.jsonl")
        return

    out_lines: list[str] = []
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            md = (
                render_section(record, args.section)
                if args.section
                else render_markdown(record)
            )
            if md is None:
                continue
            out_lines.append(md.rstrip("\n"))
            out_lines.append("\n---\n")

    joined = "".join(out_lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(joined)
        print(f"[render] wrote {args.file} -> {args.out}")
    else:
        print(joined, end="")


def cmd_rag(args) -> None:
    from src.agent.rag import build_index, query_index, build_prompt

    config = _load(args.config)
    rag_cfg = config.raw.get("rag", {})
    model = args.model or rag_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    device = args.device or rag_cfg.get("device", "auto")

    if args.rag_cmd == "build":
        n = build_index(
            args.file,
            args.out,
            model,
            device,
            rag_cfg.get("chunk_chars", 800),
            rag_cfg.get("chunk_overlap", 100),
        )
        print(f"[rag] built index with {n} chunks @ {args.out}")
    elif args.rag_cmd == "query":
        results = query_index(
            args.index, args.question, model, device, k=args.k or rag_cfg.get("top_k", 5)
        )
        if args.summarize:
            from src.agent import SelfImprovingAgent

            agent = SelfImprovingAgent(config)
            agent._ensure_model()
            prompt = build_prompt(args.question, results)
            answer = agent.actor_model.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=512
            )
            print("ANSWER:")
            print(answer)
            print("\nSOURCES:")
            for chunk, score in results:
                print(f"  [{score:.3f}] {chunk.get('meta', {}).get('url', '')}")
        else:
            for chunk, score in results:
                print(f"[{score:.3f}] {chunk['text'][:300]}")
                print("---")
    else:
        print("error: specify `build` or `query` subcommand")


def cmd_config(args) -> None:
    if args.show:
        _print_status(args)
        return

    providers = ["duckduckgo", "brave", "serpapi", "google", "searlo"]
    print("== ppte API setup ==")
    print("Select a search provider:")
    for i, p in enumerate(providers, 1):
        print(f"  {i}) {p}")
    choice = input("Provider [1-4, default 1]: ").strip()
    try:
        idx = (int(choice) - 1) if choice else 0
    except ValueError:
        idx = 0
    if idx < 0 or idx >= len(providers):
        idx = 0
    provider = providers[idx]
    print(f"Selected: {provider}")

    new_vals: dict[str, str] = {}
    if provider == "duckduckgo":
        print("DuckDuckGo needs no key. Done.")
    else:
        print("Paste each key and press Enter (blank = skip):")
        for var in _PROVIDER_KEYS[provider]:
            val = input(f"  {var}: ").strip()
            if val:
                new_vals[var] = val

    new_vals["PPTE_SEARCH_PROVIDER"] = provider
    _write_env(new_vals)
    print(f"Saved to .env (provider={provider}). Run `ppte config --show` to verify.")


# --------------------------------------------------------------------------- #
# CLI definition
# --------------------------------------------------------------------------- #
_HELP_EPILOG = """
EXAMPLES
========

Environment setup
-----------------
  ppte install torch               Install just the ML toolchain (torch, transformers, trl, ...)
  ppte install txt                 Install the text/LLM toolchain (torch + everything above)
  ppte install img                 Install the image toolchain (diffusers, torchvision, ...)
  ppte install python              Install Python 3 via the OS package manager (if missing)
  ppte install txt --dry-run       Preview install commands without running them

Collect data from the web
-------------------------
  ppte crawl "how do transformers work"          Search + crawl + build a text dataset (+ fine-tune)
  ppte crawl "topic" --type image --limit 50     Collect exactly 50 images
  ppte crawl "topic" --type video --limit 20     Collect 20 video thumbnails
  ppte crawl "topic" --raw                       Collect page data only, no model (fast)
  ppte crawl "topic" --no-train                  Build dataset without fine-tuning
  ppte fetch https://a.com https://b.com         Crawl specific URLs (no search)
  ppte fetch -f urls.txt --limit 50 --no-train   Crawl up to 50 URLs from a file
  ppte links https://blog.com/sitemap.xml --out urls.txt   Extract article URLs
  ppte links https://site.com --match "ai" --same-domain  Filter links by regex/domain

Self-improving loop (training)
------------------------------
  ppte run                        Run the task -> solve -> eval -> fine-tune loop
  ppte run --iterations 20        Run more iterations
  ppte train                      Force a LoRA fine-tune on good examples in memory
  ppte export --out sft.jsonl     Export good examples as chat-format JSONL

Manage data & config
--------------------
  ppte delete                     Delete all crawl runs
  ppte delete run_12345           Delete one crawl run
  ppte config                     Interactive search-provider / API-key setup
  ppte config --show              Show current provider and key status
  ppte render --file parsed.jsonl --out articles.md      Render dataset to Markdown
  ppte render --url https://x.com --section "Intro"      Render one URL section

RAG (retrieval-augmented generation)
------------------------------------
  ppte rag build --file parsed.jsonl --out rag_index    Embed pages into a vector index
  ppte rag query --index rag_index --question "..." --k 5       Retrieve top-5 chunks
  ppte rag query --index rag_index --question "..." --summarize  Answer from context

Image model (LoRA fine-tune + generate)
---------------------------------------
  ppte img train --file images.jsonl --out img_lora --steps 800   Fine-tune SD LoRA
  ppte img run --prompt "a red fox" --adapter img_lora --out fox.png   Generate image

Run `ppte <command> --help` for full options of each command.
"""


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default="config.yaml", help="Path to config file.")

    parser = argparse.ArgumentParser(
        prog="ppte",
        description=(
            "Python Pretraining Easy — a local, GPU-trained self-improving AI toolkit. "
            "Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP_EPILOG,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("crawl", parents=[parent], help="Search + crawl the web, build a dataset.")
    p.add_argument("query", help="Search query / topic.")
    p.add_argument("--type", choices=["text", "image", "video"], default=None,
                   help="Dataset type (default: config data_collection.dataset_type).")
    p.add_argument("--limit", type=int, default=None,
                    help="Max items in the dataset: articles (text) or images/videos (media).")
    p.add_argument("--no-train", action="store_true", help="Build dataset but skip fine-tuning.")
    p.add_argument("--raw", action="store_true",
                    help="Collect + structure page data only (no model; fast webfetch).")
    p.add_argument("--no-parse", action="store_true",
                   help="With --raw: dump flat text instead of structured parse.")
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("fetch", parents=[parent], help="Crawl given URLs directly (no search).")
    p.add_argument("urls", nargs="*", help="URLs to crawl (or use -f/--file).")
    p.add_argument("-f", "--file", default=None, help="Read URLs from a file (one per line).")
    p.add_argument("--type", choices=["text", "image", "video"], default=None,
                   help="Dataset type (default: config data_collection.dataset_type).")
    p.add_argument("--limit", type=int, default=None,
                   help="Max URLs to process from the input (default: all).")
    p.add_argument("--raw", action="store_true",
                   help="Collect + structure page data only (no model; fast webfetch).")
    p.add_argument("--no-parse", action="store_true",
                   help="With --raw: dump flat text instead of structured parse.")
    p.add_argument("--no-train", action="store_true", help="Build dataset but skip fine-tuning.")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("links", parents=[parent],
                       help="Extract article URLs from a page/sitemap into a file.")
    p.add_argument("source", help="Page or sitemap URL to scrape links from.")
    p.add_argument("--out", default="urls.txt", help="Output file (one URL per line).")
    p.add_argument("--match", default=None, help="Keep only URLs matching this regex.")
    p.add_argument("--same-domain", action="store_true", help="Keep only same-domain links.")
    p.set_defaults(func=cmd_links)

    p = sub.add_parser("run", parents=[parent], help="Run the autonomous self-improve loop.")
    p.add_argument("--iterations", type=int, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("train", parents=[parent], help="Force a fine-tune pass on good examples.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", parents=[parent], help="Export good examples to a JSONL dataset.")
    p.add_argument("--out", default="data/sft.jsonl", help="Output JSONL path.")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("delete", parents=[parent], help="Delete crawled data.")
    p.add_argument("run_id", nargs="?", default=None,
                   help="Run id to delete; omit to delete all runs.")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("config", parents=[parent],
                        help="Show or interactively set API / provider configuration.")
    p.add_argument("--show", action="store_true", help="Show current provider/key status.")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("render", parents=[parent],
                       help="Render parsed.jsonl / a URL into clean Markdown.")
    p.add_argument("--file", default=None, help="parsed.jsonl file to render.")
    p.add_argument("--url", default=None, help="Render a single URL (fetched + parsed).")
    p.add_argument("--section", default=None,
                   help="Render only the section under this heading (substring match).")
    p.add_argument("--out", default=None, help="Write Markdown to this file instead of stdout.")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("rag", parents=[parent],
                       help="Build/query a RAG index over parsed.jsonl (GPU-friendly).")
    rag_sub = p.add_subparsers(dest="rag_cmd")
    rb = rag_sub.add_parser("build", help="Embed parsed.jsonl into a vector index.")
    rb.add_argument("--file", required=True, help="parsed.jsonl to embed.")
    rb.add_argument("--out", default="rag_index", help="Output index directory.")
    rb.add_argument("--model", default=None, help="Embedding model id.")
    rb.add_argument("--device", default=None, help="cuda | cpu | auto")
    rq = rag_sub.add_parser("query", help="Retrieve top-k chunks for a question.")
    rq.add_argument("--index", default="rag_index", help="Index directory from `rag build`.")
    rq.add_argument("--question", required=True, help="Question to retrieve for.")
    rq.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve.")
    rq.add_argument("--model", default=None, help="Embedding model id (matches build).")
    rq.add_argument("--device", default=None, help="cuda | cpu | auto")
    rq.add_argument("--summarize", action="store_true",
                    help="Also run the local model to answer from retrieved context.")
    p.set_defaults(func=cmd_rag)

    p = sub.add_parser("img", parents=[parent],
                       help="Fine-tune a Stable-Diffusion-style model (LoRA) on collected images, or generate.")
    img_sub = p.add_subparsers(dest="img_cmd")
    it = img_sub.add_parser("train", help="LoRA fine-tune on an images.jsonl dataset.")
    it.add_argument("--file", default=None,
                    help="images.jsonl (or a crawl dir; uses the newest run).")
    it.add_argument("--out", default=None, help="Output LoRA adapter dir.")
    it.add_argument("--base", default=None, help="Base diffusion model id (default from config).")
    it.add_argument("--device", default=None, help="cuda | cpu | auto")
    it.add_argument("--steps", type=int, default=None, help="Max training steps.")
    ir = img_sub.add_parser("run", help="Generate an image with a trained LoRA adapter.")
    ir.add_argument("--prompt", default=None, help="Prompt to generate.")
    ir.add_argument("--adapter", default=None, help="LoRA adapter dir from `img train`.")
    ir.add_argument("--out", default=None, help="Output image path.")
    ir.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.set_defaults(func=cmd_img)

    p = sub.add_parser("install", parents=[parent],
                       help="Set up a Python environment and install the toolchains.")
    inst_sub = p.add_subparsers(dest="install_cmd")
    itx = inst_sub.add_parser("txt", help="Create a venv + install text/LLM tools (torch, transformers, trl, ...).")
    itx.add_argument("--dry-run", action="store_true", help="Print what would run without installing.")
    iim = inst_sub.add_parser("img", help="Install image tools (diffusers, torchvision, peft, ...).")
    iim.add_argument("--dry-run", action="store_true", help="Print what would run without installing.")
    ito = inst_sub.add_parser("torch", help="Install the ML toolchain (torch, transformers, trl, ...).")
    ito.add_argument("--dry-run", action="store_true", help="Print what would run without installing.")
    ipy = inst_sub.add_parser("python", help="Attempt to install Python via the OS package manager.")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("help", parents=[parent],
                       help="Show a full usage guide with examples for every command.")
    p.set_defaults(func=cmd_help)

    return parser


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    cli()
