# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import time
import urllib.parse
from pathlib import Path

from src.config import Config
from src.agent.tools.web_search import WebSearch
from src.agent.tools.crawl import Crawler


SYNTHESIS_PROMPT = """You are a dataset curator. Read the web content below and produce
up to {k} high-quality (instruction, response) training pairs an AI assistant
could learn from. Prefer factual, self-contained tasks (explain, summarize,
extract, compare). Respond with ONLY a JSON array:
[{{"instruction": "...", "response": "..."}}]

CONTENT:
{content}
"""


class DataCollector:
    """Build a local training dataset from the web.

    Two entry points:
      - ``collect(query, dataset_type)`` : search -> crawl -> dataset
      - ``fetch(urls, dataset_type)``    : crawl given URLs directly (no search)

    ``dataset_type``:
      - "text"  : pages -> (instruction, response) pairs (needs the local model)
      - "image" : images + captions derived from each image's source-page text
      - "video" : video URLs + thumbnails

    Raw media and the derived dataset live under ``paths.crawl_dir/<run_id>/``.
    Call ``cleanup(run_id)`` to delete a run.
    """

    def __init__(self, config: Config, model: LocalModel | None = None) -> None:
        self.config = config
        self.model = model
        self.search = WebSearch(config)
        self.crawler = Crawler(config)
        self.dc_cfg = config.raw.get("data_collection", {})
        self.root = Path(config.get("paths", "crawl_dir", default="data/crawl"))
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- dispatch ---------------------------------------------------------
    def collect(self, query: str, dataset_type: str = "text", raw: bool = False, parsed: bool = True, limit: int | None = None) -> tuple[list[dict], str]:
        if dataset_type == "text":
            return self._collect_text(query, raw=raw, parsed=parsed, limit=limit)
        if dataset_type == "image":
            return self._collect_image(query, limit)
        if dataset_type == "video":
            return self._collect_video(query, limit)
        raise ValueError(f"unknown dataset_type: {dataset_type}")

    def fetch(self, urls: list[str], dataset_type: str = "text", raw: bool = False, parsed: bool = True, limit: int | None = None) -> tuple[list[dict], str]:
        urls = [u for u in urls if u]
        if limit is not None:
            urls = urls[:limit]
        if dataset_type == "text":
            return self._collect_raw(urls, parsed=parsed, limit=limit) if raw else self._synthesize(urls, limit=limit)
        if dataset_type == "image":
            items = [{"title": "", "url": u, "source": u} for u in urls]
            return self._save_images(items)
        if dataset_type == "video":
            items = [{"title": "", "url": u, "thumbnail": None} for u in urls]
            return self._save_videos(items)
        raise ValueError(f"unknown dataset_type: {dataset_type}")

    def _collect_raw(self, urls: list[str], parsed: bool = True, limit: int | None = None) -> tuple[list[dict], str]:
        """Collect from each URL with no model (fast webfetch).

        With ``parsed=True`` (default) the embedded :class:`ContentParser`
        structures each page into headings/paragraphs/lists/code/tables/links,
        written to ``parsed.jsonl``. With ``parsed=False`` it dumps the flat
        cleaned text as before (``corpus.jsonl``). ``limit`` caps how many URLs
        are actually crawled.
        """
        run_id = f"run_{int(time.time())}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if parsed:
            docs = self.crawler.crawl_parsed(urls, max_pages=limit)
            records = docs
            out_name = "parsed.jsonl"
            print(f"[fetch:raw] parsed {len(docs)} pages -> {run_dir}/{out_name}")
        else:
            docs = self.crawler.crawl(urls, max_pages=limit)
            records = [{"text": d["text"], "meta": {"url": d["url"]}} for d in docs]
            out_name = "corpus.jsonl"
            print(f"[fetch:raw] collected {len(docs)} pages -> {run_dir}/{out_name}")

        with open(run_dir / out_name, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return records, run_id

    # ---- text -------------------------------------------------------------
    def _collect_text(self, query: str, raw: bool = False, parsed: bool = True, limit: int | None = None) -> tuple[list[dict], str]:
        max_results = limit or self.dc_cfg.get("max_results_per_query", 10)
        results = self.search.search(query, max_results)
        urls = [r["url"] for r in results if r.get("url")]
        print(f"[collect:text] query='{query}' -> {len(urls)} urls")
        if not urls:
            print(
                "[collect:text] no search results. DuckDuckGo is rate-limited from this IP; "
                "set a keyed provider (google/brave/serpapi) in config — see `ppte config`."
            )
            return [], "run_none"
        if raw:
            return self._collect_raw(urls, parsed=parsed, limit=limit)
        return self._synthesize(urls, limit=limit)

    def _synthesize(self, urls: list[str], limit: int | None = None) -> tuple[list[dict], str]:
        k = self.dc_cfg.get("examples_per_doc", 2)
        docs = self.crawler.crawl(urls, max_pages=limit)
        run_id = f"run_{int(time.time())}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "raw.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        examples = self._build_examples(docs, k)
        with open(run_dir / "dataset.jsonl", "w") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")

        print(f"[collect:text] {len(docs)} pages -> {len(examples)} examples @ {run_dir}")
        return examples, run_id

    def _build_examples(self, docs: list[dict], k: int) -> list[dict]:
        examples: list[dict] = []
        for d in docs:
            content = d["text"][: self.dc_cfg.get("max_chars_per_doc", 4000)]
            prompt = SYNTHESIS_PROMPT.format(k=k, content=content)
            raw = self.model.chat(
                [{"role": "user", "content": prompt}], temperature=0.8, max_new_tokens=1024
            )
            for pair in self._parse_pairs(raw):
                examples.append(
                    {
                        "task": pair.get("instruction", ""),
                        "solution": pair.get("response", ""),
                        "meta": {"source": "web", "url": d["url"]},
                    }
                )
        return examples

    @staticmethod
    def _parse_pairs(raw: str) -> list[dict]:
        import re

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            return [p for p in data if isinstance(p, dict) and p.get("instruction")]
        except (ValueError, TypeError):
            return []

    # ---- image ------------------------------------------------------------
    def _collect_image(self, query: str, limit: int | None = None) -> tuple[list[dict], str]:
        """Search for ``query`` images and save (download) up to ``limit`` of them.

        ``limit`` is the number of images you want on disk. We over-fetch from the
        search provider (by ``image_overfetch``) to absorb download failures, then
        stop once ``limit`` images have been saved.
        """
        n = limit or self.dc_cfg.get("max_media", 20)
        overfetch = self.dc_cfg.get("image_overfetch", 1.3)
        candidates = self.search.search_images(query, max(1, int(n * overfetch)))
        print(f"[collect:image] query='{query}' -> {len(candidates)} candidates, want {n}")
        return self._save_images(candidates, target=n)

    @staticmethod
    def _image_ext(url: str) -> str:
        """Pick a sane file extension from the URL path, else content-type, else .jpg."""
        path = urllib.parse.urlparse(url).path
        ext = Path(path).suffix
        if ext and 1 < len(ext) <= 5:
            return ext
        return ".jpg"

    def _save_images(self, items: list[dict], target: int | None = None) -> tuple[list[dict], str]:
        run_id = f"run_{int(time.time())}"
        run_dir = self.root / run_id
        img_dir = run_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        cap_from_src = self.dc_cfg.get("caption_from_source", True)
        cap_cache: dict[str, str | None] = {}
        records: list[dict] = []
        saved = 0
        for it in items:
            ext = self._image_ext(it.get("url", ""))
            dest = img_dir / f"{len(records):04d}{ext}"
            img_path = None
            if self.dc_cfg.get("download_media", True):
                try:
                    self.crawler.download_binary(
                        it["url"], str(dest), retries=1, referer=it.get("source")
                    )
                    img_path = str(dest)
                    saved += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[collect:image] skip {it.get('url')}: {exc}")
                self.crawler._sleep()

            # Caption: prefer text from the image's source page (title + meta
            # description + lead), falling back to the search-result title.
            caption = it.get("title", "")
            if cap_from_src and it.get("source"):
                src = it["source"]
                if src not in cap_cache:
                    cap_cache[src] = self._caption_from_source(src)
                if cap_cache[src]:
                    caption = cap_cache[src]

            rec = {"caption": caption, "meta": {"source": it.get("source"), "query": it.get("query")}}
            if img_path:
                rec["image_path"] = img_path
            else:
                rec["image_url"] = it.get("url")
            records.append(rec)

            # Stop once we have the requested number of downloaded images.
            if target is not None and saved >= target:
                break
        with open(run_dir / "images.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[collect:image] saved {saved}/{len(records)} images @ {run_dir}")
        return records, run_id

    def _caption_from_source(self, url: str) -> str | None:
        """Build a caption from a page's meta description / title (no model)."""
        try:
            docs = self.crawler.crawl([url])
        except Exception as exc:  # noqa: BLE001
            print(f"[collect:image] caption fetch failed {url}: {exc}")
            return None
        if not docs:
            return None
        d = docs[0]
        cap = (d.get("description") or "").strip() or self._clean_title(d.get("title") or "")
        return cap[: self.dc_cfg.get("max_caption_chars", 300)] or None

    @staticmethod
    def _clean_title(title: str) -> str:
        # Drop a trailing site name, e.g. "Red fox - Wikipedia" -> "Red fox".
        return re.sub(r"\s*[-–|]\s*[^[-]+$", "", title).strip()

    # ---- video ------------------------------------------------------------
    def _collect_video(self, query: str, limit: int | None = None) -> tuple[list[dict], str]:
        n = limit or self.dc_cfg.get("max_media", 20)
        overfetch = self.dc_cfg.get("image_overfetch", 1.3)
        candidates = self.search.search_videos(query, max(1, int(n * overfetch)))
        print(f"[collect:video] query='{query}' -> {len(candidates)} candidates, want {n}")
        return self._save_videos(candidates, target=n)

    def _save_videos(self, items: list[dict], target: int | None = None) -> tuple[list[dict], str]:
        run_id = f"run_{int(time.time())}"
        run_dir = self.root / run_id
        thumb_dir = run_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        saved = 0
        for it in items:
            rec = {"url": it.get("url"), "title": it.get("title", ""), "meta": {"query": it.get("query")}}
            if self.dc_cfg.get("download_media", True) and it.get("thumbnail"):
                dest = thumb_dir / f"{len(records):04d}.jpg"
                try:
                    self.crawler.download_binary(it["thumbnail"], str(dest), retries=1)
                    rec["thumbnail_path"] = str(dest)
                    saved += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[collect:video] thumb skip: {exc}")
            records.append(rec)
            self.crawler._sleep()
            if target is not None and saved >= target:
                break
        with open(run_dir / "videos.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[collect:video] saved {saved}/{len(records)} videos @ {run_dir}")
        return records, run_id

    def cleanup(self, run_id: str) -> None:
        target = self.root / run_id
        if target.exists():
            shutil.rmtree(target)
            print(f"[cleanup] deleted crawl data: {target}")
