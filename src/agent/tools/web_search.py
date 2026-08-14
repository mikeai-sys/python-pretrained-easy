# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import os
import time
import urllib.parse

from src.config import Config


class WebSearch:
    """Web search to seed the crawler.

    Providers (set ``tools.web_search.provider``):
      - duckduckgo : no key, rate-limited (default)
      - brave      : needs BRAVE_API_KEY
      - serpapi    : needs SERPAPI_KEY
      - google     : needs GOOGLE_API_KEY + GOOGLE_CSE_ID (Programmable Search)
      - searlo     : needs SEARLO_API_KEY (CSE replacement; full-web Google index)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        sec = config.raw.get("tools", {}).get("web_search", {})
        # PPTE_SEARCH_PROVIDER (set by `ppte config`) overrides the config choice.
        self.provider = os.environ.get("PPTE_SEARCH_PROVIDER") or sec.get("provider", "duckduckgo")
        self.max_results = sec.get("max_results", 10)
        self.retries = sec.get("retries", 3)
        self.retry_delay = sec.get("retry_delay", 5)
        g = sec.get("google", {})
        sl = sec.get("searlo", {})
        # Load every keyed provider's keys up front so the fallback chain can
        # detect them regardless of which provider is configured.
        self.google_api_key = config.api_key(g.get("api_key_env", "GOOGLE_API_KEY"))
        self.google_cse_id = config.api_key(g.get("cse_id_env", "GOOGLE_CSE_ID"))
        self.brave_key = config.api_key(sec.get("brave_api_key_env", "BRAVE_API_KEY"))
        self.serpapi_key = config.api_key(sec.get("serpapi_api_key_env", "SERPAPI_KEY"))
        self.searlo_url = sl.get("api_url", "https://api.searlo.tech/api/v1")
        self.searlo_key = config.api_key(sl.get("api_key_env", "SEARLO_API_KEY"))
        # "html" backend avoids DuckDuckGo's throttled JSON (i.js) endpoint.
        self.backend = sec.get("backend", "html")

    @staticmethod
    def _retry_ddg(fn, tries: int = 3, delay: int = 5):
        """Retry a DuckDuckGo call; it rate-limits (403) aggressively from some IPs."""
        from duckduckgo_search.exceptions import RatelimitException

        last: Exception | None = None
        for attempt in range(tries):
            try:
                return fn()
            except RatelimitException as exc:
                last = exc
                if attempt < tries - 1:
                    print(f"[search] rate-limited, retry {attempt + 1}/{tries} in {delay}s")
                    time.sleep(delay)
        raise last  # type: ignore[misc]

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        return self._resolve("text", query, max_results or self.max_results)

    def search_images(self, query: str, max_results: int | None = None) -> list[dict]:
        return self._resolve("image", query, max_results or self.max_results)

    def search_videos(self, query: str, max_results: int | None = None) -> list[dict]:
        return self._resolve("video", query, max_results or self.max_results)

    def _has_key(self, prov: str, kind: str) -> bool:
        if prov == "google":
            return kind != "video" and bool(self.google_api_key and self.google_cse_id)
        if prov == "brave":
            return kind == "text" and bool(self.brave_key)
        if prov == "serpapi":
            return kind == "text" and bool(self.serpapi_key)
        if prov == "searlo":
            return kind == "text" and bool(self.searlo_key)
        return False

    def _candidates(self, kind: str) -> list[str]:
        """Providers to try, in priority order, skipping ones missing keys.

        An explicitly configured keyed provider is tried first; otherwise any
        keyed provider with keys is preferred over DuckDuckGo (which rate-limits
        aggressively from some IPs). DuckDuckGo is always the final fallback.
        """
        prefs: list[str] = []
        if self.provider != "duckduckgo":
            prefs.append(self.provider)
        for p in ("google", "brave", "serpapi", "searlo"):
            if p not in prefs and self._has_key(p, kind):
                prefs.append(p)
        prefs.append("duckduckgo")
        seen: set[str] = set()
        return [p for p in prefs if not (p in seen or seen.add(p))]

    def _resolve(self, kind: str, query: str, max_results: int) -> list[dict]:
        last: Exception | None = None
        for prov in self._candidates(kind):
            try:
                return self._dispatch(prov, kind, query, max_results)
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[search] {prov} ({kind}) failed: {exc}; falling back")
        raise last or RuntimeError("no usable search provider")

    def _dispatch(self, prov: str, kind: str, query: str, max_results: int) -> list[dict]:
        if prov == "duckduckgo":
            if kind == "text":
                return self._duckduckgo(query, max_results)
            if kind == "image":
                return self._duckduckgo_images(query, max_results)
            return self._duckduckgo_videos(query, max_results)
        if prov == "brave":
            return self._brave(query, max_results)
        if prov == "serpapi":
            return self._serpapi(query, max_results)
        if prov == "google":
            return self._google(query, max_results, search_type="image" if kind == "image" else None)
        if prov == "searlo":
            return self._searlo(query, max_results, search_type="image" if kind == "image" else None)
        raise ValueError(f"unknown provider: {prov}")

    def _google(self, query: str, max_results: int, search_type: str | None = None) -> list[dict]:
        import requests

        if not self.google_api_key or not self.google_cse_id:
            raise RuntimeError(
                "set GOOGLE_API_KEY and GOOGLE_CSE_ID (see tools.web_search.google in config)"
            )
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(max_results, 10),
        }
        if search_type:
            params["searchType"] = search_type
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1", params=params, timeout=15
        )
        resp.raise_for_status()
        out = []
        for it in resp.json().get("items", []):
            rec = {
                "title": it.get("title"),
                "url": it.get("link"),
                "snippet": it.get("snippet"),
            }
            if search_type == "image":
                rec["thumbnail"] = it.get("image", {}).get("thumbnailLink")
                rec["source"] = it.get("image", {}).get("contextLink")
            out.append(rec)
        return out

    def _searlo(self, query: str, max_results: int, search_type: str | None = None) -> list[dict]:
        """Searlo — a CSE replacement that queries the full Google index.

        ``GET {api_url}/search`` with the ``X-API-Key`` header. Results live in
        the ``organic`` array (``title``/``link``/``snippet``); image searches
        add a ``thumbnail`` (the image) and keep ``link`` as the source page.
        The base URL is configurable via ``tools.web_search.searlo.api_url``.
        """
        import requests

        if not self.searlo_key:
            raise RuntimeError("set tools.web_search.searlo.api_key_env (SEARLO_API_KEY)")
        url = self.searlo_url.rstrip("/") + "/search"
        # Paginate (Searlo returns ~10/request) until we have ``max_results``.
        items: list[dict] = []
        page = 1
        while len(items) < max_results and page <= 10:
            params = {"q": query, "num": min(max_results, 10), "page": page}
            if search_type:
                params["type"] = search_type
            resp = requests.get(url, headers={"X-API-Key": self.searlo_key}, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") is False:
                raise RuntimeError(data.get("message", "searlo request failed"))
            batch = data.get("organic", [])
            if not batch:
                break
            items.extend(batch)
            if not data.get("nextPage"):
                break
            page += 1
        out = []
        for it in items[:max_results]:
            if search_type == "image":
                rec = {
                    "title": it.get("title"),
                    "url": it.get("thumbnail") or it.get("link"),
                    "source": it.get("link"),
                }
            else:
                rec = {
                    "title": it.get("title"),
                    "url": it.get("link"),
                    "snippet": it.get("snippet"),
                }
            out.append(rec)
        return out

    def _duckduckgo(self, query: str, max_results: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError("pip install duckduckgo_search") from e

        def _call():
            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append(
                        {
                            "title": r.get("title"),
                            "url": r.get("href"),
                            "snippet": r.get("body"),
                        }
                    )
            return out

        return self._retry_ddg(_call, self.retries, self.retry_delay)

    def _duckduckgo_images(self, query: str, max_results: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError("pip install duckduckgo_search") from e

        def _call():
            out = []
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=max_results):
                    out.append(
                        {
                            "title": r.get("title"),
                            "url": r.get("image"),       # direct image url
                            "source": r.get("url"),       # source page
                        }
                    )
            return out

        return self._retry_ddg(_call, self.retries, self.retry_delay)

    def _duckduckgo_videos(self, query: str, max_results: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError("pip install duckduckgo_search") from e

        def _call():
            out = []
            with DDGS() as ddgs:
                for r in ddgs.videos(query, max_results=max_results):
                    out.append(
                        {
                            "title": r.get("title"),
                            "url": r.get("url"),          # watch page
                            "thumbnail": r.get("thumbnail"),
                        }
                    )
            return out

        return self._retry_ddg(_call, self.retries, self.retry_delay)

    def _brave(self, query: str, max_results: int) -> list[dict]:
        import requests

        if not self.brave_key:
            raise RuntimeError("set tools.web_search.brave_api_key_env (BRAVE_API_KEY)")
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.brave_key},
            params={"q": query, "count": max_results},
            timeout=15,
        )
        resp.raise_for_status()
        return [
            {"title": w.get("title"), "url": w.get("url"), "snippet": w.get("description")}
            for w in resp.json().get("web", {}).get("results", [])
        ]

    def _serpapi(self, query: str, max_results: int) -> list[dict]:
        import requests

        if not self.serpapi_key:
            raise RuntimeError("set tools.web_search.serpapi_api_key_env (SERPAPI_KEY)")
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "num": max_results, "api_key": self.serpapi_key},
            timeout=15,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
            for r in resp.json().get("organic_results", [])
        ]
