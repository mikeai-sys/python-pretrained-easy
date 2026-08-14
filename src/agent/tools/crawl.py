# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.robotparser

import requests
from bs4 import BeautifulSoup

from src.config import Config


class BrowserTool:
    """Full browser automation via Playwright (JS-rendered pages, clicks, etc.).

    Optional — needs `playwright install chromium`. Falls back to plain HTTP
    fetching in Crawler when disabled.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sec = config.raw.get("tools", {}).get("browser", {})
        self.timeout = self.sec.get("timeout", 20000)
        self.headless = self.sec.get("headless", True)

    def get_text(self, url: str) -> str:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.goto(url, timeout=self.timeout, wait_until="networkidle")
            text = page.inner_text()
            browser.close()
        return text or ""

    def get_html(self, url: str) -> str:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.goto(url, timeout=self.timeout, wait_until="networkidle")
            html = page.content()
            browser.close()
        return html or ""


class Crawler:
    """Crawls a list of URLs into clean text.

    - Respects robots.txt by default.
    - Rate-limited with a configurable delay between requests.
    - Uses Playwright when ``tools.browser.enabled`` is true, else plain
      requests + BeautifulSoup (faster, lighter).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        sec = config.raw.get("tools", {}).get("crawl", {})
        self.max_pages = sec.get("max_pages", 5)
        self.respect_robots = sec.get("respect_robots", True)
        self.delay = sec.get("delay", 1.0)
        self.timeout = sec.get("timeout", 10)
        self.user_agent = sec.get(
            "user_agent", "SelfImprovingAgent/0.1 (+educational research crawler)"
        )
        self.use_browser = config.raw.get("tools", {}).get("browser", {}).get("enabled", False)
        self._rp_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._browser: BrowserTool | None = BrowserTool(config) if self.use_browser else None

    def crawl(self, urls: list[str], max_pages: int | None = None) -> list[dict]:
        cap = max_pages if max_pages is not None else self.max_pages
        docs: list[dict] = []
        for url in urls:
            if len(docs) >= cap:
                break
            if not self._allowed(url):
                continue
            try:
                meta = self._fetch(url)
            except Exception as exc:  # noqa: BLE001
                print(f"[crawl] skip {url}: {exc}")
                continue
            text = meta.get("text", "")
            if text.strip():
                docs.append(
                    {
                        "url": url,
                        "text": text,
                        "title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                    }
                )
            self._sleep()
        return docs

    def fetch_html(self, url: str) -> str | None:
        """Return raw page HTML (respecting robots), or None on failure."""
        if not self._allowed(url):
            return None
        try:
            if self._browser is not None:
                return self._browser.get_html(url)
            resp = requests.get(
                url, headers={"User-Agent": self.user_agent}, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            print(f"[crawl] skip {url}: {exc}")
            return None

    def crawl_parsed(self, urls: list[str], max_pages: int | None = None) -> list[dict]:
        """Fetch pages and run each through the embedded :class:`ContentParser`."""
        from src.agent.parser import ContentParser

        cap = max_pages if max_pages is not None else self.max_pages
        parser = ContentParser()
        docs: list[dict] = []
        for url in urls:
            if len(docs) >= cap:
                break
            html = self.fetch_html(url)
            if html:
                docs.append(parser.parse(html, url))
            self._sleep()
        return docs

    def _fetch(self, url: str) -> dict:
        if self._browser is not None:
            text = self._browser.get_text(url)
            return {"text": text, "title": "", "description": ""}
        resp = requests.get(
            url, headers={"User-Agent": self.user_agent}, timeout=self.timeout
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta_desc and meta_desc.get("content"):
            desc = meta_desc["content"].strip()
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return {
            "text": soup.get_text(separator="\n", strip=True),
            "title": title,
            "description": desc,
        }

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urllib.parse.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._rp_cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                # Fetch + decode manually: some sites (e.g. Wikipedia) serve a UTF-8
                # BOM that breaks RobotFileParser.read() and yields disallow_all.
                r = requests.get(
                    base + "/robots.txt",
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                rp.parse(r.content.decode("utf-8-sig", "replace").splitlines())
            except Exception:  # noqa: BLE001
                rp = None
            self._rp_cache[base] = rp
        rp = self._rp_cache[base]
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def extract_links(self, url: str, match: str | None = None, same_domain: bool = False) -> list[str]:
        """Pull article URLs out of a page or sitemap XML.

        Used to build a ``urls.txt`` for ``fetch -f`` — e.g. point it at a
        category page or a ``sitemap.xml`` to harvest many article links at once.
        """
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"SSL error fetching {url} (certificate expired or untrusted). "
                "Make sure the URL is a real site, then try again."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Could not fetch {url}: {exc}") from exc
        content = resp.text

        if "<loc>" in content:  # sitemap
            found = re.findall(r"<loc>(.*?)</loc>", content, re.DOTALL)
        else:  # html page
            soup = BeautifulSoup(content, "lxml")
            found = [urllib.parse.urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

        seen: set[str] = set()
        out: list[str] = []
        for link in found:
            link = link.strip()
            if link and link not in seen:
                seen.add(link)
                out.append(link)

        if same_domain:
            netloc = urllib.parse.urlparse(url).netloc
            out = [l for l in out if urllib.parse.urlparse(l).netloc == netloc]
        if match:
            out = [l for l in out if re.search(match, l)]
        # drop fragment anchors (they point at the same page)
        out = [l.split("#")[0] for l in out]
        # re-dedupe after fragment stripping
        seen2: set[str] = set()
        out = [l for l in out if not (l in seen2 or seen2.add(l))]
        return out

    def download_binary(
        self,
        url: str,
        path: str,
        max_bytes: int = 25_000_000,
        retries: int = 1,
        referer: str | None = None,
    ) -> int:
        """Download a file (image/video/thumbnail) to ``path``. Returns bytes written.

        Retries once with a browser-like User-Agent (and an optional ``Referer``
        for hotlink-protected hosts) so transient 403s don't silently drop the
        file. Follows redirects.
        """
        last: Exception | None = None
        headers = {"User-Agent": self.user_agent}
        if referer:
            headers["Referer"] = referer
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    headers["User-Agent"] = (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True,
                )
                resp.raise_for_status()
                written = 0
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                        if written > max_bytes:
                            break
                return written
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < retries:
                    self._sleep()
        raise last  # type: ignore[misc]
