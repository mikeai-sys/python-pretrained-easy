# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup


class ContentParser:
    """Tiny embedded content extractor — a Readability-lite heuristic, no ML.

    Turns raw page HTML into a structured record: title, meta description,
    headings, paragraphs, lists, code blocks, tables and links, plus a clean
    ``main_text`` prose view. Boilerplate (nav/footer/ads) is dropped and very
    short paragraphs are filtered, which is the "tiny model" that decides what
    is real content vs. noise.
    """

    _BOILERPLATE_TAGS = (
        "script", "style", "nav", "footer", "header", "aside",
        "noscript", "form", "iframe", "svg", "figure",
    )

    def parse(self, html: str, url: str = "") -> dict:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(self._BOILERPLATE_TAGS):
            tag.decompose()

        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        desc = ""
        md = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if md and md.get("content"):
            desc = md["content"].strip()

        # Focus on the article body when the page marks one.
        root = soup.find("main") or soup.find("article") or soup.body or soup

        # Walk the document in order so headings, paragraphs, lists, code and
        # tables keep their original interleaving. ``blocks`` is the
        # document-order spine used for Markdown rendering and section
        # extraction; the separate *_arrays below are kept for backward
        # compatibility and quick prose flattening.
        candidates = root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "pre", "table"]
        )

        blocks: list[dict] = []
        headings = []
        paragraphs = []
        lists: list[list[str]] = []
        code_blocks = []
        tables: list[list[list[str]]] = []

        for el in candidates:
            name = el.name
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    level = int(name[1])
                    headings.append({"level": level, "text": txt})
                    blocks.append({"type": "heading", "level": level, "text": txt})
            elif name == "p":
                txt = el.get_text(" ", strip=True)
                if len(txt) > 20:
                    paragraphs.append(txt)
                    blocks.append({"type": "paragraph", "text": txt})
            elif name in ("ul", "ol"):
                items = [li.get_text(" ", strip=True) for li in el.find_all("li")]
                items = [i for i in items if i]
                if items:
                    lists.append(items)
                    blocks.append(
                        {"type": "list", "ordered": name == "ol", "items": items}
                    )
            elif name == "pre":
                code = el.get_text()
                if code:
                    code_blocks.append(code)
                    blocks.append({"type": "code", "text": code})
            elif name == "table":
                rows = []
                for tr in el.find_all("tr"):
                    cells = [
                        c.get_text(" ", strip=True)
                        for c in tr.find_all(["td", "th"])
                    ]
                    if cells:
                        rows.append(cells)
                if rows:
                    tables.append(rows)
                    blocks.append({"type": "table", "rows": rows})

        links: list[dict] = []
        for a in root.find_all("a", href=True):
            txt = a.get_text(" ", strip=True)
            href = urljoin(url, a["href"])
            if txt and href.startswith(("http://", "https://")):
                links.append({"text": txt, "href": href})

        main_text = "\n\n".join(
            filter(
                None,
                [  # noqa: E501
                    "\n".join(paragraphs),
                    *["\n".join(items) for items in lists],
                    *code_blocks,
                ],
            )
        )

        return {
            "url": url,
            "title": title,
            "description": desc,
            "headings": headings,
            "paragraphs": paragraphs,
            "lists": lists,
            "code_blocks": code_blocks,
            "tables": tables,
            "links": links,
            "blocks": blocks,
            "main_text": main_text,
        }


def render_markdown(record: dict) -> str:
    """Render a parsed ``ContentParser`` record into clean Markdown.

    Uses the document-order ``blocks`` spine so the output preserves the
    original heading/paragraph/list/code/table interleaving.
    """
    lines: list[str] = []
    if record.get("title"):
        lines.append(f"# {record['title']}")
        lines.append("")
    if record.get("description"):
        lines.append(f"> {record['description']}")
        lines.append("")
    if record.get("url"):
        lines.append(f"*Source: {record['url']}*")
        lines.append("")

    for b in record.get("blocks", []):
        btype = b.get("type")
        if btype == "heading":
            level = b.get("level", 2)
            lines.append(f"{'#' * level} {b.get('text', '')}")
            lines.append("")
        elif btype == "paragraph":
            lines.append(b.get("text", ""))
            lines.append("")
        elif btype == "list":
            for i, item in enumerate(b.get("items", []), 1):
                if b.get("ordered"):
                    lines.append(f"{i}. {item}")
                else:
                    lines.append(f"- {item}")
            lines.append("")
        elif btype == "code":
            lines.append("```")
            lines.append(b.get("text", "").rstrip("\n"))
            lines.append("```")
            lines.append("")
        elif btype == "table":
            rows = b.get("rows", [])
            if rows:
                header = rows[0]
                sep = ["---"] * len(header)
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(sep) + " |")
                for row in rows[1:]:
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def extract_section(record: dict, name: str) -> dict | None:
    """Return a copy of ``record`` containing only the section under ``name``.

    Finds the first heading whose text contains ``name`` (case-insensitive)
    and slices ``blocks`` up to the next heading of equal-or-higher level.
    Returns ``None`` when no matching heading is found.
    """
    blocks = record.get("blocks", [])
    needle = name.lower()
    start = None
    start_level = None
    for i, b in enumerate(blocks):
        if b.get("type") == "heading" and needle in b.get("text", "").lower():
            start = i
            start_level = b.get("level", 2)
            break
    if start is None:
        return None

    end = len(blocks)
    for j in range(start + 1, len(blocks)):
        b = blocks[j]
        if b.get("type") == "heading" and b.get("level", 2) <= start_level:
            end = j
            break

    section = dict(record)
    section["blocks"] = blocks[start:end]
    return section


def render_section(record: dict, name: str) -> str | None:
    """Render a single named section of ``record`` to Markdown, or ``None``."""
    section = extract_section(record, name)
    if section is None:
        return None
    return render_markdown(section)
