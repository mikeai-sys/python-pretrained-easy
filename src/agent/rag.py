# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass
class Chunk:
    text: str
    meta: dict = field(default_factory=dict)


def _record_text(rec: dict) -> str:
    """Flatten a parsed record into prose, preferring the ordered ``blocks``."""
    blocks = rec.get("blocks")
    if blocks:
        parts: list[str] = []
        for b in blocks:
            btype = b.get("type")
            if btype == "heading":
                parts.append(b.get("text", ""))
            elif btype == "paragraph":
                parts.append(b.get("text", ""))
            elif btype == "list":
                parts.append(" ".join(b.get("items", [])))
            elif btype == "code":
                parts.append(b.get("text", ""))
        return "\n\n".join(p for p in parts if p)
    return rec.get("main_text") or rec.get("text") or ""


def _split(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + size])
        start += max(1, size - overlap)
    return out


def load_chunks(parsed_jsonl: str, chunk_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Turn a ``parsed.jsonl`` file into a flat list of text chunks."""
    chunks: list[Chunk] = []
    with open(parsed_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _record_text(rec)
            if not text:
                continue
            for piece in _split(text, chunk_chars, overlap):
                chunks.append(
                    Chunk(piece, {"url": rec.get("url"), "title": rec.get("title")})
                )
    return chunks


def _embed(texts: list[str], model_name: str, device: str):
    """Return a numpy matrix of embeddings. Requires sentence-transformers."""
    if np is None:
        raise RuntimeError("pip install numpy")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RAG needs sentence-transformers: pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name, device=device if device != "auto" else None)
    return np.asarray(
        model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        dtype="float32",
    )


def build_index(
    parsed_jsonl: str,
    out_dir: str,
    model_name: str,
    device: str = "auto",
    chunk_chars: int = 800,
    overlap: int = 100,
) -> int:
    """Embed ``parsed_jsonl`` and write a numpy index to ``out_dir``.

    Writes ``embeddings.npy``, ``chunks.jsonl`` and ``config.json``. Uses a
    FAISS ``IndexFlatIP`` when ``faiss`` is installed, otherwise falls back to
    numpy dot-product similarity (fine for modest corpora).
    """
    chunks = load_chunks(parsed_jsonl, chunk_chars, overlap)
    if not chunks:
        raise ValueError("no chunks produced — is the parsed.jsonl empty?")
    embeddings = _embed([c.text for c in chunks], model_name, device)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    with open(out / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.text, "meta": c.meta}) + "\n")
    (out / "config.json").write_text(
        json.dumps({"model": model_name, "device": device})
    )
    # Optional FAISS accelerator index.
    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(out / "faiss.index"))
    except ImportError:
        pass
    return len(chunks)


def _load_index(index_dir: str):
    index = Path(index_dir)
    embeddings = np.load(index / "embeddings.npy")
    chunks: list[dict] = []
    with open(index / "chunks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    cfg = json.loads((index / "config.json").read_text())
    return embeddings, chunks, cfg


def query_index(
    index_dir: str,
    question: str,
    model_name: str | None = None,
    device: str | None = None,
    k: int = 5,
    use_faiss: bool = True,
) -> list[tuple[dict, float]]:
    """Return the ``k`` most similar chunks to ``question`` as (chunk, score)."""
    if np is None:
        raise RuntimeError("pip install numpy")
    embeddings, chunks, cfg = _load_index(index_dir)
    q_emb = _embed([question], model_name or cfg["model"], device or cfg.get("device", "auto"))
    q = q_emb[0]

    try:
        import faiss

        if use_faiss and (Path(index_dir) / "faiss.index").exists():
            index = faiss.read_index(str(Path(index_dir) / "faiss.index"))
            faiss.normalize_L2(q.reshape(1, -1))
            scores, ids = index.search(q.reshape(1, -1), k)
            out: list[tuple[dict, float]] = []
            for score, i in zip(scores[0], ids[0]):
                if i >= 0:
                    out.append((chunks[i], float(score)))
            return out
    except ImportError:
        pass

    # Numpy fallback: cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1)
    qn = np.linalg.norm(q)
    sims = embeddings @ q / np.where(norms == 0, 1, norms) / (qn or 1)
    order = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in order]


def build_prompt(question: str, results: list[tuple[dict, float]], max_chars: int = 3000) -> str:
    """Assemble a context window from retrieved chunks for an LLM to answer with."""
    ctx = []
    used = 0
    for chunk, _score in results:
        text = chunk["text"]
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        ctx.append(f"[{chunk.get('meta', {}).get('title', 'source')}]\n{text}")
        used += len(text)
        if used >= max_chars:
            break
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say so.\n\n"
        f"CONTEXT:\n{'---'.join(ctx)}\n\nQUESTION: {question}\n\nANSWER:"
    )
