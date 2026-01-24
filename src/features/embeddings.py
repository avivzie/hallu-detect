# src/features/embeddings.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for embedding extraction.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_col: str = "answer"  # or "response" depending on your dataframe schema
    normalize: bool = True
    batch_size: int = 64
    device: Optional[str] = None  # e.g., "cpu", "cuda", "mps"
    max_length: Optional[int] = None  # None = model default
    cache_dir: str = "artifacts/embeddings"


def _safe_model_id(model_name: str) -> str:
    # filesystem-safe model id
    return model_name.replace("/", "__").replace(":", "_")


def _hash_texts(texts: List[str]) -> str:
    """
    Stable hash for a list of texts (order-sensitive).
    Used to create cache keys.
    """
    h = hashlib.sha256()
    for t in texts:
        if t is None:
            t = ""
        # preserve boundaries
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n<<SEP>>\n")
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _npz_paths(cache_root: Path, cache_key: str) -> Tuple[Path, Path]:
    """
    Return (data_path, meta_path)
    """
    data_path = cache_root / f"{cache_key}.npz"
    meta_path = cache_root / f"{cache_key}.meta.json"
    return data_path, meta_path


def _load_cached(cache_root: Path, cache_key: str) -> Optional[np.ndarray]:
    data_path, _ = _npz_paths(cache_root, cache_key)
    if not data_path.exists():
        return None
    try:
        with np.load(data_path) as z:
            return z["embeddings"]
    except Exception:
        # corrupted cache -> ignore
        return None


def _save_cached(cache_root: Path, cache_key: str, embeddings: np.ndarray, meta: dict) -> None:
    _ensure_dir(cache_root)
    data_path, meta_path = _npz_paths(cache_root, cache_key)
    np.savez_compressed(data_path, embeddings=embeddings.astype(np.float32, copy=False))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def embed_texts(
    texts: List[str],
    cfg: EmbeddingConfig,
    *,
    cache_tag: Optional[str] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts using Sentence-Transformers.

    Caching strategy:
      cache_key = sha256(texts) + model/params + optional cache_tag
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Missing dependency: sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from e

    # sanitize inputs
    clean_texts = [("" if t is None else str(t)) for t in texts]

    cache_root = Path(cfg.cache_dir) / _safe_model_id(cfg.model_name)
    params_fingerprint = {
        "model_name": cfg.model_name,
        "normalize": cfg.normalize,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "max_length": cfg.max_length,
        "cache_tag": cache_tag,
    }
    key_payload = json.dumps(params_fingerprint, sort_keys=True)
    cache_key = hashlib.sha256((key_payload + "::" + _hash_texts(clean_texts)).encode("utf-8")).hexdigest()

    if use_cache:
        cached = _load_cached(cache_root, cache_key)
        if cached is not None:
            return cached

    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    if cfg.max_length is not None:
        # SentenceTransformer exposes max_seq_length on many models
        try:
            model.max_seq_length = int(cfg.max_length)
        except Exception:
            pass

    emb = model.encode(
        clean_texts,
        batch_size=int(cfg.batch_size),
        normalize_embeddings=bool(cfg.normalize),
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb = np.asarray(emb, dtype=np.float32)

    meta = {
        **params_fingerprint,
        "n_texts": len(clean_texts),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else None,
    }
    if use_cache:
        _save_cached(cache_root, cache_key, emb, meta)

    return emb


def embed_dataframe(
    df,
    cfg: EmbeddingConfig,
    *,
    cache_tag: Optional[str] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: embed df[cfg.text_col] in row order.

    Note:
      This returns embeddings only, keeping your existing pipelines/eval unchanged.
    """
    if cfg.text_col not in df.columns:
        raise ValueError(f"Text column '{cfg.text_col}' not found in df.columns={list(df.columns)}")

    texts = df[cfg.text_col].astype(str).fillna("").tolist()
    return embed_texts(texts, cfg, cache_tag=cache_tag, use_cache=use_cache)
