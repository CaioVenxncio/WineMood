"""Frozen pretrained sentence encoder (protocol N2, amended 2026-10-12).

Mean-pooled, L2-normalised token embeddings from a pretrained encoder; the
encoder is never fine-tuned. Weights are loaded from ``safetensors`` only (no
pickle). Requires the optional extra: ``pip install -e ".[transformer]"``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EncoderUnavailableError(RuntimeError):
    """The optional [transformer] dependencies are not installed."""


def _imports():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EncoderUnavailableError(
            "The frozen-encoder experiment needs the optional extra: "
            'pip install -e ".[transformer]"'
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _resolve_revision(model_name: str) -> str | None:
    """Pin the exact commit of the model repository so runs are reproducible."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_name).sha
    except Exception:  # offline or hub unavailable: fall back to the cached default
        return None


class FrozenEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = None,
        max_length: int = 128,
        threads: int | None = None,
    ) -> None:
        torch, auto_model, auto_tokenizer = _imports()
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        revision = revision or _resolve_revision(model_name)
        self.tokenizer = auto_tokenizer.from_pretrained(model_name, revision=revision)
        self.model = auto_model.from_pretrained(
            model_name, revision=revision, use_safetensors=True
        ).eval()
        self.revision = revision

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "revision": self.revision,
            "max_length": self.max_length,
            "pooling": "mean",
            "normalised": True,
            "fine_tuned": False,
        }

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def cached_embeddings(
    encoder: FrozenEncoder, texts: list[str], ids: np.ndarray, cache_dir: Path
) -> np.ndarray:
    """Embed ``texts`` once per (model, revision, ids, texts) and reuse from disk."""
    key = hashlib.sha256()
    key.update(f"{encoder.model_name}@{encoder.revision}@{encoder.max_length}".encode())
    key.update(np.asarray(ids, dtype=np.int64).tobytes())
    key.update("\n".join(texts).encode("utf-8"))
    path = cache_dir / f"{key.hexdigest()[:24]}.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False)
    emb = encoder.encode(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb, allow_pickle=False)
    return emb
