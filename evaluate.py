"""Embedding model wrapper. Used by rag_official.py to load KURE."""
from __future__ import annotations

import numpy as np


class DenseEmbedder:
    """Wraps a sentence-transformers model behind a common encode() interface."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 8,
                 model=None, query_prompt_name: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Instruction-tuned models need a prompt on queries. Documents are encoded as is,
        # and models without a prompt use None.
        self.query_prompt_name = query_prompt_name
        if model is not None:
            # Reuse an already-loaded model so the same weights are not loaded twice.
            print(f"Reusing already-loaded {model_name}", flush=True)
            self.model = model
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {model_name} ({device})", flush=True)
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        kwargs = dict(batch_size=self.batch_size,
                      normalize_embeddings=True, show_progress_bar=False)
        if is_query and self.query_prompt_name is not None:
            kwargs["prompt_name"] = self.query_prompt_name
        vecs = self.model.encode(texts, **kwargs)
        return np.asarray(vecs, dtype=np.float32)
