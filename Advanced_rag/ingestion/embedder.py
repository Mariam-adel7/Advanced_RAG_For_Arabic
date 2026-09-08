from __future__ import annotations
import hashlib
import logging
import re
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class BaseEmbedder:
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._is_m3 = "m3" in model_name.lower()

    def _prepare_documents(self, texts: List[str]) -> List[str]:
        if not self._is_m3:
            return texts
        return [f"passage: {text}" for text in texts]

    def _prepare_query(self, text: str) -> str:
        if not self._is_m3:
            return text
        return f"query: {text}"

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode( self._prepare_documents(texts), normalize_embeddings=True, show_progress_bar=False, )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode( [self._prepare_query(text)], normalize_embeddings=True, show_progress_bar=False, )[0]
        return np.asarray(vector, dtype=np.float32)

class HashingFallbackEmbedder(BaseEmbedder):
    _TOKEN_RE = re.compile( r"[a-z0-9]+"
                            r"|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+" )

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros( (len(texts), self.dim), dtype=np.float32, )
        for i, text in enumerate(texts):
            for token in self._TOKEN_RE.findall(text.lower()):
                h = int( hashlib.md5( token.encode("utf-8") ).hexdigest(), 16,)
                idx = h % self.dim
                sign = ( 1.0 if (h // self.dim) % 2 == 0 else -1.0 )
                vectors[i, idx] += sign
        norms = np.linalg.norm( vectors, axis=1, keepdims=True )
        norms[norms == 0] = 1.0
        return vectors / norms

class Embedder(BaseEmbedder):
    def __init__( self, model_name: str = "BAAI/bge-m3", backend: str | None = None, ):
        backend = backend or self._auto_detect()
        if backend == "sentence_transformers": 
            self._impl = SentenceTransformerEmbedder( model_name )
        elif backend == "hashing_fallback":
            logger.warning( "sentence-transformers not available — "
                            "using HashingFallbackEmbedder. "
                            "Install `sentence-transformers` and use "
                            "BAAI/bge-m3 for the real RAG evaluation." )
            self._impl = HashingFallbackEmbedder()
        else:
            raise ValueError( f"Unknown embedder backend: {backend}" )

        self.dim = self._impl.dim
        self.backend_name = backend
        self.model_name = model_name

    @staticmethod
    def _auto_detect() -> str:
        try:
            import sentence_transformers 
            return "sentence_transformers"
        except ImportError:
            return "hashing_fallback"

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._impl.embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._impl.embed_query(text)