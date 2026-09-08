from __future__ import annotations
from typing import Dict, List, Optional
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from .base import BaseRetriever, RetrievalResult

class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__( self, vector_store: VectorStore, embedder: Embedder, ):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve( self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None, ) -> List[RetrievalResult]:
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []

        query_vector = self.embedder.embed_query(query)
        hits = self.vector_store.search( query_vector, top_k=top_k, metadata_filter=metadata_filter, )
        return [ RetrievalResult( chunk=chunk, score=score, retriever=self.name, ) for chunk, score in hits ]