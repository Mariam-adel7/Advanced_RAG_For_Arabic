from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from ingestion.schema import Chunk

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    retriever: str

    def __repr__(self) -> str:
        preview = self.chunk.text[:60].replace("\n", " ")
        return ( f"RetrievalResult(" f"{self.retriever}, " f"score={self.score:.4f}, " f"text='{preview}...'" f")" )

class BaseRetriever(ABC):
    name: str = "base"
    @abstractmethod
    def retrieve( self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None, ) -> List[RetrievalResult]:
        ...