from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Literal, Optional, Tuple
from ingestion.vector_store import metadata_matches
from .base import BaseRetriever, RetrievalResult
from .keyword_retriever import KeywordRetriever
from .semantic_retriever import SemanticRetriever

class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__( self, semantic: SemanticRetriever, keyword: KeywordRetriever, method: Literal["rrf", "weighted"] = "rrf", alpha: float = 0.5, rrf_k: int = 60, fetch_k: int = 40, ):
        self.semantic = semantic
        self.keyword = keyword
        self.method = method
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.rrf_k = max(1, int(rrf_k))
        self.fetch_k = max(1, int(fetch_k))

    def retrieve( self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None, ) -> List[RetrievalResult]:
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []
        fetch_k = max(self.fetch_k, top_k)
        semantic_hits = self.semantic.retrieve( query, top_k=fetch_k, )
        keyword_hits = self.keyword.retrieve( query, top_k=fetch_k, )

        if self.method == "rrf":
            fused = self._reciprocal_rank_fusion( semantic_hits, keyword_hits, )
        elif self.method == "weighted":
            fused = self._weighted_fusion( semantic_hits, keyword_hits, )
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        ranked = sorted( fused.items(), key=lambda item: item[1][0], reverse=True, )
        if metadata_filter:
            ranked = [ item for item in ranked if metadata_matches(item[1][1], metadata_filter) ]
        ranked = ranked[:top_k]
        results: List[RetrievalResult] = []

        for _, ( score, chunk, semantic_score, keyword_score, semantic_rank, keyword_rank, ) in ranked:
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            metadata["_hybrid_fusion_score"] = float(score)
            metadata["_hybrid_method"] = self.method
            metadata["_semantic_score"] = ( None if semantic_score is None else float(semantic_score) )
            metadata["_keyword_score"] = ( None if keyword_score is None else float(keyword_score) )
            metadata["_semantic_rank"] = semantic_rank
            metadata["_keyword_rank"] = keyword_rank
            metadata["_retrieved_by_semantic"] = semantic_rank is not None
            metadata["_retrieved_by_keyword"] = keyword_rank is not None

            result_chunk = replace(chunk, metadata=metadata)
            results.append(  RetrievalResult( chunk=result_chunk, score=float(score), retriever=self.name, ))
        return results

    def _reciprocal_rank_fusion( self, semantic_hits: List[RetrievalResult], keyword_hits: List[RetrievalResult], ) -> Dict[str, Tuple]:
        fused: Dict[str, Tuple] = {}

        def add_results( results: List[RetrievalResult], retriever_type: str, ) -> None:
            for rank_zero_based, result in enumerate(results):
                rank = rank_zero_based + 1
                chunk_id = result.chunk.chunk_id
                rrf_score = 1.0 / (self.rrf_k + rank)

                if chunk_id not in fused:
                    current_score = rrf_score
                    current_chunk = result.chunk
                    semantic_score = None
                    keyword_score = None
                    semantic_rank = None
                    keyword_rank = None
                else:
                    ( old_score, old_chunk, semantic_score, keyword_score, semantic_rank, keyword_rank, ) = fused[chunk_id]
                    current_score = old_score + rrf_score
                    current_chunk = old_chunk

                if retriever_type == "semantic":
                    semantic_score = float(result.score)
                    semantic_rank = rank
                else:
                    keyword_score = float(result.score)
                    keyword_rank = rank

                fused[chunk_id] = ( current_score, current_chunk, semantic_score, keyword_score, semantic_rank, keyword_rank, )

        add_results(semantic_hits, "semantic")
        add_results(keyword_hits, "keyword")
        return fused

    def _weighted_fusion( self, semantic_hits: List[RetrievalResult], keyword_hits: List[RetrievalResult], ) -> Dict[str, Tuple]:
        semantic_norm = self._min_max_normalize(semantic_hits)
        keyword_norm = self._min_max_normalize(keyword_hits)
        fused: Dict[str, Tuple] = {}
        for chunk_id, ( score, chunk, native_score, rank, ) in semantic_norm.items():
            fused[chunk_id] = ( self.alpha * score, chunk, native_score, None, rank, None, )
        for chunk_id, ( score, chunk, native_score, rank, ) in keyword_norm.items():
            if chunk_id in fused:
                ( previous_score, previous_chunk, semantic_score, _, semantic_rank, _, ) = fused[chunk_id]
                fused[chunk_id] = ( previous_score + (1.0 - self.alpha) * score, previous_chunk, semantic_score, native_score, semantic_rank, rank, )
            else:
                fused[chunk_id] = ( (1.0 - self.alpha) * score, chunk, None, native_score, None, rank,)
        return fused

    @staticmethod
    def _min_max_normalize( results: List[RetrievalResult], ) -> Dict[str, Tuple]:
        if not results:
            return {}
        scores = [float(result.score) for result in results]
        minimum = min(scores)
        maximum = max(scores)
        span = maximum - minimum
        output: Dict[str, Tuple] = {}
        for rank_zero_based, result in enumerate(results):
            normalized = ( 0.0 if span == 0 else (float(result.score) - minimum) / span )
            output[result.chunk.chunk_id] = ( normalized, result.chunk, float(result.score), rank_zero_based + 1, )
        return output