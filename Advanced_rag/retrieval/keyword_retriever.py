from __future__ import annotations
import re
from typing import Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from ingestion.schema import Chunk
from .base import BaseRetriever, RetrievalResult

_TOKEN_RE = re.compile( r"[a-z0-9]+"
    r"|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

_STOPWORDS_EN = frozenset({ "a", "an", "the", "and", "or", "but", "if", "then",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by",
    "as", "it", "this", "that", "these", "those", "from", })

_STOPWORDS_AR = frozenset({"من", "الى", "إلى", "على", "عن", "في", "و", "أو", "او",
    "ثم", "أن", "ان", "إن", "ما", "لا", "لم", "لن", "قد",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "التي",
    "الذي", "الذين", "بين", "مع", "كل", "بعد", "قبل", "حيث",
    "كما", "أي", "اي", "له", "لها", "لهم", "به", "بها",
    "منه", "منها",})

_ARABIC_DIACRITICS_RE = re.compile( r"[\u064B-\u065F\u0670\u06D6-\u06ED]" )

def _normalize_arabic(token: str) -> str:
    token = _ARABIC_DIACRITICS_RE.sub("", token)
    token = re.sub(r"[إأآا]", "ا", token)
    token = token.replace("ى", "ي")
    token = token.replace("ة", "ه")

    return token

_STOPWORDS_AR_NORMALIZED = frozenset(_normalize_arabic(w) for w in _STOPWORDS_AR)

def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        token = token.strip("؟،؛ـ")
        if not token:
            continue
        if re.match( r"^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+$", token, ):
            token = _normalize_arabic(token)
            if token and token not in _STOPWORDS_AR_NORMALIZED:
                tokens.append(token)
        else:
            if token not in _STOPWORDS_EN:
                tokens.append(token)
    return tokens

class KeywordRetriever(BaseRetriever):
    name = "keyword"

    def __init__(self, chunks: List[Chunk]):
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self.refresh(chunks)

    def refresh(self, chunks: List[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self._bm25 = None
            return
        tokenized_corpus = [ tokenize(chunk.text) for chunk in self.chunks ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _metadata_matches( chunk: Chunk, metadata_filter: Optional[Dict] = None, ) -> bool:
        if not metadata_filter:
            return True

        metadata = getattr(chunk, "metadata", {}) or {}
        for key, expected in metadata_filter.items():
            if expected is None:
                continue
            actual = metadata.get(key)

            if key not in { "issue_year", "review_year", "year_gt", }:
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                else:
                    if actual != expected:
                        return False
                continue

            if key == "issue_year":
                if actual is None:
                    return False
                try:
                    if int(actual) != int(expected):
                        return False
                except (TypeError, ValueError):
                    return False

            elif key == "review_year":
                if actual is None:
                    return False
                try:
                    if int(actual) != int(expected):
                        return False
                except (TypeError, ValueError):
                    return False

            elif key == "year_gt":
                years = []
                for field in ("issue_year", "review_year"):
                    value = metadata.get(field)
                    if value is not None:
                        try:
                            years.append(int(value))
                        except (TypeError, ValueError):
                            pass
                if not years or max(years) <= int(expected):
                    return False
        return True

    def retrieve( self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None, ) -> List[RetrievalResult]:
        if self._bm25 is None or not self.chunks:
            return []
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray( self._bm25.get_scores(query_tokens), dtype=np.float32, )
        candidate_indices = [ i for i, chunk in enumerate(self.chunks) if self._metadata_matches(chunk,metadata_filter, ) ]
        if not candidate_indices:
            return []
        doc_freqs = getattr(self._bm25, "doc_freqs", None)

        def has_overlap(i: int) -> bool:
            if doc_freqs is None:
                return True 
            return any(t in doc_freqs[i] for t in query_tokens)
        candidate_scores = [ (i, float(scores[i])) for i in candidate_indices if has_overlap(i) ]
        if not candidate_scores:
            return []
        candidate_scores.sort( key=lambda x: x[1], reverse=True,)
        candidate_scores = candidate_scores[:top_k]
        return [ RetrievalResult(chunk=self.chunks[i],score=score,retriever=self.name, ) for i, score in candidate_scores ]