from __future__ import annotations
import math
import pickle
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from .schema import Chunk

def _normalize_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

def _extract_years(value) -> List[int]:
    if value is None:
        return []
    return [ int(y) for y in re.findall( r"\b(19\d{2}|20\d{2})\b", str(value), )]

def metadata_matches( chunk: Chunk, metadata_filter: Dict | None, ) -> bool:
    if not metadata_filter:
        return True
    
    metadata = getattr( chunk, "metadata", {} ) or {}
    metadata = { str(key).lower(): value for key, value in metadata.items() }
    text_fields = [ "document", "procedure", "section", "version", "department", ]

    for field in text_fields:
        expected = metadata_filter.get(field)
        if expected is None:
            continue
        actual = metadata.get(field)
        if actual is None:
            return False
        if _normalize_value(expected) not in _normalize_value(actual):
            return False

    for field in [ "issue_date", "review_date", ]:
        expected = metadata_filter.get(field)
        if expected is None:
            continue
        actual = metadata.get(field)
        if actual is None:
            return False
        if _normalize_value(expected) not in _normalize_value(actual):
            return False


    for field in [ "issue_year", "review_year", ]:
        expected = metadata_filter.get(field)
        if expected is None:
            continue
        actual = metadata.get(field)
        if actual is None:
            date_field = field.replace( "_year", "_date", )
            actual = metadata.get( date_field )
        years = _extract_years(actual)
        if not years:
            return False
        if int(expected) not in years:
            return False


    if metadata_filter.get("year_gt") is not None:
        threshold = int( metadata_filter["year_gt"] )
        candidate_years = []
        for field in [ "year", "publication_year", "issue_year", "review_year", "issue_date", "review_date",]:
            candidate_years.extend( _extract_years( metadata.get(field) ))
        if not candidate_years:
            return False
        if not any( year > threshold for year in candidate_years ):
            return False

    return True

class VectorStore:
    def __init__( self, dim: int, ):
        self.dim = dim
        self.chunks: List[Chunk] = []
        self._vectors: List[np.ndarray] = []

    def add( self, chunks: List[Chunk], vectors: np.ndarray, ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError( "chunks/vectors length mismatch" )

        for chunk, vector in zip( chunks, vectors, ):
            vector = np.asarray( vector, dtype=np.float32, )
            if ( vector.ndim != 1 or len(vector) != self.dim ):
                raise ValueError( f"Expected vector dimension " f"{self.dim}, got shape " f"{vector.shape}" )

            self.chunks.append(chunk)
            self._vectors.append(vector)
        self._on_add()

    def _on_add(self) -> None:
        pass

    def search( self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: Dict | None = None, ) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.chunks)

    def save( self, path: str, ) -> None:
        with open( path, "wb", ) as f:
            pickle.dump( self, f, )

    @staticmethod
    def load( path: str, ) -> "VectorStore":
        with open( path, "rb", ) as f:
            return pickle.load(f)

class NumpyVectorStore(VectorStore):
    def __init__( self, dim: int, ):
        super().__init__(dim)
        self._matrix: np.ndarray | None = None

    def _on_add(self) -> None:
        if not self._vectors:
            self._matrix = None
            return
        matrix = np.stack( self._vectors )
        norms = np.linalg.norm( matrix, axis=1, keepdims=True, )
        norms[norms == 0] = 1.0
        self._matrix = ( matrix / norms )

    def search( self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: Dict | None = None, ) -> List[Tuple[Chunk, float]]:
        if ( self._matrix is None or not self.chunks ):
            return []

        q = np.asarray( query_vector, dtype=np.float32, )
        q = q / ( np.linalg.norm(q) or 1.0 )
        scores = self._matrix @ q
        candidate_indices = [ i for i, chunk in enumerate( self.chunks ) if metadata_matches( chunk, metadata_filter, ) ]
        if not candidate_indices:
            return []

        candidate_scores = scores[ candidate_indices ]
        k = min( top_k, len(candidate_indices), )
        if k <= 0:
            return []
        if k == len(candidate_indices):
            ordered = np.argsort( -candidate_scores )
        else:
            local_idx = np.argpartition( -candidate_scores, k - 1, )[:k]
            ordered = local_idx[ np.argsort( -candidate_scores[local_idx] ) ]

        return [( self.chunks[ candidate_indices[i] ], float( candidate_scores[i]), ) for i in ordered ]

class FaissVectorStore(VectorStore):
    def __init__( self, dim: int, ):
        import faiss
        super().__init__(dim)
        self._index = ( faiss.IndexFlatIP(dim) )

    def _on_add(self) -> None:
        import faiss
        if not self._vectors:
            self._index = (faiss.IndexFlatIP(self.dim))
            return
        matrix = np.stack( self._vectors ).astype( np.float32 )
        norms = np.linalg.norm( matrix, axis=1, keepdims=True, )
        norms[norms == 0] = 1.0
        normalized = ( matrix / norms )
        self._index = (faiss.IndexFlatIP(self.dim))
        self._index.add( normalized )

    def search( self, query_vector: np.ndarray, top_k: int = 5, metadata_filter: Dict | None = None, ) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = np.asarray( query_vector, dtype=np.float32, ).reshape( 1, -1, )
        q = q / ( np.linalg.norm(q) or 1.0 )

        if not metadata_filter:
            top_k = min( top_k, len(self.chunks), )
            scores, idx = ( self._index.search( q, top_k, ) )
            return [ ( self.chunks[i], float(s), ) for s, i in zip( scores[0], idx[0] ) if i != -1 ]

        candidate_indices = [ i for i, chunk in enumerate( self.chunks ) if metadata_matches( chunk,  metadata_filter, ) ] 
        if not candidate_indices:
            return []

        matrix = np.stack([ self._vectors[i] for i in candidate_indices ] ).astype( np.float32 )
        norms = np.linalg.norm( matrix, axis=1, keepdims=True, )
        norms[norms == 0] = 1.0
        matrix = ( matrix / norms )
        scores = ( matrix @ q[0] )
        k = min( top_k, len(candidate_indices), )
        if k <= 0:
            return []
        if k == len(candidate_indices):
            ordered = np.argsort( -scores )
        else:
            local_idx = np.argpartition( -scores, k - 1, )[:k]
            ordered = local_idx[ np.argsort( -scores[ local_idx ] ) ]
        return [( self.chunks[ candidate_indices[i] ], float( scores[i] ),) for i in ordered ]

    def __getstate__(self) -> dict:
        import faiss
        state = self.__dict__.copy()
        state["_index"] = ( faiss.serialize_index( self._index ))
        return state

    def __setstate__( self, state: dict, ) -> None:
        import faiss
        index_bytes = state.pop( "_index" )
        self.__dict__.update( state )
        self._index = ( faiss.deserialize_index( index_bytes ) )

def create_vector_store( dim: int, backend: str = "auto", ) -> VectorStore:
    if backend == "numpy":
        return NumpyVectorStore( dim )
    if backend == "faiss":
        return FaissVectorStore( dim )
    try:
        return FaissVectorStore( dim )
    except ImportError:
        return NumpyVectorStore( dim )