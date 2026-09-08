from __future__ import annotations
import logging
from typing import List
from .chunkers import BaseChunker
from .embedder import Embedder
from .parsers import DocumentParser
from .schema import Chunk, Document
from .vector_store import VectorStore, create_vector_store

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, chunker: BaseChunker, embedder: Embedder | None = None, vector_store: VectorStore | None = None, parser: DocumentParser | None = None):
        self.parser = parser if parser is not None else DocumentParser()
        self.chunker = chunker
        self.embedder = embedder if embedder is not None else Embedder()
        self.vector_store = vector_store if vector_store is not None else create_vector_store(self.embedder.dim)

    def ingest_file(self, path: str) -> List[Chunk]:
        document = self.parser.parse(path)
        chunks = self._ingest_document(document)
        self._embed_and_store(chunks)
        return chunks

    def ingest_directory(self, dir_path: str, recursive: bool = True):
      documents = self.parser.parse_directory(dir_path, recursive=recursive)
      print(f"\n>>> Parsing complete: {len(documents)} documents")
      all_chunks = []
  
      for document in documents:
          print(f"\n>>> Processing document: {document.source}")
          chunks = self._ingest_document(document)
          print(f">>> Document produced {len(chunks)} chunks")
          all_chunks.extend(chunks)
  
      print(f"\n>>> TOTAL CHUNKS: {len(all_chunks)}")
      print(">>> Starting embedding...")
      self._embed_and_store(all_chunks)
      return all_chunks

    def _embed_and_store(self, chunks: List[Chunk]) -> None:
      if not chunks:
          print(">>> No chunks to embed.")
          return
      vectors = self.embedder.embed([chunk.text for chunk in chunks])
      self.vector_store.add(chunks, vectors)
      print( f">>> Embedded and stored {len(chunks)} chunks "
             f"(vector store now holds {len(self.vector_store)} chunks total)" )

    def _ingest_document(self, document):
     print(f"\n{'=' * 70}")
     print(f"INGESTING: {document.source}")
     print(f"ELEMENTS: {len(document.elements)}")
     print(f"{'=' * 70}")
     print(">>> Starting chunking...")
     chunks = self.chunker.chunk(document)
     print(f">>> Chunking finished: {len(chunks)} chunks")
     return chunks