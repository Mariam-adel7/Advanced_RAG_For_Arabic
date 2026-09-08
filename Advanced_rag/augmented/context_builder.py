from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple
from retrieval.base import RetrievalResult

@dataclass
class SourceRef:
    index: int
    source: str
    chunk_id: str
    chunk_type: str
    score: float
    page: int | None = None
    section: str | None = None

class ContextBuilder:
    def __init__( self, max_context_chars: int = 10000, dedupe: bool = True, ):
        self.max_context_chars = max_context_chars
        self.dedupe = dedupe

    def build( self, results: List[RetrievalResult], ) -> Tuple[str, List[SourceRef]]:
        ordered = results
        if self.dedupe:
            seen = set()
            deduped = []
            for result in ordered:
                chunk_id = result.chunk.chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                deduped.append(result)
            ordered = deduped

        blocks: List[str] = []
        sources: List[SourceRef] = []
        total_chars = 0
        for i, result in enumerate(ordered, start=1):
            chunk = result.chunk
            filename = os.path.basename(chunk.source)
            chunk_type = ( "TABLE" if chunk.chunk_type == "table" else "TEXT" )
            page = chunk.metadata.get("page")
            section = chunk.metadata.get("section")
            header_parts = [ f"[Source {i}", f"TYPE={chunk_type}", f"FILE={filename}", ]

            if page is not None:
                header_parts.append(f"PAGE={page}")
            if section:
                header_parts.append(f"SECTION={section}")

            header_parts.append(f"SCORE={result.score:.4f}]")
            header = " | ".join(header_parts)

            if chunk.chunk_type == "table":
                body = ( "TABLE CONTENT:\n" f"{chunk.text}\n" "\n" "IMPORTANT: Preserve the relationship between " "table labels, rows, columns, values, and formulas." )
            else:
                body = ( "TEXT CONTENT:\n" f"{chunk.text}" )

            block = f"{header}\n{body}"
            block_size = len(block)
            if total_chars + block_size > self.max_context_chars:
                if not blocks:
                    remaining = self.max_context_chars
                    if remaining > len(header) + 20:
                        block = block[:remaining]
                        blocks.append(block)
                        sources.append( SourceRef( index=i, source=chunk.source, chunk_id=chunk.chunk_id, chunk_type=chunk.chunk_type, score=result.score, page=page, section=section, ))
                    break
                break

            blocks.append(block)
            total_chars += block_size
            sources.append( SourceRef( index=i, source=chunk.source, chunk_id=chunk.chunk_id, chunk_type=chunk.chunk_type, score=result.score, page=page, section=section, ))

        context = "\n\n---\n\n".join(blocks)
        return context, sources
    