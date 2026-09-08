from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

@dataclass
class Table:
    rows: List[List[str]]              
    caption: Optional[str] = None
    page: Optional[int] = None

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        header, *body = self.rows
        header = [c.strip() if c else "" for c in header]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in body:
            row = list(row) + [""] * (len(header) - len(row))  
            row = [c.strip() if c else "" for c in row[: len(header)]]
            lines.append("| " + " | ".join(row) + " |")
        md = "\n".join(lines)
        if self.caption:
            md = f"**Table: {self.caption}**\n\n{md}"
        return md

    def __len__(self) -> int:
        return len(self.rows)

@dataclass
class Element:
    type: str
    text: str
    level: Optional[int] = None        
    page: Optional[int] = None
    table: Optional[Table] = None       
    metadata: Dict[str, Any] = field(default_factory=dict)
    def render(self) -> str:
        if self.type == "table" and self.table is not None:
            return self.table.to_markdown()
        if self.type == "heading":
            return f"{'#' * (self.level or 1)} {self.text}"
        return self.text

@dataclass
class Document:
    doc_id: str
    source: str                     
    elements: List[Element]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(e.render() for e in self.elements)

    @classmethod
    def create(cls, source: str, elements: List[Element], **metadata) -> "Document":
        return cls(doc_id=new_id("doc"), source=source, elements=elements, metadata=metadata)

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_type: str = "text"
    strategy: str = "unknown"
    position: int = 0
    section: str | None = None
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_id: str | None = None
    parent_position: int | None = None
    parent_text: str | None = None
    parent_section: str | None = None
    parent_page: int | None = None
    child_index: int | None = None
    child_count: int | None = None
    is_parent: bool = False
    is_child: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        doc_id: str,
        source: str,
        text: str,
        chunk_type: str = "text",
        strategy: str = "unknown",
        position: int = 0,
        section: str | None = None,
        page: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        parent_id: str | None = None,
        parent_position: int | None = None,
        parent_text: str | None = None,
        parent_section: str | None = None,
        parent_page: int | None = None,
        child_index: int | None = None,
        child_count: int | None = None,
        is_parent: bool = False,
        is_child: bool = False,
        metadata: Dict[str, Any] | None = None, ) -> "Chunk":
        
        return cls(
            chunk_id=new_id("chk"),
            doc_id=doc_id,
            source=source,
            text=text,
            chunk_type=chunk_type,
            strategy=strategy,
            position=position,
            section=section,
            page=page,
            char_start=char_start,
            char_end=char_end,
            parent_id=parent_id,
            parent_position=parent_position,
            parent_text=parent_text,
            parent_section=parent_section,
            parent_page=parent_page,
            child_index=child_index,
            child_count=child_count,
            is_parent=is_parent,
            is_child=is_child,
            metadata=metadata or {}, )

    