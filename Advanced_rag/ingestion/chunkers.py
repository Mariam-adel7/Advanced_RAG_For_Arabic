from __future__ import annotations
import re
from abc import ABC, abstractmethod
from .schema import Chunk, Document, Element
from typing import Any, Dict, List, Tuple

Segment = Tuple[str, str, Dict[str, Any]] 

def segment_document(document: Document) -> List[Segment]:
    segments: List[Segment] = []
    buffer: List[str] = []
    buffer_meta: Dict[str, Any] = {}
    current_page = None

    def flush():
        nonlocal current_page
        if buffer:
            meta = dict(buffer_meta)
            if current_page is not None:
                meta["page"] = current_page
                meta.setdefault("pages", [current_page])
            segments.append(("text", "\n\n".join(buffer), meta))
            buffer.clear()
            buffer_meta.clear()
            current_page = None

    for el in document.elements:
        if el.type == "table":
            flush()
            meta = dict(el.metadata or {})
            meta["page"] = el.page
            if el.page is not None:
                meta["pages"] = [el.page]
            segments.append(("table", el.render(), meta))
            continue
        page = el.page

        if buffer and page is not None and current_page is not None and page != current_page:
            flush()

        buffer.append(el.render())
        if page is not None:
            if current_page is None:
                current_page = page
            buffer_meta.update({k: v for k, v in (el.metadata or {}).items() if k not in buffer_meta})
            pages = list(buffer_meta.get("pages", []))
            if page not in pages:
                pages.append(page)
            buffer_meta["pages"] = pages

    flush()
    return segments

def split_table_by_rows(markdown_table_rows: List[List[str]], max_chars: int, caption: str | None = None) -> List[str]:
    from .schema import Table
    if not markdown_table_rows:
        return []
    header, *body = markdown_table_rows
    pieces: List[str] = []
    current_rows: List[List[str]] = []

    def render(rows: List[List[str]]) -> str:
        return Table(rows=[header] + rows, caption=caption).to_markdown()

    for row in body:
        trial = render(current_rows + [row])
        if len(trial) > max_chars and current_rows:
            pieces.append(render(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        pieces.append(render(current_rows))
    return pieces or [render([])]

class BaseChunker(ABC):
    name: str = "base"
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        ...
    def _table_chunks( self, document: Document, table_markdown_rows, caption, page, position, ) -> List[Chunk]:
        raise NotImplementedError

class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.full_text
        if not text:
            return []
        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        position = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="text", strategy=self.name, position=position, char_start=start, char_end=end, ))
                position += 1
            if end == n:
                break
            start += step
        return chunks

class RecursiveCharacterChunker(BaseChunker):
    name = "recursive_character"
    DEFAULT_SEPARATORS = [ "\n\n", "\n", "۔ ", ". ", "؟ ", "? ", "، ", ", ", " ", "", ]

    def __init__( self, chunk_size: int = 1000, chunk_overlap: int = 150, separators: List[str] | None = None, ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> List[Chunk]:
        segments = segment_document(document)
        chunks: List[Chunk] = []
        position = 0
        for seg_type, seg_text, seg_meta in segments:
            if seg_type == "table":
                for piece in self._table_pieces(seg_text):
                    chunks.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="table", strategy=self.name, position=position, **seg_meta, ))
                    position += 1
            else:
                for piece in self._split_text( seg_text, self.separators ):
                    piece = piece.strip()
                    if piece:
                        chunks.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="text", strategy=self.name, position=position, **seg_meta, ))
                        position += 1
        return chunks

    def _table_pieces(self, table_markdown: str) -> List[str]:
        if len(table_markdown) <= self.chunk_size:
            return [table_markdown]

        rows = [ line for line in table_markdown.split("\n") if line.strip().startswith("|") ]
        parsed_rows = [ [c.strip() for c in r.strip("|").split("|")] for r in rows if "---" not in r ]
        return split_table_by_rows( parsed_rows, self.chunk_size )

    def _split_text(self, text: str, separators: List[str], ) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator, remaining = separators[0], separators[1:]
        if separator == "":
            splits = [ text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size) ]
        else:
            splits = text.split(separator)
            splits = ( [s + separator for s in splits[:-1]] + splits[-1:] )

        atomic: List[str] = []

        for s in splits:
            if len(s) > self.chunk_size and remaining:
                atomic.extend( self._split_text(s, remaining) )
            elif len(s) > self.chunk_size:
                atomic.extend([ s[i:i + self.chunk_size] for i in range(0, len(s), self.chunk_size) ] )
            elif s.strip():
                atomic.append(s)

        return self._merge_with_overlap(atomic)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        merged: List[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                merged.append(current)
                current = current[-self.chunk_overlap:] + piece
            else:
                current += piece

        if current.strip():
            merged.append(current)
        return merged

class StructuralChunker(BaseChunker):
    name = "structural"

    def __init__( self, max_chunk_size: int = 1500, chunk_overlap: int = 100, ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback = RecursiveCharacterChunker( chunk_size=max_chunk_size, chunk_overlap=chunk_overlap, )

    def chunk(self, document: Document) -> List[Chunk]:
        sections = self._group_into_sections(document)
        chunks: List[Chunk] = []
        position = 0
        for heading_path, elements in sections:
            prefix = " > ".join(heading_path) if heading_path else None
            page_groups: List[tuple] = []
            current_page = None
            current_elements = []

            for element in elements:
                element_page = getattr(element, "page", None)
                if current_elements and element_page is not None and current_page is not None and element_page != current_page:
                    page_groups.append((current_page, current_elements))
                    current_elements = []
                    current_page = element_page
                current_elements.append(element)
                if element_page is not None and current_page is None:
                    current_page = element_page

            if current_elements:
                page_groups.append((current_page, current_elements))

            for page, page_elements in page_groups:
                table_elements = [e for e in page_elements if e.type == "table"]
                text_elements = [e for e in page_elements if e.type != "table"]
                section_text = "\n\n".join(e.render() for e in text_elements).strip()
                if section_text:
                    body = f"{prefix}\n\n{section_text}" if prefix else section_text
                    if len(body) <= self.max_chunk_size:
                        pieces = [body]
                    else:
                        pieces = self._fallback._split_text(body, self._fallback.separators)

                    for piece in pieces:
                        piece = piece.strip()
                        if piece:
                            chunks.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="text", strategy=self.name, position=position, section=prefix, page=page, metadata={"pages": [page]} if page is not None else {},) )
                            position += 1

                for table_el in table_elements:
                    table_md = table_el.render()
                    caption = prefix
                    if len(table_md) <= self.max_chunk_size:
                        pieces = [table_md]
                    else:
                        rows = table_el.table.rows if table_el.table else []
                        pieces = split_table_by_rows(rows, self.max_chunk_size, caption=caption)

                    for piece in pieces:
                        piece = piece.strip()
                        if piece:
                            table_page = getattr(table_el, "page", None)
                            chunks.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="table", strategy=self.name, position=position, section=prefix, page=table_page, metadata={"pages": [table_page]} if table_page is not None else {},))
                            position += 1
        return chunks

    @staticmethod
    def _group_into_sections(document: Document):
        sections = []
        heading_stack: List[tuple] = []
        current_elements = []

        def path():
            return [t for _, t in heading_stack]

        for el in document.elements:
            if el.type == "heading":
                if current_elements or heading_stack: 
                    sections.append( (path(), current_elements) )

                current_elements = []
                while ( heading_stack and heading_stack[-1][0] >= (el.level or 1) ):
                    heading_stack.pop()
                heading_stack.append( (el.level or 1, el.text))
            else:
                current_elements.append(el)

        if current_elements or not sections:
            sections.append( (path(), current_elements) )

        return [ s for s in sections if s[1] ]

class ManualProcedureChunker(BaseChunker):
    name = "manual_procedure"

    PROCEDURE_RE = re.compile(
        r"^\s*(?:"
        r"[.．]?\s*(?:\d+(?:\.\d+){0,3})\s*[-–—:.)]?\s*"
        r"(?:إجراءات|اجراءات|إجراء|اجراء|procedure|procedures|process|task)"
        r"(?:\s+.*)?"
        r"|"
        r"(?:procedure|process|task)\s*[:#\-]?\s*\d+(?:\.\d+)*.*"
        r"|"
        r"(?:الإجراء|الإجراءات|العملية|المهمة)\s*[:#\-]?\s*.*"
        r")$", re.IGNORECASE, )

    STEP_RE = re.compile(
        r"^\s*(?:"
        r"\d+[.)]"
        r"|[-•]"
        r"|step\s+\d+"
        r"|الخطوة\s+\d+"
        r")\s+", re.IGNORECASE, )

    def __init__(self, target_size: int = 1200, overlap: int = 150):
        if overlap >= target_size:
            raise ValueError("overlap must be smaller than target_size")
        self.target_size = target_size
        self.overlap = overlap
        self._fallback = RecursiveCharacterChunker( chunk_size=target_size, chunk_overlap=overlap, )

    @classmethod
    def _is_procedure_start(cls, text: str) -> bool:
        first_line = (text or "").strip().splitlines()[0] if text else ""
        return bool(cls.PROCEDURE_RE.match(first_line))

    @classmethod
    def _is_step(cls, text: str) -> bool:
        return bool(cls.STEP_RE.match((text or "").strip()))

    @staticmethod
    def _infer_table_title(table_text: str) -> str | None:
        t = table_text or ""
        if "المخزون الراكد" in t or "المواد الراكدة" in t:
            return "إجراءات المخزون الراكد من اللوازم والقرطاسية"
        if "اسم النموذج" in t or "مذكرة اعتماد" in t:
            return "النماذج"
        if "آلية استخراج التقرير" in t or "الغاية من التقرير" in t:
            return "التقارير"
        if "أسباب التعديل" in t or "الجهة طالبة التعديل" in t:
            return "جدول التعديلات"
        if "ملحق" in t or "القيود المحاسبية" in t:
            return "الملاحق"
        return None

    @staticmethod
    def _with_title(title: str | None, piece: str) -> str:
        piece = piece.strip()
        if not title:
            return piece
        title = str(title).strip()
        if not title or piece.startswith(title):
            return piece
        return f"{title}\n\n{piece}"

    def chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        position = 0
        current: List = []
        current_title: str | None = None
        current_pages: List[int] = []

        def add_page(page):
            if page is not None and page not in current_pages:
                current_pages.append(page)

        def flush(reset_title: bool = False):
            nonlocal position, current, current_title, current_pages
            if not current:
                return

            rendered = "\n\n".join( e.render() for e in current if getattr(e, "render", None) ).strip()
            if not rendered:
                current = []
                current_pages = []
                if reset_title:
                    current_title = None
                return

            if len(rendered) <= self.target_size:
                pieces = [rendered]
            else:
                pieces = self._fallback._split_text( rendered, self._fallback.separators )

            first_page = current_pages[0] if current_pages else None
            for piece in pieces:
                piece = self._with_title(current_title, piece)
                if not piece:
                    continue
                chunks.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="text", strategy=self.name, position=position, section=current_title, page=first_page, metadata={"pages": list(current_pages),"procedure_title": current_title,"procedure_chunk": True, },))
                position += 1

            current = []
            current_pages = []
            if reset_title:
                current_title = None

        for el in document.elements:
            element_page = getattr(el, "page", None)
            text_value = (getattr(el, "text", None) or "").strip()

            if el.type == "table":
                table_text = el.render().strip()
                table_title = current_title or self._infer_table_title(table_text)

                if current and table_title and current_title is None:
                    current_title = table_title
                add_page(element_page)

                if len(table_text) <= self.target_size:
                    pieces = [table_text]
                else:
                    rows = el.table.rows if el.table else []
                    pieces = split_table_by_rows( rows, self.target_size, caption=table_title )

                flush(reset_title=False)
                for piece in pieces:
                    piece = self._with_title(table_title, piece)
                    if not piece:
                        continue
                    chunks.append(
                        Chunk.create( doc_id=document.doc_id, source=document.source, text=piece, chunk_type="table", strategy=self.name, position=position, section=table_title, page=element_page, metadata={ "pages": [element_page] if element_page is not None else [], "procedure_title": table_title, "table_chunk": True, }, ))
                    position += 1
                continue

            if self._is_procedure_start(text_value):
                if current:
                    flush(reset_title=True)
                current = [el]
                current_title = text_value
                current_pages = []
                add_page(element_page)
                continue
            current.append(el)
            add_page(element_page)

        flush(reset_title=True)
        return chunks

class ParentChildChunker(BaseChunker):
    name = "parent_child"

    def __init__( self, parent_size: int = 2400, child_size: int = 700, child_overlap: int = 100, ):
        if parent_size <= 0 or child_size <= 0:
            raise ValueError("parent_size and child_size must be positive")
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        if child_size > parent_size:
            raise ValueError("child_size must not exceed parent_size")

        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self._parent_builder = ManualProcedureChunker( target_size=parent_size, overlap=min(150, max(1, parent_size // 10)), )
        self._child_splitter = RecursiveCharacterChunker( chunk_size=child_size, chunk_overlap=child_overlap, )

    def chunk(self, document: Document) -> List[Chunk]:
        parent_chunks = self._parent_builder.chunk(document)
        children: List[Chunk] = []
        position = 0

        for parent_index, parent in enumerate(parent_chunks):
            parent_text = (parent.text or "").strip()
            if not parent_text:
                continue

            parent_id = ( f"{document.doc_id}:parent:{parent_index}")
            if len(parent_text) <= self.child_size:
                child_texts = [parent_text]
            else:
                child_texts = self._child_splitter._split_text( parent_text, self._child_splitter.separators,)

            child_texts = [x.strip() for x in child_texts if x and x.strip()]
            child_count = len(child_texts)

            for child_index, child_text in enumerate(child_texts):
                parent_title = getattr(parent, "section", None)
                retrieval_text = child_text

                if parent_title:
                    title = str(parent_title).strip()
                    if title and not child_text.startswith(title):
                        retrieval_text = f"{title}\n\n{child_text}"

                children.append( Chunk.create( doc_id=document.doc_id, source=document.source, text=retrieval_text, chunk_type="text", strategy=self.name, position=position, section=getattr(parent, "section", None), page=getattr(parent, "page", None), parent_id=parent_id, parent_position=parent_index, parent_text=parent_text, parent_section=getattr(parent, "section", None), parent_page=getattr(parent, "page", None), child_index=child_index, child_count=child_count, is_parent=False, is_child=True,) )
                position += 1

        return children

