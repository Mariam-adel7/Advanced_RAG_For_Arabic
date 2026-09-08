from __future__ import annotations
from augmented import ContextBuilder, PromptBuilder
from config import Config
from generation import (QAEngine, create_llm_client,)
from advanced_rag.engine import AdvancedRAGEngine
from ingestion import (
    Embedder,
    FixedSizeChunker,
    IngestionPipeline,
    RecursiveCharacterChunker,
    StructuralChunker,
    ManualProcedureChunker,
    ParentChildChunker,
    VectorStore,
    create_vector_store,)
from retrieval import (
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever, )

_CHUNKERS = {
    "fixed_size": FixedSizeChunker,
    "recursive_character": RecursiveCharacterChunker,
    "structural": StructuralChunker,
    "manual_procedure": ManualProcedureChunker,
    "parent_child": ParentChildChunker,}

def build_chunker(config: Config):
    cls = _CHUNKERS.get( config.chunking_strategy )
    if cls is None:
        raise ValueError( f"Unknown chunking_strategy " f"'{config.chunking_strategy}', " f"choose from {list(_CHUNKERS)}" )
    if cls is StructuralChunker:
        return cls( max_chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, )
    if cls is ManualProcedureChunker:
        return cls( target_size=config.chunk_size, overlap=config.chunk_overlap, )
    if cls is ParentChildChunker:
        return cls(parent_size = config.parent_size, child_size = config.child_size, child_overlap = config.child_overlap,)
    
    return cls( chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, )


def build_ingestion_pipeline( config: Config ) -> IngestionPipeline:
    chunker = build_chunker( config )
    embedder = Embedder( model_name=config.embedding_model )
    vector_store = create_vector_store( embedder.dim, backend=config.vector_store_backend, )

    return IngestionPipeline( chunker=chunker, embedder=embedder, vector_store=vector_store, )


def build_retriever( config: Config, vector_store: VectorStore, embedder: Embedder, ):
    semantic = SemanticRetriever( vector_store, embedder, )
    keyword = KeywordRetriever( vector_store.chunks )
    if config.retrieval_strategy == "semantic":
        return semantic
    if config.retrieval_strategy == "keyword":
        return keyword
    if config.retrieval_strategy == "hybrid":
        return HybridRetriever( semantic, keyword, method=config.hybrid_fusion_method, alpha=config.hybrid_alpha, fetch_k=40, )

    raise ValueError( f"Unknown retrieval_strategy " f"'{config.retrieval_strategy}', " f"choose from " f"semantic|keyword|hybrid" )


def build_qa_engine( config: Config, vector_store: VectorStore, embedder: Embedder, ) -> QAEngine:
    retriever = build_retriever( config, vector_store, embedder, )
    llm_client = create_llm_client( config.llm_backend, model=config.llm_model, api_key=config.gemini_api_key, )
    context_builder = ContextBuilder( max_context_chars=config.max_context_chars)
    prompt_builder = PromptBuilder()

    return QAEngine( retriever=retriever, llm_client=llm_client, context_builder=context_builder, prompt_builder=prompt_builder, top_k=config.top_k, max_tokens=config.max_answer_tokens, )


def build_advanced_rag_engine( config: Config, vector_store: VectorStore, embedder: Embedder, ) -> AdvancedRAGEngine:
    retriever = build_retriever( config, vector_store, embedder, )
    llm_client = create_llm_client( config.llm_backend, model=config.llm_model, api_key=config.gemini_api_key, )
    context_builder = ContextBuilder( max_context_chars=config.max_context_chars )
    prompt_builder = PromptBuilder()

    return AdvancedRAGEngine( config=config, retriever=retriever, llm_client=llm_client, context_builder=context_builder, prompt_builder=prompt_builder, top_k=config.top_k, max_tokens=config.max_answer_tokens, )
