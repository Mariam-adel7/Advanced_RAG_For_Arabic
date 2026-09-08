from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from augmented.context_builder import ContextBuilder, SourceRef
from augmented.prompt_builder import PromptBuilder
from retrieval.base import BaseRetriever, RetrievalResult
from .llm_client import BaseLLMClient

@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[SourceRef]
    retrieved: List[RetrievalResult]

class QAEngine:
    def __init__(self, retriever: BaseRetriever, llm_client: BaseLLMClient, context_builder: Optional[ContextBuilder] = None, prompt_builder: Optional[PromptBuilder] = None, top_k: int = 5, max_tokens: int = 1024):
        self.retriever = retriever
        self.llm_client = llm_client
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> QAResult:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        context, sources = self.context_builder.build(retrieved)
        system, messages = self.prompt_builder.build_messages( question, context, sources, chat_history=chat_history,)
        answer_text = self.llm_client.generate(system, messages, max_tokens=self.max_tokens)
        return QAResult(question=question, answer=answer_text, sources=sources, retrieved=retrieved)
