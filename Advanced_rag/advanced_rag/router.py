from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RouteDecision:
    route: str
    reason: str
    techniques: List[str]

class Router:
    SYSTEM = """You are the routing component of an Advanced RAG system.
Classify the user's question into exactly one route:
- simple:
  Use ONLY when the question is clearly general knowledge and does NOT
  require information from the supplied documents.
- basic_rag:
  Use when the question asks for a specific fact, definition, formula,
  procedure, rule, value, instruction, or explanation that may be contained
  in the supplied documents.
- advanced_rag:
  Use when there are multiple information needs, ambiguity, metadata
  constraints, weak retrieval risk, or a query transformation/retrieval
  improvement is useful.

IMPORTANT:
If the question asks for a formula, calculation method, procedure, rule,
policy, instruction, or organization-specific fact, prefer basic_rag because
the answer may be defined differently in the supplied documents.
Do NOT classify a document-specific formula or procedure as simple merely
because a similar standard formula is known from general knowledge.
Choose only techniques that are actually justified:
rewriting, multi_query, decomposition, hyde, self_query, reranking,
compression, crag.
Is the question simple or complex?
Does it require information from the provided documents?
Does it contain multiple information needs?
Does it require multiple search perspectives?
Does it contain metadata constraints?
Is the wording vague or conversational?

Return JSON only:
{"route":"simple|basic_rag|advanced_rag",
 "reason":"...",
 "techniques":["..."]}.
Do not force advanced techniques for a simple question.
"""

    DOCUMENT_SENSITIVE_TERMS = [
        "formula",
        "calculate",
        "calculation",
        "equation",
        "procedure",
        "procedures",
        "process",
        "policy",
        "policies",
        "rule",
        "rules",
        "instruction",
        "instructions",
        "manual",
        "according to",
        "according to the manual",
        "according to the document",
        "how is",
        "how are",
        "what is the procedure",
        "what is the process",
        "reorder point",
        "minimum stock",
        "maximum stock",
        "safety stock",
        "inventory replenishment",

        "معادلة",
        "حساب",
        "طريقة الحساب",
        "كيفية الحساب",
        "إجراء",
        "إجراءات",
        "عملية",
        "قاعدة",
        "قواعد",
        "تعليمات",
        "الدليل",
        "الوثيقة",
        "حسب الدليل",
        "وفقا للدليل",
        "حد الطلب",
        "الحد الأدنى",
        "الحد الأقصى",
        "المخزون",
        "إعادة الطلب",]

    # Conditional wording: "if / in case / what if ..."
    CONDITIONAL_MARKERS = [
        "إذا", "إذا ما", "لو", "في حال", "في حالة", "ماذا لو",
        "if ", "in case", "what if",
    ]

    # Exception / fallback wording: signals a genuine impossibility or an
    # explicit ask for a workaround/alternative — NOT generic "what happens
    # if" phrasing, which also covers plain single-condition lookups that
    # belong in basic_rag (e.g. "what happens if a card is lost").
    EXCEPTION_MARKERS = [
        "لم يمكن", "لم تتمكن", "لم يتم", "تعذر", "تعذّر",
        "عدم إمكانية", "عدم امكانية", "عدم القدرة",
        "البديل", "بديل", "الحل البديل",
        "cannot", "can not", "could not", "not possible",
        "unable to", "alternative",
    ]

    # A second (or third) distinct question clause joined by "و" ("and") —
    # in Arabic this attaches directly to the next word with no space
    # ("ومن", "وكيف", "وما"...). Signals the question bundles multiple
    # separate information needs, which decomposition handles far better
    # than a single merged retrieval pass.
    MULTI_PART_MARKERS = [
        "ومن", "وكيف", "وما", "ومتى", "وأين", "ولماذا", "وهل", "وكم",
        "و من", "و كيف", "و ما", "و متى", "و أين", "و لماذا", "و هل", "و كم",
        " and what ", " and who ", " and how ", " and when ",
        " and where ", " and why ", " and does ", " and is ",
    ]

    def __init__(self, llm):
        self.llm = llm

    def _is_multi_part_question(self, question: str) -> bool:
        q = question.strip()
        q = re.sub(r"\s+", " ", q)
        ql = q.lower()
        return any(marker in q for marker in self.MULTI_PART_MARKERS) or \
            any(marker in ql for marker in self.MULTI_PART_MARKERS)

    def _is_document_sensitive(self, question: str) -> bool:
        q = question.lower().strip()
        q = re.sub(r"\s+", " ", q)
        return any(term in q for term in self.DOCUMENT_SENSITIVE_TERMS)

    def _is_conditional_scenario(self, question: str) -> bool:
        q = question.lower().strip()
        q = re.sub(r"\s+", " ", q)
        has_conditional = any(term in q for term in self.CONDITIONAL_MARKERS)
        has_exception = any(term in q for term in self.EXCEPTION_MARKERS)
        return has_conditional and has_exception

    def route(self, question: str) -> RouteDecision:
        if self._is_multi_part_question(question):
            return RouteDecision(
                route="advanced_rag",
                reason=( "The question bundles multiple distinct information "
                         "needs joined by 'and' (e.g. a procedure, plus who is "
                         "involved, plus how exceptions/discrepancies are "
                         "handled). Each part is retrieved independently via "
                         "decomposition so no sub-question is diluted by the "
                         "others, then the evidence is compressed before the "
                         "final answer."),
                techniques=["decomposition", "compression"],)

        if self._is_conditional_scenario(question):
            return RouteDecision(
                route="advanced_rag",
                reason=( "The question describes a conditional/exception "
                         "scenario (e.g. 'if X fails and Y is not possible, "
                         "what is the alternative'). This phrasing is unlikely "
                         "to match the manual's own wording verbatim, so query "
                         "rewriting and reranking are used to improve retrieval, "
                         "with a corrective check in case the first pass misses "
                         "the relevant procedure."),
                techniques=["rewriting", "reranking", "crag"],)

        if self._is_document_sensitive(question):
            return RouteDecision(
                route="basic_rag",
                reason=( "The question asks for a formula, procedure, rule, "
                         "calculation method, or document-sensitive fact. "
                         "Retrieval is required to ensure the answer follows "
                          "the supplied documents rather than general knowledge."), techniques=[],)
        prompt = ( f"Question:\n{question}\n\n" "Return the JSON routing decision." )
        try:
            obj = self.llm.generate_json( self.SYSTEM, prompt, max_tokens=300, step="router", )
            route = obj.get("route", "basic_rag")
            if route not in { "simple", "basic_rag", "advanced_rag", }:
                route = "basic_rag"

            techniques = [ t for t in obj.get("techniques", []) if t in { "rewriting", "multi_query", "decomposition", "hyde", "self_query", "reranking", "compression", "crag", }]
            return RouteDecision( route, str(obj.get("reason", "")), techniques, )
        
        except Exception:
            q = question.lower()
            if any( x in q for x in ["compare", "advantages", "disadvantages", "and explain", "multiple", ]):
                return RouteDecision( "advanced_rag", "Multi-part question", [ "decomposition",  "compression", ],)

            if any( x in q for x in [ "after 2024", "before ", "published", "year", "date", ]):
                return RouteDecision( "advanced_rag", "Metadata/date constraint", [ "self_query", ],)

            if any( x in q for x in [ "why is it bad", "what does it refer", "this", "that", ]):
                return RouteDecision( "advanced_rag", "Ambiguous wording", [ "rewriting", ],)

            if ( len(question.split()) <= 7 and any( x in q for x in ["what is", "define", "why", "how", ]) ):
                return RouteDecision( "basic_rag", "Focused question", [], )

            return RouteDecision( "basic_rag", "Default focused retrieval", [], )