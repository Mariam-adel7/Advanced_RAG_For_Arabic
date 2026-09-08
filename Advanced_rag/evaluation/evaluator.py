from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class EvaluationRecord:
    id: str
    question: str
    route: str
    techniques: List[str]
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    correctness: Optional[float]
    total_cost: float
    latency: float
    calls: List[Dict[str, Any]]
    details: Dict[str, Any]

def _tokens(text):
    text = str(text or "").lower()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return set(re.findall(r"[A-Za-z0-9_]+|[\u0600-\u06FF]+", text))

def lexical_relevance(question, context):
    q, c = _tokens(question), _tokens(context)
    return round(min(1.0, len(q & c) / max(1, len(q) * 0.45)), 3)

def faithfulness_heuristic(answer, context):
    if not context:
        return 1.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    vals = []
    c = _tokens(context)
    for s in sentences:
        if "[source" in s.lower() and len(_tokens(s)) > 3:
            vals.append(min(1.0, len(_tokens(s) & c) / max(1, len(_tokens(s)) * 0.45)))
        else:
            vals.append(min(1.0, len(_tokens(s) & c) / max(1, len(_tokens(s)) * 0.55)))
    return round(sum(vals)/len(vals), 3)

def answer_relevance(question, answer):
    q, a = _tokens(question), _tokens(answer)
    return round(min(1.0, len(q & a) / max(1, len(q) * 0.35)), 3)

class Evaluator:

    RUBRIC = """Score each metric from 0 to 1 using the same rubric:
context_relevance: retrieved context contains evidence useful for the question.
faithfulness: answer claims are supported by the retrieved context (for simple
questions with no context, judge whether the answer avoids unsupported
document claims).
answer_relevance: answer directly addresses the user's information need.
correctness: answer is factually correct; use the supplied reference answer
when present, otherwise judge against the question and context conservatively.
Return JSON only:
{"context_relevance":0,"faithfulness":0,"answer_relevance":0,"correctness":0,
 "reason":"..."}"""
    
    def __init__(self, llm=None, use_llm_judge=True):
        self.llm = llm
        self.use_llm_judge = use_llm_judge

    def evaluate(self, qid, result, reference_answer: Optional[str] = None):
        context = str(getattr(result, "context", "") or "")
        answer = str(getattr(result, "answer", "") or "")
        question = str(getattr(result, "question", "") or "")
        route_obj = getattr(result, "route", None)
        route = getattr(route_obj, "route", str(route_obj or ""))
        techniques = list(getattr(result, "techniques_used", []) or [])
        cr = lexical_relevance(question, context) if context else (1.0 if route == "simple" else 0.0)
        fa = faithfulness_heuristic(answer, context) if context else 1.0 if route == "simple" else 0.0
        ar = answer_relevance(question, answer)
        co = None
        judge_reason = ""

        if self.use_llm_judge and self.llm is not None:
            prompt = f"""Question:
{question}

Retrieved context:
{context or '(none; this was a direct/simple question)'}

Answer:
{answer}

Reference answer (if available):
{reference_answer or '(not available)'}

{self.RUBRIC}

Important: if no reference answer is available, score correctness conservatively from the supplied context; do not equate correctness with answer relevance."""
            try:
                obj = self.llm.generate_json( "You are a strict, consistent RAG evaluation judge.", prompt, max_tokens=400, step="evaluation_judge", )
                cr = float(obj.get("context_relevance", cr))
                fa = float(obj.get("faithfulness", fa))
                ar = float(obj.get("answer_relevance", ar))
                if reference_answer:
                    co = float(obj.get("correctness", 0.0))
                else:
                    co = None
                judge_reason = str(obj.get("reason", ""))
            except Exception as exc:
                judge_reason = f"LLM judge unavailable: {exc}"

        if reference_answer and co is None:
            ref = _tokens(reference_answer)
            ans = _tokens(answer)
            overlap = len(ref & ans)
            precision = overlap / max(1, len(ans))
            recall = overlap / max(1, len(ref))
            co = round(2 * precision * recall / max(1e-9, precision + recall), 3) if precision + recall else 0.0

        calls = list(getattr(result, "calls", []) or [])
        if self.llm is not None:
            try:
                result_call_count = len(calls)
                calls = calls + [asdict(c) for c in self.llm.calls[result_call_count:]]
            except Exception:
                pass

        total_cost = sum(float(c.get("cost", 0.0) or 0.0) for c in calls)
        latency = float(getattr(result, "latency", 0.0) or 0.0)
        intermediate = getattr(result, "intermediate", {}) or {}
        reason = getattr(route_obj, "reason", "")

        return EvaluationRecord(
            id=qid,
            question=question,
            route=route,
            techniques=techniques,
            context_relevance=round(max(0, min(1, cr)), 3),
            faithfulness=round(max(0, min(1, fa)), 3),
            answer_relevance=round(max(0, min(1, ar)), 3),
            correctness=(round(max(0, min(1, co)), 3) if co is not None else None),
            total_cost=total_cost,
            latency=latency,
            calls=calls,
            details={
                "reason": reason,
                "intermediate": intermediate,
                "reference_answer": reference_answer,
                "judge_reason": judge_reason,
                "correctness_status": "scored_against_reference" if reference_answer else "not_scored_without_reference",
            },
        )

def save_results(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

def save_csv(records, path):
    import csv
    rows = [asdict(r) for r in records]
    fields = ["id","route","techniques","context_relevance","faithfulness", "answer_relevance","correctness","total_cost","latency"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k:r[k] for k in fields})
