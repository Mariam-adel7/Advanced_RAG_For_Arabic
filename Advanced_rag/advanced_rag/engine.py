from __future__ import annotations
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List
from advanced_rag.router import Router, RouteDecision
from evaluation.evaluator import Evaluator
from retrieval.base import RetrievalResult
from sentence_transformers import CrossEncoder

@dataclass
class AdvancedResult:
    question: str
    answer: str
    route: RouteDecision
    techniques_used: List[str]
    retrieved: List[RetrievalResult]
    context: str
    sources: Any
    calls: List[Dict[str, Any]]
    latency: float
    intermediate: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    confidence_label: str = "Very Low"
    total_cost: float = 0.0
    cost_by_technique: Dict[str, float] = field(default_factory=dict)
    evaluation: Any = None

class AdvancedRAGEngine:
    FINAL_SYSTEM = """You are a careful document-grounded assistant.

GROUNDING IS MANDATORY:
- Use ONLY the supplied evidence/context for document-grounded claims.
- Do not use your pretrained/general knowledge to fill missing information.
- Do not infer a rule, number, date, threshold, procedure, or policy that is
  not explicitly supported by the supplied evidence.
- If the evidence does not answer the question, explicitly say that the
  supplied documents do not provide enough information.
- If there is no formal definition sentence but the evidence gives an
  operational description of the requested concept, answer from that
  description instead of declaring the evidence insufficient. When useful,
  introduce it with "بحسب الإجراء في الدليل".
- Every factual claim derived from the documents must have a citation using
  ONLY a source label that actually appears in the supplied context.
- NEVER invent, renumber, or guess a [Source N] citation.
- Keep source/page information exactly as supplied by the context.
- Do not claim a page number unless a page number is present in the evidence.

EVIDENCE SELECTION:
- Prefer evidence that directly answers the question over merely related text.
- If multiple evidence blocks support different parts of the answer, cite each
  claim with the corresponding source label.
- Do not merge facts from unrelated chunks.
- If two evidence blocks conflict, report the conflict instead of silently
  choosing or correcting one.
- For Arabic source text, preserve the source terminology and meaning as much
  as possible; translate only as needed to answer the user's language.

TABLE RULES:
- If the context contains a table, align each value with its correct row and
  column.
- Do not mix values from different rows.
- Use table labels to identify the correct information.

FORMULA RULES:
- If the context contains a formula that answers the question, reproduce the
  document's formula faithfully.
- Do not replace a document-specific formula with a standard formula from
  general knowledge.
- Do not change numbers, time periods, variables, units, or operators.
- Do not reinterpret or "correct" a formula.
- If multiple formulas appear, select the formula whose row, label, or
  description directly matches the question.
- If the exact formula cannot be determined from the supplied context, say so.

OUTPUT FORMAT:
- Return ONLY the final answer to the user.
- Do NOT output chain-of-thought or internal reasoning.
- Do NOT output sections named "Thinking", "Analysis", "Reasoning", or similar.
- Do NOT describe how you searched or ranked the documents.
- Do NOT mention retrieval, reranking, prompts, or internal processing.
- Do NOT explain your internal decision-making process.
- If the evidence is insufficient, say so directly and briefly.

Return a concise, direct answer.
"""

    TRANSFORM_SYSTEM = """You transform search queries for a document RAG system.
Return JSON only.
Never answer the user's question.
Preserve the user's meaning.
Never invent metadata values.
For bilingual/document retrieval, it is acceptable to produce equivalent
queries in the source language when this improves retrieval.

When transforming an English question for an Arabic document:
- Preserve important English concepts.
- Produce natural Arabic terminology.
- Prefer terminology likely to appear in a formal organizational manual.
- Do not invent a specific procedure name, rule, number, or fact.
"""

    def __init__( self, config, retriever, llm_client, context_builder, prompt_builder, top_k=5, max_tokens=2048, debug=False, reranker_model="BAAI/bge-reranker-v2-m3", reranker_candidates=15, ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_client
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.debug = debug
        self.router = Router(llm_client)
        self.evaluator = Evaluator( llm=llm_client, use_llm_judge=True, )
        self.reranker_model_name = reranker_model
        self.reranker_candidates = max( self.top_k, reranker_candidates,)
        self.cross_encoder = CrossEncoder( self.reranker_model_name )
        self.simulated_pricing = {
            "default": {"input": 0.10, "output": 0.40},
            "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
            "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
            "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
            "qwen3-max": {"input": 1.20, "output": 6.00},
            "qwen-plus": {"input": 0.40, "output": 1.20},
            "qwen-turbo": {"input": 0.05, "output": 0.20}, }

    def _get_simulated_price(self, model):
        model_name = str(model or "").lower()
        candidates = [name for name in self.simulated_pricing if name != "default"]
        for name in sorted(candidates, key=len, reverse=True):
            if name.lower() in model_name:
                return self.simulated_pricing[name]
        return self.simulated_pricing["default"]

    def _cost_technique(self, step):
        step = str(step or "unknown").strip()
        mapping = { "router": "routing", "rewriter": "rewriting", "multi_query": "multi_query", "decomposition": "decomposition", "hyde": "hyde", "self_query": "self_query", "final_generation": "final_generation", "direct_generation": "direct_generation", }
        return mapping.get(step, step)

    def _build_cost_report(self):
        calls = []
        cost_by_technique = {}
        total_cost = 0.0

        for call in self.llm.calls:
            item = asdict(call)
            input_tokens = int(item.get("input_tokens") or 0)
            output_tokens = int(item.get("output_tokens") or 0)
            pricing = self._get_simulated_price( item.get("model", ""))
            input_cost = ( input_tokens / 1_000_000 ) * pricing["input"]
            output_cost = ( output_tokens / 1_000_000 ) * pricing["output"]
            simulated_cost = input_cost + output_cost
            item["actual_cost"] = item.get("cost", 0.0)
            item["cost"] = round( simulated_cost, 10, )
            item["cost_type"] = "simulated"
            item["input_cost"] = round(input_cost, 10)
            item["output_cost"] = round(output_cost, 10)
            technique = self._cost_technique( item.get("step", "unknown") )
            item["technique"] = technique
            cost_by_technique[technique] = ( cost_by_technique.get(technique, 0.0) + simulated_cost)
            total_cost += simulated_cost
            calls.append(item)

        cost_by_technique = {  key: round(value, 10)  for key, value in cost_by_technique.items() }
        return ( calls,  round(total_cost, 10), cost_by_technique, )

    def _attach_cost_report(self, intermediate):
        calls, total_cost, cost_by_technique = ( self._build_cost_report() )
        intermediate["cost"] = { "type": "simulated", "currency": "USD", "total": total_cost, "by_technique": cost_by_technique, "pricing_per_1m_tokens": self.simulated_pricing, }
        return calls, total_cost, cost_by_technique

    def _call_llm_with_retry( self, func, *args, **kwargs, ):
        max_retries = 3
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2

    def _evaluate_result( self, result: AdvancedResult, qid: str = "question", reference_answer: str | None = None, ):
        try:
            return self.evaluator.evaluate( qid=qid, result=result, reference_answer=reference_answer, )
        except Exception as e:
            if self.debug:
                print("Evaluation error:", e)
            return None

    def answer( self, question: str, chat_history=None, evaluate: bool = False, qid: str = "question", reference_answer: str | None = None, ) -> AdvancedResult:
        started = time.perf_counter()
        self.llm.reset_calls()
        route = self.router.route(question)
        techniques = list(route.techniques)
        intermediate: Dict[str, Any] = {}

        if route.route == "simple":
            answer = self._direct_answer( question, chat_history, )
            calls, total_cost, cost_by_technique = self._attach_cost_report(intermediate)
            return AdvancedResult( question, answer, route, [], [], "", [], calls, time.perf_counter() - started, intermediate, 0.0, "Not Applicable", total_cost, cost_by_technique, )

        queries = [question]
        metadata_filter = {}
        decomposed_subquestions = []

        if route.route == "advanced_rag":
            generated_queries: List[str] = []
            if "rewriting" in techniques:
                q = self._rewrite(question)
                if q:
                    generated_queries.append(q)
                    intermediate["rewritten_query"] = q

            if "multi_query" in techniques:
                qs = self._multi_query(question)
                generated_queries.extend(qs)
                intermediate["multi_queries"] = qs

            if "decomposition" in techniques:
                decomposed_subquestions = self._decompose( question)
                intermediate["subquestions"] = ( decomposed_subquestions )

            if "hyde" in techniques:
                hypo = self._hyde(question)
                generated_queries.append(hypo)
                intermediate["hyde_document"] = hypo

            if "self_query" in techniques:
                semantic_query, metadata_filter = ( self._self_query(question) )
                if semantic_query:
                    generated_queries.append( semantic_query )
                intermediate["self_query"] = { "semantic": semantic_query, "filters": metadata_filter, }

            queries = list( dict.fromkeys([question] + generated_queries ) )

        if decomposed_subquestions:
            decomposed_hits = self._retrieve_decomposed( decomposed_subquestions, metadata_filter, )
            original_hits = self._retrieve_many( [question], metadata_filter, )
            hits = self._merge_hits( decomposed_hits + original_hits )
            intermediate["retrieval_mode"] = ( "decomposition")

        else:
            hits = self._retrieve_many( queries, metadata_filter,)
            intermediate["retrieval_mode"] = (  "multi_query"  if len(queries) > 1  else "single_query" )

        retrieval_debug = { "queries": self._query_retrieval_stats(queries,metadata_filter, )}

        if decomposed_subquestions:
            retrieval_debug["decomposition"] = ( self._decomposition_retrieval_stats( decomposed_subquestions, metadata_filter, ))
        intermediate["retrieval_debug"] = retrieval_debug

        if self.debug:
            self._debug_retrieval( queries, metadata_filter, hits, )

        if ( "reranking" in techniques or route.route == "advanced_rag" ):
            hits = self._rerank( question, hits )
            if route.route == "advanced_rag" and "reranking" not in techniques:
                techniques.append("reranking")

        if self.debug:
            self._debug_print_chunks( hits )

        confidence_details = ( self._calculate_confidence( question, hits, ))
        retrieval_confidence = ( confidence_details["confidence"] )
        intermediate["confidence"] = ( confidence_details )

        # NOTE: Auto-escalation from basic_rag -> advanced_rag on low confidence
        # has been removed. A basic_rag route now always stays basic_rag;
        # advanced techniques are only used when the Router itself decides
        # they're needed up front (route.route == "advanced_rag").

        if "crag" in techniques:
            crag = self._crag( question, hits, )
            intermediate["crag"] = crag

            if crag["action"] == "rewrite":
                q = self._rewrite( question, )
                retry_hits = self._retrieve_many( [q], metadata_filter, )
                hits = self._rerank( question, retry_hits, )

                if self.debug:
                    self._debug_print_chunks( hits )

                confidence_details = ( self._calculate_confidence( question, hits, ) )
                retrieval_confidence = ( confidence_details["confidence"] )
                intermediate["confidence"] = (confidence_details)

        if "compression" in techniques:
            before = sum( len(h.chunk.text) for h in hits )
            hits = self._compress( question, hits, )
            after = sum( len(h.chunk.text) for h in hits )
            intermediate["compression"] = { "chars_before": before, "chars_after": after, "chars_removed":max(0,before - after,) }

        context, sources = ( self.context_builder.build( hits ))
        answer = self._final_answer( question, context, sources, chat_history, )
        result = AdvancedResult( question, answer, route, techniques, hits, context, sources, [],  time.perf_counter() - started, intermediate, retrieval_confidence, self._confidence_label(retrieval_confidence), 0.0,  {}, )
        if evaluate:
            evaluation = self._evaluate_result( result, qid=qid, reference_answer=reference_answer, )
            result.evaluation = evaluation

            if evaluation is not None:
                intermediate["evaluation"] = asdict(evaluation)

        calls, total_cost, cost_by_technique = self._attach_cost_report( intermediate )
        result.calls = calls
        result.total_cost = total_cost
        result.cost_by_technique = cost_by_technique

        return result

    def _retrieve_decomposed( self, subquestions, metadata_filter, ):
        all_results = []
        for index, subquestion in enumerate( subquestions, start=1, ):
            sub_hits = self._retrieve_single( subquestion, metadata_filter, )
            for hit in sub_hits:
                metadata = dict( getattr(hit.chunk,"metadata",{}, ) or {})
                metadata[ "_decomposition_subquestion" ] = subquestion
                metadata[ "_decomposition_index" ] = index
                metadata[ "_retrieval_query" ] = subquestion
                metadata.setdefault( "_retrieval_queries", [subquestion], )
                metadata.setdefault(  "_retrieval_scores",  [float(hit.score)], )
                chunk = replace( hit.chunk, metadata=metadata,)
                all_results.append( RetrievalResult( chunk=chunk, score=hit.score, retriever=hit.retriever, ))
        return all_results

    def basic_answer( self, question: str, chat_history=None, evaluate: bool = False, qid: str = "question", reference_answer: str | None = None ):
        started = time.perf_counter()
        self.llm.reset_calls()
        hits = self._retrieve_many( [question], {}, )
        hits = hits[:self.top_k]

        if self.debug:
            self._debug_print_chunks( hits )

        confidence_details = ( self._calculate_confidence( question, hits, ))
        context, sources = ( self.context_builder.build( hits ))
        answer = self._final_answer(  question,  context,  sources,  chat_history,)
        route = RouteDecision( "basic_rag", "Baseline for comparison", [], )
        intermediate = { "confidence": confidence_details }
        result = AdvancedResult( question, answer, route, [], hits, context, sources, [], time.perf_counter() - started, intermediate, confidence_details["confidence"], confidence_details["label"], 0.0, {}, )

        if evaluate:
            evaluation = self._evaluate_result( result, qid=qid, reference_answer=reference_answer, )
            result.evaluation = evaluation

            if evaluation is not None:
                intermediate["evaluation"] = asdict(evaluation)

        calls, total_cost, cost_by_technique = self._attach_cost_report( intermediate )
        result.calls = calls
        result.total_cost = total_cost
        result.cost_by_technique = cost_by_technique

        return result

    def _direct_answer( self, question, chat_history, ):
        messages = list( chat_history or [] ) + [ { "role": "user", "content": question, } ]
        return self._call_llm_with_retry( self.llm.generate, ( "Answer the general-knowledge "
         "question directly. "
         "Do not claim to use supplied documents." ), messages, max_tokens=self.max_tokens, step="direct_generation", )

    def _final_answer( self, question, context, sources, chat_history, ):
        system, messages = ( self.prompt_builder.build_messages( question, context, sources, chat_history, ) )
        system = ( system + "\n\n" + self.FINAL_SYSTEM )
        return self._call_llm_with_retry( self.llm.generate, system, messages, max_tokens=self.max_tokens, step="final_generation", )

    def _retrieve_single( self, q, metadata_filter, ):
        try:
            hits = self.retriever.retrieve( q, top_k=max(self.top_k,8, ), metadata_filter=(metadata_filter or None ), )
            native_filtering = bool( metadata_filter )
        except TypeError:
            hits = self.retriever.retrieve( q, top_k=max(self.top_k,8, ), )
            native_filtering = False

        filtered_hits = []
        for hit in hits:
            if ( metadata_filter and not native_filtering and not self._metadata_match(hit,metadata_filter, ) ):
                continue
            filtered_hits.append(hit)
        return filtered_hits

    def _retrieve_many( self, queries, metadata_filter, ):
        merged = {}
        def retrieve_with_query(q):
            hits = self._retrieve_single( q, metadata_filter, )
            enriched = []

            for hit in hits:
                metadata = dict( getattr( hit.chunk, "metadata", {}, ) or {} )
                metadata[ "_retrieval_query" ] = q
                metadata.setdefault( "_retrieval_queries", [q], )
                metadata.setdefault( "_retrieval_scores", [float(hit.score)], )
                chunk = replace( hit.chunk, metadata=metadata, )
                enriched.append( RetrievalResult( chunk=chunk, score=hit.score, retriever=hit.retriever, ) )
            return enriched

        with ThreadPoolExecutor( max_workers=min( 8, max(1,len(queries), ), ) ) as executor:
            results = executor.map( retrieve_with_query, queries, )

        for hits in results:
            for hit in hits:
                chunk_id = hit.chunk.chunk_id
                old = merged.get( chunk_id )
                if old is None:
                    merged[chunk_id] = hit
                    continue

                old_md = dict( getattr( old.chunk, "metadata", {}, ) or {} )
                new_md = dict( getattr( hit.chunk, "metadata", {}, ) or {} )
                old_queries = list( old_md.get( "_retrieval_queries", [],))
                new_queries = list(new_md.get( "_retrieval_queries", [], ) )
                old_scores = list( old_md.get( "_retrieval_scores", [],))
                new_scores = list( new_md.get( "_retrieval_scores", [],))

                for q in new_queries:
                    if q not in old_queries:
                        old_queries.append(q)

                old_scores.extend( float(x) for x in new_scores )
                old_md[ "_retrieval_queries" ] = old_queries
                old_md[ "_retrieval_scores" ] = old_scores

                if hit.score > old.score:
                    old_md[ "_retrieval_query" ] = new_md.get( "_retrieval_query", "", )
                    merged[chunk_id] = ( RetrievalResult( chunk=replace(hit.chunk,metadata=old_md, ), score=hit.score, retriever=hit.retriever, ) )

                else:
                    merged[chunk_id] = ( RetrievalResult( chunk=replace( old.chunk, metadata=old_md, ), score=old.score, retriever=old.retriever, ))

        return sorted( merged.values(), key=lambda x: x.score, reverse=True, )[ :max( self.top_k * 4, 20, )]

    @staticmethod
    def _merge_hits( hits ):
        merged = {}
        for hit in hits:
            chunk_id = hit.chunk.chunk_id
            old = merged.get( chunk_id )

            if old is None:
                md = dict( getattr( hit.chunk, "metadata", {}, ) or {} )
                md.setdefault( "_retrieval_queries", [], )
                md.setdefault( "_retrieval_scores", [], )
                q = md.get( "_retrieval_query" )
                if ( q and q not in md["_retrieval_queries" ] ):
                    md[ "_retrieval_queries" ].append(q)
                md[ "_retrieval_scores" ].append(float(hit.score) )
                merged[chunk_id] = ( RetrievalResult( chunk=replace(hit.chunk,metadata=md, ), score=hit.score, retriever=hit.retriever,))
                continue

            old_md = dict( getattr( old.chunk,"metadata", {}, ) or {} )
            new_md = dict( getattr( hit.chunk,"metadata", {}, ) or {} )
            queries = list(old_md.get( "_retrieval_queries", [], ))
            for q in new_md.get( "_retrieval_queries", [], ):
                if q not in queries:
                    queries.append(q)

            q = new_md.get( "_retrieval_query" )
            if q and q not in queries:
                queries.append(q)

            scores = list( old_md.get( "_retrieval_scores", [], ))
            scores.extend( float(x) for x in new_md.get("_retrieval_scores",[hit.score], ))
            old_md[ "_retrieval_queries" ] = queries
            old_md[ "_retrieval_scores" ] = scores

            if hit.score > old.score:
                old_md[ "_retrieval_query" ] = new_md.get( "_retrieval_query", old_md.get("_retrieval_query" ),)
                merged[chunk_id] = ( RetrievalResult( chunk=replace(hit.chunk,metadata=old_md, ), score=hit.score, retriever=hit.retriever,))
            else:
                merged[chunk_id] = ( RetrievalResult( chunk=replace(old.chunk,metadata=old_md, ), score=old.score, retriever=old.retriever,))

        return sorted( merged.values(), key=lambda x: x.score, reverse=True, )[ :max(20,4 * 5, )]

    def _query_retrieval_stats( self, queries, metadata_filter, ):
        stats = []
        for index, query in enumerate( queries, start=1, ):
            hits = self._retrieve_single( query, metadata_filter,)
            stats.append({
                    "query_index": index,
                    "query": query,
                    "result_count": len( hits ),
                    "top_scores": [ round( float(h.score), 6, ) for h in hits[:5]],
                    "top_chunks": [{
                            "chunk_id": h.chunk.chunk_id,
                            "page": self._page(h.chunk ),
                            "section": self._section( h.chunk ),
                            "source": self._source( h.chunk ),
                            "text": h.chunk.text[:500], } for h in hits[:5] ], } )
        return stats

    def _decomposition_retrieval_stats( self, subquestions, metadata_filter, ):
        stats = []
        for index, subquestion in enumerate( subquestions, start=1, ):
            hits = self._retrieve_single( subquestion, metadata_filter, )
            stats.append( {
                    "subquestion_index": index,
                    "subquestion": subquestion,
                    "result_count": len(hits),
                    "top_scores": [ round( float(h.score), 6, ) for h in hits[:5] ],
                    "top_chunks": [{
                        "chunk_id": h.chunk.chunk_id,
                        "page": self._page(h.chunk ),
                        "section": self._section( h.chunk ),
                        "source": self._source( h.chunk ),
                        "text": h.chunk.text[:500], } for h in hits[:5] ], } )
        return stats

    @staticmethod
    def _page(chunk):
        page = getattr( chunk, "page", None,)
        if page is not None:
            return page
        metadata = getattr( chunk, "metadata", {}, ) or {}
        return metadata.get("page")

    @staticmethod
    def _section(chunk):
        section = getattr( chunk, "section", None, )
        if section is not None:
            return section
        metadata = getattr( chunk, "metadata", {}, ) or {}
        return metadata.get( "section")

    @staticmethod
    def _source(chunk):
        source = getattr(chunk,"source",None, )
        if source:
            return source
        metadata = getattr( chunk, "metadata", {}, ) or {}
        return metadata.get( "source")

    def _debug_retrieval( self, queries, metadata_filter=None, hits=None, ):
        if not self.debug:
            return
        print( "\n" + "=" * 100 )
        print( "QUERY-BY-QUERY RETRIEVAL DEBUG")
        print(  "=" * 100 )
        print( "METADATA FILTER:", metadata_filter or {}, )

        for index, query in enumerate( queries, start=1,):
            print( f"\nQUERY #{index}:" )
            print(query)
            query_hits = ( self._retrieve_single( query, metadata_filter or {}, ))
            print( f"RESULT COUNT: {len(query_hits)}" )
            for j, hit in enumerate( query_hits[:5], start=1, ):
                chunk = hit.chunk

                print( f"\n  Candidate {j}" )
                print( "  Score:", round(float(hit.score),6, ),)
                print( "  Retriever:", hit.retriever,)
                print( "  Chunk:", chunk.chunk_id,)
                print( "  Page:", self._page(chunk),)
                print( "  Section:", self._section(chunk),)
                print( "  Source:", self._source(chunk),)
                print( "  Text:", chunk.text[:500],)
            print( "\n" + "=" * 100 )

    def _rerank( self, question, hits, ):
        if not hits:
            return []
        candidates = hits[ :self.reranker_candidates ]
        pairs = []
        ranking_queries = []
        for hit in candidates:
            metadata = getattr( hit.chunk,"metadata", {}, ) or {} 
            subq = metadata.get( "_decomposition_subquestion" )
            variants = list( metadata.get("_retrieval_queries",[], ))
            if ( metadata.get("_retrieval_query" ) and metadata["_retrieval_query" ] not in variants ):
                variants.append( metadata[ "_retrieval_query" ] )
            if subq:
                rank_queries = [ subq, question,] + variants
            else:
                rank_queries = [ question ] + variants

            unique_queries = []

            for q in rank_queries:
                q = str(q).strip()
                if ( q and q not in unique_queries ):
                    unique_queries.append(q)

            unique_queries = unique_queries[:5]
            ranking_queries.append(unique_queries)
            pairs.extend((q,hit.chunk.text,)for q in unique_queries )

        cross_scores = (self.cross_encoder.predict( pairs, batch_size=8, show_progress_bar=False, ))
        cross_scores = [ float(x) for x in cross_scores]
        scored = []
        cursor = 0
        bonuses = ( self.config.rerank_bonuses)

        for hit, queries_for_hit in zip( candidates, ranking_queries,):
            n = len( queries_for_hit )
            logits = cross_scores[ cursor: cursor + n ]
            cursor += n
            features = [  self._sigmoid(x)  for x in logits ]
            best_index = max( range(len(features)), key=lambda i: features[i], )
            ce_feature = features[ best_index]
            ce_logit = logits[ best_index ]
            best_query = ( queries_for_hit[best_index ])
            text = ( hit.chunk.text or "" ).lower()
            q = ( best_query.lower() )
            q_words = set( re.findall(r"\w+",q, ))
            words = set( re.findall(r"\w+",text, ))
            lexical_overlap = (len(q_words & words)/ max(1,len(q_words),))
            lexical_overlap = min( 1.0, lexical_overlap,)
            variant_overlaps = []

            for variant in queries_for_hit:
                v_words = set( re.findall(r"\w+",variant.lower(), ))
                if v_words:
                    variant_overlaps.append(len( v_words & words)/ len(v_words))

            lexical_overlap = max( [lexical_overlap] + variant_overlaps )
            table_bonus = ( bonuses["table_bonus"] if hit.chunk.chunk_type == "table" else 0.0 )
            formula_pattern = re.compile(r"\d\s*[=×÷+\-]\s*\d")
            formula_bonus = bonuses["formula_bonus"] if (formula_pattern.search(text) or "معادلة" in text or "حد الطلب" in text) else 0.0
            important_terms = [
                "حد الطلب",
                "الحد الأدنى",
                "الحد الأقصى",
                "المخزون",
                "إعادة الطلب",
                "الاستهلاك",
                "مدة المادة",
                "معادلة",
                "حساب",
                "المخزون الراكد",
                "راكد",
                "مستودع",
                "مستودعات",
                "stagnant",
                "warehouse",
                "material",
                "manual",
                "procedure", ]

            matched_terms = sum( 1 for term in important_terms if any( term in variant.lower() for variant in queries_for_hit ) and term in text )
            document_keyword_bonus = min(bonuses["keyword_bonus_max"],matched_terms* bonuses["keyword_bonus_step"], )
            exact_phrases = [ "المخزون الراكد", "حد الطلب", "الحد الأدنى", "الحد الأقصى", "إعادة الطلب", ]
            phrase_bonus = 0.0

            for phrase in exact_phrases:
                phrase_lower = ( phrase.lower() )
                if ( any( phrase_lower in variant.lower() for variant in queries_for_hit ) and phrase_lower in text ):
                    phrase_bonus += 0.08

            phrase_bonus = min( phrase_bonus, bonuses["keyword_bonus_max" ],)
            retrieval_feature = min( 1.0, max( 0.0, float(hit.score) * 10.0, ),)
            real_hit_count = len(metadata.get("_retrieval_queries", []))
            multi_hit_bonus = min(0.08, max(0, real_hit_count - 1) * 0.02)            
            new_score = ( 0.74 * ce_feature + 0.07 * retrieval_feature + 0.11 * lexical_overlap + table_bonus + formula_bonus + document_keyword_bonus + phrase_bonus + multi_hit_bonus)
            metadata = dict( getattr(hit.chunk,"metadata",{}, ) or {})
            metadata["_rerank_ce_logit"] = ce_logit
            metadata["_rerank_ce_feature"] = ce_feature
            metadata["_rerank_ce_best_query"] = best_query
            metadata["_rerank_query_variants"] = queries_for_hit
            metadata["_original_retrieval_score"] = float(hit.score)
            metadata["_rerank_lexical_overlap"] = lexical_overlap
            metadata["_rerank_match_count"] = len(queries_for_hit)
            metadata["_rerank_phrase_bonus" ] = phrase_bonus
            scored.append( ( new_score, RetrievalResult( chunk=replace( hit.chunk, metadata=metadata,),score=new_score,retriever=hit.retriever,), ce_logit,ce_feature,lexical_overlap,best_query,))
        scored.sort( key=lambda x: x[0], reverse=True,)
        final_hits = [ x[1] for x in scored[ :max(5,self.top_k, )]]

        if self.debug:
            print( "\n" + "=" * 100 )
            print("CROSS-ENCODER RERANKING" )
            print( "=" * 100 )
            print( "Model:", self.reranker_model_name, )
            print( "Original question is always included; "
                   "best multilingual query variant is selected." )

            for rank, item in enumerate( scored, start=1, ):
                print( f"\nRank #{rank}" )
                print("Chunk ID       :",item[1].chunk.chunk_id,)
                print( "Page           :", self._page(item[1].chunk ),)
                print( "Source         :", self._source(item[1].chunk ),)
                print( "Best query     :", item[5],)
                print( f"CE Feature     : {item[3]:.4f}" )
                print( f"CE Logit       : {item[2]:.4f}" )
                print( f"Lexical overlap: {item[4]:.4f}" )
                metadata = getattr( item[1].chunk, "metadata", {}, ) or {}
                print( "Phrase bonus   :", f"{metadata.get('_rerank_phrase_bonus', 0.0):.4f}",)
                print( f"Final score    : {item[0]:.4f}")
            print( "\n" + "=" * 100 )

        return final_hits

    @staticmethod
    def _sigmoid( value ):
        value = max( -50.0, min(50.0,float(value), ),)
        return 1.0 / ( 1.0 + math.exp(-value) )

    @staticmethod
    def _metadata_match( hit, filt, ):
        if not filt:
            return True

        metadata = getattr( hit.chunk, "metadata", {}, ) or {}
        metadata = { str(k).lower(): v for k, v in metadata.items() }
        chunk = hit.chunk

        def normalize(value):
            if value is None:
                return ""
            return str( value ).strip().lower()

        text_fields = [ "document", "procedure", "section", "version", "department", ]

        for field in text_fields:
            expected = filt.get( field )
            if expected is None:
                continue
            actual = metadata.get(field)
            if ( actual is None and field == "section" ):
                actual = ( AdvancedRAGEngine._section(chunk ))
            if ( actual is None and field == "document" ):
                actual = ( AdvancedRAGEngine._source(chunk ) )
            if actual is None:
                return False
            if ( normalize(expected) not in normalize(actual) ):
                return False

        date_fields = [ "issue_date", "review_date", ]
        for field in date_fields:
            expected = filt.get( field )
            if expected is None:
                continue

            actual = metadata.get( field )
            if actual is None:
                return False

            if ( normalize(expected) not in normalize(actual) ):
                return False

        for field in [ "issue_year", "review_year", ]:
            expected = filt.get( field )
            if expected is None:
                continue

            actual = metadata.get( field )
            if actual is None: 
                date_field = field.replace( "_year", "_date",)
                actual_date = metadata.get( date_field )
                if actual_date:
                    years = re.findall( r"\b(19\d{2}|20\d{2})\b", str(actual_date), )
                    actual = ( years[0] if years else None )

            try:
                if int(actual) != int( expected ):
                    return False

            except ( TypeError, ValueError, ):
                return False

        if filt.get( "year_gt" ) is not None:
            threshold = int( filt["year_gt"] )
            candidate_years = []
            structured_fields = [
                "year",
                "publication_year",
                "issue_year",
                "review_year",
                "issue_date",
                "review_date", ]

            for key in structured_fields:
                value = metadata.get( key )
                if value is None:
                    continue

                candidate_years.extend( int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b",str(value), ))

            if not candidate_years:
                fallback_text = (
                    ( AdvancedRAGEngine._source(chunk ) or "" ) + " " + (chunk.text or "" ))

                candidate_years.extend( int(y) for y in re.findall( r"\b(19\d{2}|20\d{2})\b", fallback_text,))

            if not any( year > threshold for year in candidate_years ):
                return False

        return True

    def _rewrite( self, question, ):

        prompt = f"""
Rewrite this question into one precise standalone search query.
The user may be asking about a supplied manual/document.
Preserve all document-specific terms and the information need.
If the question is in English and the source documents may be Arabic, you may produce an Arabic-equivalent search query.
Do not invent a procedure name or fact.

Question:
{question}

JSON:
{{"query":"..."}}
"""

        obj = self._call_llm_with_retry( self.llm.generate_json, self.TRANSFORM_SYSTEM, prompt, 250, "rewriter", )
        return ( str( obj.get("query",question, ) ).strip() or question )

    def _multi_query( self, question,):
        prompt = f"""
Generate 3 diverse search queries for retrieving the answer from an
organizational manual.

Requirements:

1. Preserve exactly the user's information need.
2. Do not invent facts.
3. Do not invent metadata.
4. Use concise search-engine-style queries.
5. If the question is English and the document may be Arabic:
   - produce at least one English query,
   - produce at least one natural Arabic equivalent,
   - produce one terminology-focused query.
6. Prefer terminology likely to occur in a formal manual.
7. Preserve important concepts from the original question.
8. Do not answer the question.

Question:
{question}

Return JSON ONLY:
{{
  "queries": [
    "...",
    "...",
    "..."
  ]
}}
"""
        obj = self._call_llm_with_retry( self.llm.generate_json, self.TRANSFORM_SYSTEM, prompt, 350, "multi_query", )
        queries = obj.get( "queries", [], )
        if not isinstance( queries, list, ):
            return []

        return [ str(x).strip() for x in queries if str(x).strip() ][:3]

    def _decompose( self, question,):
        prompt = f"""
Decompose this multi-part question into independent searchable subquestions.
Each subquestion must represent one distinct information need.


Requirements:
- Preserve the meaning of the original question.
- Do not invent facts.
- Make every subquestion independently searchable.
- Keep important document terminology.
- If useful, use terminology likely to appear in a formal manual.

Question:
{question}

JSON:
{{"subquestions":["..."]}}
"""
        obj = self._call_llm_with_retry( self.llm.generate_json, self.TRANSFORM_SYSTEM, prompt, 400, "decomposition", )
        subquestions = obj.get( "subquestions", [],)
        if not isinstance( subquestions, list, ):
            return []
        return [ str(x).strip() for x in subquestions if str(x).strip() ][:5]

    def _hyde( self, question, ):
        prompt = f"""
Write a short hypothetical passage that could help retrieve the relevant document passage.
The passage is ONLY a retrieval aid.
It is NOT a factual answer.
Do not invent metadata.
Preserve the information need from the question.

Question:
{question}

JSON:
{{"passage":"..."}}
"""
        obj = self._call_llm_with_retry( self.llm.generate_json, self.TRANSFORM_SYSTEM, prompt, 400, "hyde", )
        return ( str(  obj.get("passage",question,  ) ).strip() or question )

    def _self_query( self, question, ):
        prompt = f"""
Extract:

1. A semantic search query containing ONLY the information need.
2. Metadata filters ONLY when explicitly stated by the user.

Supported metadata fields:
- document
- procedure
- section
- version
- issue_date
- issue_year
- review_date
- review_year
- department
- year_gt

Rules:
- Never invent metadata values.
- Only create a metadata filter when the user's question explicitly
  provides or clearly requests that constraint.
- If a metadata constraint is not present, use null.
- year_gt means "newer than the specified year".
- review_year means the document review year.
- issue_year means the document issue year.

Question:
{question}

Return JSON ONLY:
{{
  "semantic_query": "...",
  "filters": {{
    "document": null,
    "procedure": null,
    "section": null,
    "version": null,
    "issue_date": null,
    "issue_year": null,
    "review_date": null,
    "review_year": null,
    "department": null,
    "year_gt": null
  }}
}}
"""
        obj = self._call_llm_with_retry( self.llm.generate_json, self.TRANSFORM_SYSTEM, prompt, 400, "self_query", )
        semantic_query = str( obj.get("semantic_query",question, ) ).strip()
        if not semantic_query:
            semantic_query = question

        raw_filters = obj.get( "filters", {}, )
        if not isinstance( raw_filters, dict, ):
            raw_filters = {}
        allowed_fields = set( self.config.allowed_metadata_fields )
        filters = {}

        for key, value in raw_filters.items():
            if key not in allowed_fields:
                continue
            if value is None or value == "":
                continue
            if key in { "issue_year", "review_year", "year_gt", }:
                try:
                    filters[key] = int( value )
                except ( TypeError, ValueError, ):
                    continue
            else:
                filters[key] = str( value ).strip()

        return ( semantic_query, filters, )

    def _calculate_confidence( self, question, hits, ):
        if not hits:
            return {
                "confidence": 0.0,
                "label": "Very Low",
                "metric":
                "retrieval_quality_heuristic",
                "top_score": 0.0,
                "average_top_score": 0.0,
                "margin": 0.0,
                "agreement": 0.0,
                "formula_evidence": 0.0,
                "table_evidence": 0.0,
                "evidence_count": 0, }

        ce_features = []

        for h in hits:
            metadata = getattr( h.chunk, "metadata", {}, ) or {}
            value = metadata.get( "_rerank_ce_feature" )
            if value is not None:
                ce_features.append( float(value) )

        if ce_features:
            scores = ce_features
        else:
            scores = [ min( 1.0, max(0.0,float(h.score), ), ) for h in hits ]

        top_score = scores[0]
        top_count = min( 3, len(scores), )
        average_top_score = ( sum( scores[:top_count] ) / top_count )

        if len(scores) > 1:
            margin = max( 0.0, scores[0] - scores[1], )
        else:
            margin = scores[0]

        evidence_count = min( 5, len(hits), )
        agreement_scores = scores[ :min( 3, len(scores), )]
        if agreement_scores:
            agreement = ( sum( agreement_scores ) / len(agreement_scores ))
        else:
            agreement = 0.0

        table_count = sum( 1 for h in hits[:5] if h.chunk.chunk_type == "table" )
        table_evidence = min( 1.0, table_count / 3.0,)
        formula_terms = [ "=", "×", "÷", "معادلة", "حد الطلب", ]
        formula_count = sum( 1 for h in hits[:5] if any( term in h.chunk.text for term in formula_terms ))
        formula_evidence = min(  1.0, formula_count / 2.0, )
        margin_score = min( 1.0, margin * 5.0,)
        weights = (self.config.confidence_weights)
        confidence = ( weights["top_score_weight" ] * top_score + weights["avg_top_weight" ] * average_top_score + weights["margin_weight" ] * margin_score + weights["agreement_weight" ] * agreement )
        confidence = max( 0.0, min(1.0,confidence, ),)

        return {
            "confidence": round( confidence, 3, ),
            "label": self._confidence_label( confidence ),
            "metric": "retrieval_quality_heuristic",
            "top_score": round( top_score, 3, ),
            "average_top_score": round( average_top_score, 3, ),
            "margin": round( margin, 3, ),
            "agreement": round( agreement, 3, ),
            "formula_evidence": round(  formula_evidence,  3, ),
            "table_evidence": round( table_evidence, 3, ),
            "evidence_count": evidence_count, }

    @staticmethod
    def _confidence_label( confidence, ):
        if confidence >= 0.80:
            return "Very High"
        if confidence >= 0.65:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        if confidence >= 0.25:
            return "Low"
        return "Very Low"

    @staticmethod
    def _debug_print_chunks( hits, ):
        print( "\n" + "=" * 100 )
        print( "RETRIEVED CHUNKS")
        print( "\n" + "=" * 100 )

        for i, hit in enumerate( hits, start=1, ):
            chunk = hit.chunk
            metadata = getattr( chunk, "metadata", {}, ) or {}
            print( "\n" + "-" * 80 )
            print( f"CHUNK #{i}")
            print( f"Chunk ID : {chunk.chunk_id}" )
            print(  f"Type     : {chunk.chunk_type}" )
            print( f"Score    : {hit.score:.4f}" )
            print( "Page     :", AdvancedRAGEngine._page( chunk ), )
            print( "Section  :", AdvancedRAGEngine._section( chunk ), )
            print( "Source   :", AdvancedRAGEngine._source( chunk ), )

            if ( "_retrieval_query" in metadata ):
                print( "Query    :", metadata[ "_retrieval_query" ],)
            if ( "_decomposition_subquestion" in metadata ):
                print( "SubQ     :", metadata[ "_decomposition_subquestion" ], )
            if ( "_original_retrieval_score" in metadata):
                print( "Original :" f" {metadata['_original_retrieval_score']:.6f}" )
            if ( "_rerank_ce_logit" in metadata ):
                print( "CE logit :" f" {metadata['_rerank_ce_logit']:.4f}" )
            if ( "_rerank_phrase_bonus" in metadata ):
                print( "Phrase   :" f" {metadata['_rerank_phrase_bonus']:.4f}" )

            print( "-" * 80 )
            print( chunk.text)
        print( "\n" + "=" * 100 )
        print( "END RETRIEVED CHUNKS" )
        print( "\n" + "=" * 100 )

    @staticmethod
    def _compress( question, hits, ):
        out = []
        for h in hits:
            if ( h.chunk.chunk_type == "table" ):
                text = h.chunk.text
            else:
                metadata = getattr( h.chunk, "metadata", {},) or {}
                queries = [ question ]
                queries.extend( metadata.get( "_retrieval_queries", [], ))
                if metadata.get( "_decomposition_subquestion" ):
                    queries.append( metadata["_decomposition_subquestion"])

                query_word_sets = [ set( re.findall( r"\w+", str(q).lower(), ))for q in queries if str(q).strip()]
                sentences = [ s.strip() for s in re.split( r"(?<=[.!?؟])\s+|\n", h.chunk.text, ) if s.strip() ]
                indices_to_keep = set()

                for i, sentence in enumerate( sentences, ):
                    sentence_words = set( re.findall( r"\w+", sentence.lower(), ))
                    if any( sentence_words & qwords for qwords in query_word_sets if qwords ):
                        indices_to_keep.add( i )
                        if i > 0:
                            indices_to_keep.add( i - 1 )
                        if ( i + 1 < len(sentences) ):
                            indices_to_keep.add( i + 1 )

                if indices_to_keep:
                    text = " ".join( sentences[i] for i in sorted(indices_to_keep ) ).strip()
                else:
                    text = h.chunk.text
            clone_chunk = replace( h.chunk,text=text, )
            out.append( RetrievalResult( clone_chunk, h.score, h.retriever, ))
        return out

    def _crag( self, question, hits ):
        if not hits:
            return { "decision": "not_relevant",
                     "action": "rewrite",
                     "score": 0.0, }

        ce_values = []
        for h in hits[:5]:
            metadata = getattr( h.chunk, "metadata", {}, ) or {}
            value = metadata.get( "_rerank_ce_feature" )
            if value is not None:
                ce_values.append( float(value) )

        if ce_values:
            avg = ( sum(ce_values) / len(ce_values) )
            if avg < 0.40:
                return { "decision": "not_relevant",
                         "action": "rewrite",
                         "score": round( avg, 3, ),
                         "metric": "cross_encoder_feature",}

            if avg < 0.55:
                return { "decision": "uncertain",
                         "action": "rewrite",
                         "score": round( avg, 3,),
                         "metric": "cross_encoder_feature", }

            return { "decision": "relevant",
                     "action": "keep",
                     "score": round( avg, 3, ),
                     "metric": "cross_encoder_feature", }

        q = set( re.findall( r"\w+", question.lower(),))
        vals = []
        for h in hits[:5]:
            words = set( re.findall( r"\w+", h.chunk.text.lower(),))
            vals.append( len(q & words ) / max(1,len(q), ))

        avg = ( sum(vals) / len(vals) )
        if avg < 0.05:
            return {"decision": "not_relevant",
                     "action": "rewrite",
                    "score": round( avg, 3, ),
                    "metric": "lexical_fallback",}

        if avg < 0.15:

            return {"decision": "uncertain",
                    "action": "rewrite",
                    "score": round( avg, 3,),
                    "metric": "lexical_fallback", }

        return { "decision": "relevant",
                 "action": "keep",
                 "score": round( avg, 3, ),
                 "metric": "lexical_fallback",}