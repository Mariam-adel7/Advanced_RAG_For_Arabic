"""
Examples
--------
set GEMINI_API_KEY=...
python main.py ingest ./sample_docs
python main.py ask "What is RAG?"
python main.py ask "What is the reorder point formula?"
python main.py evaluate
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from dataclasses import asdict
from config import DEFAULT_CONFIG, Config
from factory import build_ingestion_pipeline, build_advanced_rag_engine
from ingestion import Embedder, VectorStore
from evaluation.evaluator import Evaluator, save_results, save_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag.main")

QUESTIONS = [
    ("Q1", "What is Retrieval-Augmented Generation (RAG)?"),
    ("Q2", "What are the main limitations of RAG?"),
    ("Q3", "Why is it bad?"),
    ("Q4", "What are the challenges and failure modes of RAG?"),
    ("Q5", "Compare RAG and fine-tuning, explain their advantages and disadvantages, and state when each should be used."),
    ("Q6", "How does RAG reduce hallucination?"),
    ("Q7", "Find documents about RAG published after 2024."),
    ("Q8", "What is reranking and why is it useful in RAG?"),
    ("Q9", "Which approach should be used for a simple question versus a complex multi-part question?"),
    ("Q10", "What is the procedure for this?"),
]

def _load_engine(config: Config):
    embedder = Embedder(model_name=config.embedding_model)
    vector_store = VectorStore.load(config.store_path)
    return build_advanced_rag_engine(config, vector_store, embedder)

def cmd_ingest(args, config):
    pipeline = build_ingestion_pipeline(config)
    chunks = pipeline.ingest_directory(args.path, recursive=not args.no_recursive)
    pipeline.vector_store.save(config.store_path)
    print(f"Ingested {len(chunks)} chunks -> {config.store_path}")

def _print_result(result):
    print(f"\nQ: {result.question}\n")
    print(f"Route: {result.route.route}")
    print(f"Reason: {result.route.reason}")
    print(f"Techniques: {', '.join(result.techniques_used) or 'none'}")
    print(f"\nA: {result.answer}\n")
    if result.sources:
        print("Sources:")
        for src in result.sources:
            loc = f" p.{src.page}" if src.page else ""
            print(f"  [Source {src.index}] {src.source}{loc} ({src.chunk_type}, score={src.score:.3f})")
    print("\nLLM calls:")
    for c in result.calls:
        print(f"  {c['step']}: {c['input_tokens']} in / {c['output_tokens']} out "
              f"/ ${c['cost']:.6f} / {c['latency']:.2f}s")
    if result.intermediate:
        print("\nIntermediate:")
        print(json.dumps(result.intermediate, ensure_ascii=False, indent=2))

def cmd_ask(args, config):
    engine = _load_engine(config)
    result = engine.answer(args.question)
    _print_result(result)

def cmd_chat(args, config):
    engine = _load_engine(config)
    history=[]
    print("Advanced RAG chat — type 'exit' to quit.")
    while True:
        q=input("\nYou: ").strip()
        if q.lower() in {"exit","quit"}: break
        if not q: continue
        r=engine.answer(q, chat_history=history)
        print(f"\nAssistant: {r.answer}")
        history += [{"role":"user","content":q},{"role":"assistant","content":r.answer}]

def cmd_evaluate(args, config):
    engine = _load_engine(config)
    evaluator = Evaluator(engine.llm, use_llm_judge=config.enable_llm_judge)
    advanced_records=[]
    basic_records=[]
    for qid, question in QUESTIONS:
        print(f"\n=== {qid} === {question}")
        result=engine.answer(question)
        rec=evaluator.evaluate(qid,result)
        advanced_records.append(rec)
        print(f"Advanced: {rec.route} | {rec.techniques} | "
              f"CR={rec.context_relevance} F={rec.faithfulness} "
              f"AR={rec.answer_relevance} C={rec.correctness} "
              f"cost=${rec.total_cost:.6f} latency={rec.latency:.2f}s")
        result_b=engine.basic_answer(question)
        rec_b=evaluator.evaluate(qid,result_b)
        basic_records.append(rec_b)
        print(f"Basic:    CR={rec_b.context_relevance} F={rec_b.faithfulness} "
              f"AR={rec_b.answer_relevance} C={rec_b.correctness} "
              f"cost=${rec_b.total_cost:.6f} latency={rec_b.latency:.2f}s")

    adv_path=config.evaluation_output
    basic_path=os.path.splitext(adv_path)[0]+"_basic.json"
    adv_csv=os.path.splitext(adv_path)[0]+".csv"
    basic_csv=os.path.splitext(adv_path)[0]+"_basic.csv"
    save_results(advanced_records,adv_path)
    save_results(basic_records,basic_path)
    save_csv(advanced_records,adv_csv)
    save_csv(basic_records,basic_csv)

    print("\n=== Required Results Table ===")
    print("ID | Route | Techniques | Context Rel. | Faithfulness | Answer Rel. | Correctness | Cost | Latency")
    for r in advanced_records:
        print(f"{r.id} | {r.route} | {','.join(r.techniques) or '-'} | "
              f"{r.context_relevance:.2f} | {r.faithfulness:.2f} | {r.answer_relevance:.2f} | "
              f"{r.correctness:.2f} | ${r.total_cost:.6f} | {r.latency:.2f}s")
    print(f"\nSaved: {adv_path}")
    print(f"Saved baseline: {basic_path}")

def main():
    parser=argparse.ArgumentParser(description="Advanced RAG with routing, retrieval improvement, OCR and Gemini")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("ingest"); p.add_argument("path"); p.add_argument("--no-recursive",action="store_true"); p.set_defaults(func=cmd_ingest)
    p=sub.add_parser("ask"); p.add_argument("question"); p.set_defaults(func=cmd_ask)
    p=sub.add_parser("chat"); p.set_defaults(func=cmd_chat)
    p=sub.add_parser("evaluate"); p.set_defaults(func=cmd_evaluate)
    args=parser.parse_args()
    args.func(args,DEFAULT_CONFIG)

if __name__=="__main__":
    main()
