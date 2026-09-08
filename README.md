# Advanced_RAG_For_Arabic

This version implements the supplied Advanced RAG practical task:
routing, Basic RAG baseline, conditional query transformation, retrieval
improvement, final grounded generation, per-question evaluation, and cost /
latency tracking. The task requires the router to avoid forcing every question
through every technique, so the selected path is recorded for each question.

## 1. Install

```bash
pip install -r requirements.txt
```

For OCR of scanned Arabic/English PDFs, install Tesseract OCR separately and
make sure the `tesseract` executable is on PATH. If Arabic language data is
not installed, set `TESSERACT_LANG=eng` or install the Arabic `ara` language
data.

## 2. Free Gemini

Create a Gemini API key in Google AI Studio and set:

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Linux/macOS:
```bash
export GEMINI_API_KEY="YOUR_KEY"
```

The default model is `gemini-2.5-flash-lite`. Change it with
`RAG_LLM_MODEL`. The Gemini client records input/output tokens and latency.
Cost is recorded as `$0.00` for free-tier calls.

## 3. Ingest

```bash
python main.py ingest ./sample_docs
```

PDF parsing uses `pdfplumber` first. When native PDF extraction is empty or
low quality, the parser renders the page and uses Tesseract OCR. OCR table
rows are preserved as table chunks rather than flattened into prose.

Disable OCR if required:

```powershell
$env:OCR_ENABLED="false"
```

## 4. Ask

```bash
python -m streamlit run app.py
```

The response prints:
- route and reason
- techniques actually executed
- answer and source/chunk identifiers
- every LLM call's input/output tokens, cost and latency
- intermediate query transformations and CRAG/compression information

## 5. Run the required evaluation set

```bash
python main.py evaluate
```

This runs both:
1. the conditional Advanced RAG pipeline
2. the Basic RAG baseline

It writes JSON and CSV result files, including one row per question with:
Context Relevance, Faithfulness, Answer Relevance, Correctness, total cost,
latency, route, and techniques.

The evaluation judge uses one consistent rubric. If the Gemini judge cannot be
called, deterministic estimates are used.

## Advanced techniques implemented

- Router: `simple | basic_rag | advanced_rag`
- Rewriting
- Multi-Query
- Decomposition
- HyDE
- Self-Query with a year filter
- Hybrid retrieval already present in the project
- Reranking
- Contextual Compression
- CRAG-style retrieval correction
- Grounded final Gemini generation
- Per-question evaluation and cost tracking

The advanced techniques are conditional. For example, decomposition is useful
for a complex multi-part question but adds unnecessary LLM calls to a simple
question.

