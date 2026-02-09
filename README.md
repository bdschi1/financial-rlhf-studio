# 🧬 Financial RLHF Studio

### A Context-Aware Human-in-the-Loop (HITL) Interface for Financial Domain Adaptation.

Institutional finance requires higher accuracy than generic LLMs can provide. This repository implements a **Direct Preference Optimization (DPO)** workflow, transforming subjective domain expertise into structured training data. It allows Analysts and PMs to correct AI hallucinations on specific financial documents, building "Gold Standard" datasets for fine-tuning.

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Stack](https://img.shields.io/badge/Stack-Streamlit_|_Multi--Provider_|_DPO-blue)
![Anthropic](https://img.shields.io/badge/Model-Claude-191919?style=flat&logo=anthropic&logoColor=white)
![Gemini](https://img.shields.io/badge/Model-Gemini-4285F4?style=flat&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/Model-GPT--4o-412991?style=flat&logo=openai&logoColor=white)

---

## 🏗️ Architecture

**Data Engine** designed to capture institutional nuance. PDF ingestion pipeline: section detection → boilerplate stripping → whitespace normalization → auto-chunking. Two annotation modes:

### Section-Aware Document Ingestion
- Uploads any PDF; auto-detects 10-K/10-Q section structure (Item 1, 1A, 7, 7A, 8, etc.)
- Uses pymupdf (fitz) for layout-aware extraction; tries embedded TOC first, then regex fallback
- **Structured filings:** Section table with char counts and budget indicators; Item 7 (MD&A) auto-selected
- **Unstructured docs** (earnings calls, broker notes): page-range slider (defaults to pages 1–10)
- Configurable context budget (15k–800k chars) with truncation warnings

### Boilerplate Filtering
10-K filings contain significant regulatory boilerplate that wastes tokens and dilutes signal. The parser automatically strips:
- **Cover pages** — SEC header, form type, CIK, checkbox boilerplate ("Indicate by check mark...")
- **Forward-looking statements disclaimers** — Private Securities Litigation Reform Act language
- **Signature blocks** — "Pursuant to the requirements of Section 13..."
- **Page headers/footers** — Auto-detected via frequency analysis (lines appearing on 30%+ of pages)
- **SOX certifications** — Section 302/906 boilerplate (Exhibits 31/32)
- **Exhibit index entries** — "Filed herewith", "Incorporated by reference"
- **XBRL/iXBRL artifacts** — Tag remnants from EDGAR markup
- **TOC dotted-leader lines** — "Item 7 ............. 42"

Cleanup stats are shown in the UI after parsing (e.g., "Stripped 45,000 boilerplate (12%) + 28,000 whitespace (7%) = 73,000 total chars removed (19% of raw text)").

### Whitespace Normalization
PDF text extraction (pymupdf `get_text("text")`) reconstructs spatial layout using whitespace, producing significant char bloat that wastes tokens. Four safe, content-preserving normalizations run automatically after boilerplate stripping:

| Transform | What it does | Savings |
|-----------|-------------|---------|
| **Trailing whitespace** | Strips trailing spaces/tabs from every line | 3-6% on 10-Ks |
| **Inline multi-space** | Collapses 2+ spaces after non-space chars to single space (preserves indentation) | 8-12% on 10-Ks |
| **Horizontal rules** | Collapses `-----`, `=====`, `_____` (5+ chars) to short `---` marker | 0.5-1% on 10-Ks |
| **Excess blank lines** | Collapses 3+ consecutive newlines to double newline | 1-3% on 10-Ks |

Combined savings: **~13-22% on a 600k-char 10-K MD&A** (~20k-33k tokens reclaimed). Earnings transcripts see ~4-8% since they originate from cleaner speech-to-text pipelines.

### Auto-Chunking
Sections that exceed the context budget are automatically chunked using paragraph-aware splitting (same pattern as the [chunkers](https://github.com/bdschi1/chunkers) project):
- Breaks at paragraph boundaries (`\n\n`), never mid-sentence
- 500-char overlap between chunks for context continuity
- Chunk navigator UI (Prev/Next + dropdown) for browsing chunks
- Multi-section support: small sections stay whole, oversized ones are split

### Single Pair Mode
1.  **Section-Targeted Generation:** Upload a PDF, select the relevant section (e.g., MD&A), pick any model from the dropdown (Claude/Gemini/GPT-4o), and generate an initial analysis.
2.  **Expert Annotation:** Correct the output using a side-by-side comparison interface.
3.  **Structured Taxonomy:** Errors are tagged (e.g., *Hallucination*, *GAAP Mixup*, *Tone Issue*).
4.  **Output:** One chosen/rejected preference pair per annotation.

### K-Output Ranking Mode
1.  **Multi-Output Generation:** Generate K outputs (2-9) for the same prompt using configurable presets (temperature sweep, model comparison, cross-provider, persona sweep, or fully custom).
2.  **Expert Review:** Expanded output viewers with a scratchpad for note-taking while comparing outputs side by side.
3.  **Expert Ranking:** Drag-and-drop or number-rank all K outputs from best to worst.
4.  **Pairwise Extraction:** Automatically derives **K(K-1)/2** preference pairs from a single ranking.
5.  **Output:** Up to 36 preference pairs from one annotation session (K=9).

> **Efficiency Insight:** When an expert ranks K=4 outputs as A > B > C > D, this implicitly produces 6 pairwise comparisons (A>B, A>C, A>D, B>C, B>D, C>D) — the same training signal that would require 6 separate binary annotation sessions.

---

## 🔌 Multi-Provider Support

The generator routes to the correct API based on model name:

| Provider | Models | Env Variable |
|----------|--------|-------------|
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-haiku-4-20250414`, `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **Google** | `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-pro` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` | `OPENAI_API_KEY` |

Cross-provider presets let you rank outputs from different providers against each other in a single session.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/bdschi1/financial-rlhf-studio.git
cd financial-rlhf-studio
pip install -r requirements.txt
```

### 2. API Keys

Set at least one provider key in your `.env` file:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
financial-rlhf-studio/
├── app.py                  # Streamlit UI — tabs for Single Pair + K-Ranking
├── src/
│   ├── configs.py          # GenerationConfig dataclass, presets, model registry
│   ├── document.py         # PDF parsing, section detection, boilerplate filter, whitespace normalization, chunking
│   ├── generator.py        # Multi-provider generation (Anthropic, Gemini, OpenAI)
│   ├── ranker.py           # K-output orchestration + pairwise extraction
│   └── storage.py          # JSONL persistence utilities
├── data/                   # Output JSONL files (gitignored)
├── requirements.txt
├── DATA_CARD.md            # Dataset specification
└── .env                    # API keys (gitignored)
```

---

## 📊 JSONL Output Schema

### Single Pair Entry
```json
{
    "timestamp": "2026-02-08T14:30:00",
    "prompt": "Summarize key risks in the MD&A.",
    "chosen": "Expert-corrected text...",
    "rejected": "AI draft text...",
    "tags": ["Hallucination", "Missed Nuance"],
    "source": "10K_ACME_2025.pdf",
    "mode": "single_pair"
}
```

### Ranking-Derived Pair Entry
```json
{
    "timestamp": "2026-02-08T14:30:00",
    "prompt": "Summarize key risks in the MD&A.",
    "chosen": "Higher-ranked output text...",
    "rejected": "Lower-ranked output text...",
    "tags": ["Incorrect Tone"],
    "source": "10K_ACME_2025.pdf",
    "mode": "ranking",
    "ranking_metadata": {
        "session_id": "uuid",
        "total_k": 4,
        "chosen_rank": 1,
        "rejected_rank": 3,
        "rank_margin": 2,
        "chosen_config": {"label": "A", "model": "claude-sonnet-4-20250514", "provider": "anthropic", "temperature": 0.7},
        "rejected_config": {"label": "C", "model": "gpt-4o-mini", "provider": "openai", "temperature": 0.7},
        "full_ranking": ["A", "B", "C", "D"]
    }
}
```

The `rank_margin` field enables downstream reward model training to weight high-confidence pairs (large margin) more heavily than adjacent-rank pairs.

---

## 🔧 Ranking Mode Presets

Presets are organized into categories, selectable via horizontal radio buttons:

| Category | Preset | K | Pairs | What it compares |
|----------|--------|---|-------|-----------------|
| **Model Comparison** | Anthropic (3 models) | 3 | 3 | Sonnet 4 vs Haiku 4 vs Sonnet 3.5 |
| | Gemini (4 models) | 4 | 6 | 2.0 Flash vs Flash-Lite vs 1.5 Flash vs 1.5 Pro |
| | OpenAI (3 models) | 3 | 3 | GPT-4o-mini vs GPT-4o vs GPT-4-turbo |
| | Cross-Provider | 3 | 3 | Claude vs Gemini vs GPT-4o-mini |
| **Temperature Sweep** | 4 outputs | 4 | 6 | Same model at t=0.2, 0.5, 0.8, 1.0 |
| **Persona Sweep** | 4 outputs | 4 | 6 | Junior Analyst vs Senior PM vs Risk vs Quant |
| **Custom** | — | 2-9 | 1-36 | Mix any models, temps, and personas |

---

## 📄 Supported Document Types

| Document | Section Detection | Fallback |
|----------|------------------|----------|
| **10-K** (full annual filing) | Item 1, 1A, 7, 7A, 8, etc. auto-detected | Page range |
| **10-Q** (quarterly filing) | Shorter Item set detected | Page range |
| **Earnings transcript** | No items detected | Pages 1–10 default |
| **Broker research note** | No items detected | Pages 1–10 default |
| **Any other PDF** | Auto-detect attempted | Page range slider |

---

## **A note on 'context budget':**

The context budget controls how many characters of the document are sent to the LLM as input context. In practice:

**What it does:**
- Sets a cap on the text extracted from your PDF before it's sent to the model
- `60k` = 60,000 characters ≈ 15,000 tokens ≈ ~12,000 words

**Why it matters:**
- Every character costs tokens (roughly 4 chars ≈ 1 token)
- Larger budgets = more document context but higher cost and slower generation
- Smaller budgets = cheaper/faster but the model sees less of the document

**The presets map to model limits:**

| Budget | Tokens (~) | Good for |
|--------|-----------|----------|
| 15k | ~3.7k | Single small section |
| 60k (default) | ~15k | Most sections, balanced cost/coverage |
| 200k | ~50k | Very large sections like full MD&A |
| 400k | ~100k | GPT-4o's practical ceiling |
| 800k | ~200k | Claude's max window |

**Hypothetical document:**
An earnings transcript is 40,297 chars total — it fits entirely within the 60k default budget. No content lost. If you had a 633k-char MD&A section from a 10-K, the system would auto-chunk it into multiple pieces at the 60k budget --> navigate between chunks.

