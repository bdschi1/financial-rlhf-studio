# Dataset Card: Financial-CoT-Preferences

## Dataset Summary
This dataset contains expert-annotated pairs of (Rejected, Chosen) responses focused on financial analysis tasks. It is designed to train Reward Models (RM) for Direct Preference Optimization (DPO).

## Annotation Process
* **Annotators:** Domain experts with institutional finance backgrounds.
* **Source Documents:** Publicly available SEC filings (10-K, 10-Q), Earnings Call Transcripts, and Sell-side Research Notes.
* **Methodology:**
    1.  **Context-Aware Generation:** An LLM generates a draft based on a provided PDF.
    2.  **Expert Review:** An expert rewrites the draft to correct hallucinations, tone, or missing financial nuance.
    3.  **Tagging:** Errors are classified into a structured taxonomy (see below).

## Error Taxonomy
We track the following error types to ensure balanced training:
* **Hallucination:** Fabricated numbers or events not in the source text.
* **Math Error:** Incorrect calculation of margins, growth rates, or valuation multiples.
* **Missed Nuance:** Factually correct but misses the "so what?" (e.g., mentioning revenue growth without noting it was driven by FX).
* **Tone Issue:** Overly promotional or overly cautious language unsuited for institutional memos.

## Intended Use
* Fine-tuning Llama-3 or Mistral models for financial summarization.
* Benchmarking "Generalist" models against "Specialist" financial models.