# 🧬 Financial RLHF Studio

**A Context-Aware Human-in-the-Loop (HITL) Interface for Financial Domain Adaptation.**

Institutional finance requires higher accuracy than generic LLMs can provide. This repository implements a **Direct Preference Optimization (DPO)** workflow, allowing domain experts (Analysts/PMs) to correct AI hallucinations on specific financial documents and build "Gold Standard" datasets for fine-tuning.

![Status](https://img.shields.io/badge/Status-Active_Development-green) ![Stack](https://img.shields.io/badge/Stack-Streamlit_|_OpenAI_|_DPO-blue)

## 🏗️ Architecture

This is not just a text editor; it is a **Data Engine** designed to capture institutional nuance.

1.  **RAG-Augmented Generation:** The AI (GPT-4o-mini) reads a specific uploaded PDF (10-K, Research Note) to generate an initial analysis.
2.  **Expert Annotation:** The user corrects the output using a side-by-side comparison interface.
3.  **Visual Diff Tracking:** Real-time visualization of added/removed text to highlight the "Alpha" in the correction.
4.  **Structured Taxonomy:** Errors are tagged (e.g., *Hallucination*, *GAAP Mixup*, *Tone Issue*) to create metadata-rich training sets.

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/financial-rlhf-studio.git](https://github.com/YOUR_USERNAME/financial-rlhf-studio.git)
cd financial-rlhf-studio
pip install -r requirements.txt