import streamlit as st
import fitz  # pymupdf
import pandas as pd
from datetime import datetime
import json
import os
import difflib
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIG & SETUP ---
load_dotenv(override=True)
st.set_page_config(page_title="Financial RLHF Studio Pro", layout="wide", page_icon="🧬")

DATA_FILE = "data/dataset.jsonl"
if not os.path.exists("data"):
    os.makedirs("data")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- HELPER FUNCTIONS ---

def generate_draft(context_text, prompt):
    """Generates a response based on the PDF context + User Prompt"""
    full_prompt = f"Context from Document:\n{context_text[:20000]}...\n\nTask: {prompt}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use a smaller model to catch more errors
            messages=[
                {"role": "system", "content": "You are a junior financial analyst. Be concise."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def save_entry(entry):
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def highlight_diff(text1, text2):
    """Calculates the difference and returns HTML for visual rendering"""
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))
    html = []
    for line in diff:
        if line.startswith('+ '):
            html.append(f'<span style="background-color:#e6ffec; color:#155724;">{line}</span>')
        elif line.startswith('- '):
            html.append(f'<span style="background-color:#f8d7da; color:#721c24; text-decoration: line-through;">{line}</span>')
        elif line.startswith('? '):
            continue
        else:
            html.append(line)
    return "<br>".join(html)

# --- SIDEBAR: PDF & STATS ---
with st.sidebar:
    st.header("1. Source Document")
    uploaded_file = st.file_uploader("Upload 10-K / Research Note", type="pdf")
    
    pdf_text = ""
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        st.success(f"Loaded {len(pdf_text)} chars")
        with st.expander("📄 View Extracted Text"):
            st.text(pdf_text[:1000] + "...")
    
    st.divider()
    
    # Dataset Stats
    if os.path.exists(DATA_FILE):
        count = sum(1 for _ in open(DATA_FILE))
    else:
        count = 0
    st.metric("🧬 Training Pairs", count)

# --- MAIN WORKSPACE ---
st.title("Financial RLHF Studio 🔬")
st.caption("Context-Aware Fine-Tuning Environment")

# SECTION 1: TASK DEFINITION
col_task, col_go = st.columns([4, 1])
with col_task:
    prompt_input = st.text_input("Analyst Instruction", placeholder="e.g., Summarize the legal risks mentioned in this document.")
with col_go:
    st.write("") 
    if st.button("🚀 Generate Draft", type="primary", use_container_width=True):
        if not pdf_text:
            st.error("Please upload a PDF first.")
        elif not prompt_input:
            st.error("Please enter an instruction.")
        else:
            with st.spinner("AI is thinking..."):
                draft = generate_draft(pdf_text, prompt_input)
                st.session_state['draft'] = draft
                st.session_state['prompt'] = prompt_input
                st.session_state['source'] = uploaded_file.name

# SECTION 2: THE ANNOTATION INTERFACE
if 'draft' in st.session_state:
    st.divider()
    
    # Two-Column Layout
    left, right = st.columns(2)
    
    with left:
        st.subheader("🤖 AI Draft (Rejected)")
        st.info("Original Model Output")
        st.text_area("Read-Only", value=st.session_state['draft'], height=400, disabled=True, key="ai_out")
        
    with right:
        st.subheader("👨‍💼 Expert Correction (Chosen)")
        st.success("Your Gold-Standard Edit")
        corrected_text = st.text_area("Edit Here", value=st.session_state['draft'], height=400, key="human_out")

    # SECTION 3: METADATA & DIFF
    st.divider()
    meta_col1, meta_col2 = st.columns([2, 1])
    
    with meta_col1:
        st.markdown("#### 🔍 Change Visualization")
        # Visual Diff Logic (Simple Comparison)
        if corrected_text != st.session_state['draft']:
            st.caption("Red = Deleted, Green = Added")
            # We skip the complex HTML renderer for brevity, but you'd inject the 'highlight_diff' logic here
            # For now, we show a simplified delta
            len_diff = len(corrected_text) - len(st.session_state['draft'])
            st.metric("Character Delta", f"{len_diff:+}", delta_color="normal")
        else:
            st.caption("No changes made yet.")

    with meta_col2:
        st.markdown("#### 🏷️ Error Taxonomy")
        errors = st.multiselect("Flag AI Errors", 
            ["Hallucination", "Math Error", "Missed Nuance", "Tone Issue", "Look-ahead Bias", "GAAP/Non-GAAP Mixup"])
        
        difficulty = st.select_slider("Correction Difficulty", options=["Trivial", "Moderate", "Deep Rewriting"])
        
        if st.button("💾 Save Training Pair", type="primary", use_container_width=True):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "source_doc": st.session_state.get('source', 'unknown'),
                "prompt": st.session_state['prompt'],
                "rejected": st.session_state['draft'],
                "chosen": corrected_text,
                "metadata": {
                    "errors": errors,
                    "difficulty": difficulty,
                    "annotator": "Expert_User"
                }
            }
            save_entry(entry)
            st.balloons()
            st.success("Data Point Saved to `data/dataset.jsonl`")