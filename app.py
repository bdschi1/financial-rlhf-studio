import streamlit as st
import time
import json
import os
from datetime import datetime

# --- Imports for Logic ---
try:
    import pypdf
except ImportError:
    st.error("⚠️ Please install pypdf: `pip install pypdf`")

try:
    from openai import OpenAI
except ImportError:
    st.error("⚠️ Please install openai: `pip install openai`")

try:
    from dotenv import load_dotenv
    # FORCE OVERRIDE: This ignores the terminal's cache and reads the .env file fresh every time
    load_dotenv(override=True) 
except ImportError:
    pass 

# --- 1. Page Configuration (Tight & Wide) ---
st.set_page_config(
    page_title="Financial RLHF Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS Styling ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        .stButton button { width: 100%; border-radius: 8px; font-weight: bold;}
        
        /* Custom Header Styling */
        .custom-header {
            background: linear-gradient(90deg, #0e1117 0%, #262730 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #444;
            margin-bottom: 1rem;
            color: white;
        }
        .custom-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            background: -webkit-linear-gradient(eee, #999);
            -webkit-background-clip: text;
        }
        .custom-header p {
            margin: 0;
            font-size: 1rem;
            color: #aaa;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. Session State & Helper Functions ---
if "dataset" not in st.session_state:
    st.session_state.dataset = []
if "ai_draft" not in st.session_state:
    st.session_state.ai_draft = ""

def extract_text_from_pdf(file_obj):
    """Helper to extract text from the uploaded PDF object."""
    try:
        reader = pypdf.PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def generate_draft(context, user_prompt):
    """Call OpenAI or Fallback to Simulation."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    
    if not api_key:
        return (
            f"⚠️ SYSTEM: No API Key found in .env or environment.\n\n"
            f"We successfully read the file.\n"
            f"--- START CONTEXT ---\n{context[:1000]}...\n--- END CONTEXT ---\n\n"
            f"(To get real AI analysis, add OPENAI_API_KEY to your .env file and restart)"
        )
    
    # DEBUG: Show user which key is active (last 4 digits only)
    if len(api_key) > 4:
        st.toast(f"🔑 Authenticating with key ending in: ...{api_key[-4:]}", icon="🔒")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a junior financial analyst. Use the provided context to answer the user prompt."},
                {"role": "user", "content": f"Context: {context[:15000]}... \n\n Instruction: {user_prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

# --- 4. Sidebar (Data & Config) ---
with st.sidebar:
    st.header("🧬 Data Management")
    st.metric("Pairs Collected", len(st.session_state.dataset))
    
    if st.button("Download .JSONL"):
        json_str = "\n".join([json.dumps(x) for x in st.session_state.dataset])
        st.download_button(
            label="Save File",
            data=json_str,
            file_name="dpo_dataset.jsonl",
            mime="application/json"
        )
    st.divider()
    
    st.markdown("### 🛠️ Instructions")
    st.info("1. Upload PDF\n2. Generate Draft\n3. Edit to 'Gold Standard'\n4. Tag & Save")
    
    st.divider()
    
    # --- TECH STACK SECTION ---
    st.markdown("### 🏗️ Tech Stack")
    st.markdown("""
    ![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)
    ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
    ![OpenAI](https://img.shields.io/badge/Model-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
    ![DPO](https://img.shields.io/badge/Method-DPO%20%2F%20RLHF-orange?style=flat)
    ![Pandas](https://img.shields.io/badge/Data-Pandas%20%2F%20JSONL-150458?style=flat&logo=pandas&logoColor=white)
    """)

# --- 5. Main Layout ---

# Custom HTML Header
st.markdown("""
<div class="custom-header">
    <h1>🧬 Financial RLHF Studio</h1>
    <p>Direct Preference Optimization (DPO) Data Engine | Institutional Grade Alignment</p>
</div>
""", unsafe_allow_html=True)

# --- ROW 1: Context & Translation ---
doc_col1, doc_col2 = st.columns(2)

with doc_col1:
    with st.expander("ℹ️ System Overview (DPO)", expanded=False):
        st.markdown("""
        **Objective:** Create a Direct Preference Optimization (DPO) dataset.
        * **Rejected:** Baseline Model Output (Generic).
        * **Chosen:** Expert Human Revision (Specific).
        
        This aligns the model with institutional standards by capturing corrections in real-time.
        """)

with doc_col2:
    with st.expander("💼 Buy-Side Translation (The 'Why')", expanded=False):
        st.markdown("""
        <div style="font-size: 0.9em; font-style: italic; color: #444;">
        <strong>Automated Analyst Training Program</strong>
        <br>
        1. <strong>The Intern (AI)</strong> writes a draft.
        2. <strong>You (The PM)</strong> use the Red Pen to correct tone/nuance.
        3. The system learns from your edits to automate the grunt work next time.
        </div>
        """, unsafe_allow_html=True)

# --- ROW 2: Control Center ---
with st.container(border=True):
    st.markdown("#### 🎛️ Control Panel")
    
    input_c1, input_c2 = st.columns([1, 2])
    
    with input_c1:
        uploaded_file = st.file_uploader("Source Context (PDF)", type="pdf", label_visibility="collapsed")
        if not uploaded_file:
            st.caption("📂 1. Upload Document")
        else:
            st.caption(f"✅ {uploaded_file.name}")

    with input_c2:
        # Layout: Prompt takes 85%, Button takes 15%
        p_col, b_col = st.columns([6, 1])
        
        with p_col:
            prompt = st.text_input("Prompt", value="Summarize key risks in the MD&A.", label_visibility="collapsed")
        
        with b_col:
            # Icon-only button to prevent wrapping
            if st.button("⚡", type="primary", use_container_width=True, help="Run Generation"):
                if not uploaded_file:
                    st.error("Please upload a PDF first.")
                else:
                    with st.spinner("Reading & Generating..."):
                        # 1. Read PDF
                        raw_text = extract_text_from_pdf(uploaded_file)
                        # 2. Call Logic
                        st.session_state.ai_draft = generate_draft(raw_text, prompt)
            
            # Label below button
            st.markdown("<div style='text-align: center; font-size: 0.75em; margin-top: -5px; color: #666;'>Generate</div>", unsafe_allow_html=True)

# --- ROW 3: The Workbench ---
if st.session_state.ai_draft:
    with st.container(border=True):
        st.markdown("#### ✍️ DPO Workbench")
        
        # Editors (Side by Side)
        edit_c1, edit_c2 = st.columns(2)
        
        with edit_c1:
            st.markdown("**🤖 Rejected (AI Draft)**")
            draft_text = st.text_area("Rejected", value=st.session_state.ai_draft, height=200, label_visibility="collapsed", disabled=True)
            
        with edit_c2:
            st.markdown("**🧑‍🏫 Chosen (Expert Rewrite)**")
            expert_text = st.text_area("Chosen", value=st.session_state.ai_draft, height=200, label_visibility="collapsed")

        st.divider()
        
        # Actions
        meta_c1, meta_c2 = st.columns([3, 1])
        
        with meta_c1:
            tags = st.multiselect(
                "Error Taxonomy",
                ["Hallucination", "Missed Nuance", "Incorrect Tone", "Math Error", "GAAP vs Non-GAAP", "Outdated Info"],
                placeholder="Select error types..."
            )
            
        with meta_c2:
            st.write("") # Formatting spacer
            st.write("")
            if st.button("💾 Commit Pair", use_container_width=True):
                if draft_text == expert_text:
                    st.error("No changes detected.")
                else:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "chosen": expert_text,
                        "rejected": draft_text,
                        "tags": tags,
                        "source": uploaded_file.name if uploaded_file else "Manual"
                    }
                    st.session_state.dataset.append(entry)
                    st.toast(f"Saved! Total pairs: {len(st.session_state.dataset)}", icon="✅")

else:
    st.info("👆 Upload a document and click the ⚡ button to begin.")

# --- Footer ---
st.caption("v0.3 | Financial RLHF Studio")