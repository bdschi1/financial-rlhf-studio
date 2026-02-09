import streamlit as st
import json
import os
import uuid
from datetime import datetime

# --- Imports for Logic ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    from streamlit_sortables import sort_items
except ImportError:
    sort_items = None

from src.configs import (
    GenerationConfig, PRESETS, PRESET_CATEGORIES, AVAILABLE_MODELS, ALL_MODELS,
    DEFAULT_SYSTEM_PROMPTS, provider_for_model,
)
from src.ranker import generate_k_outputs, extract_pairwise_preferences, count_pairs
from src.generator import generate_draft
from src.document import parse_document, assemble_context, get_section_summary, chunk_context

# --- 1. Page Configuration ---
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

# --- 3. Session State ---
if "dataset" not in st.session_state:
    st.session_state.dataset = []
if "ai_draft" not in st.session_state:
    st.session_state.ai_draft = ""
if "ranking_outputs" not in st.session_state:
    st.session_state.ranking_outputs = {}
if "ranking_configs" not in st.session_state:
    st.session_state.ranking_configs = []
if "ranking_session_id" not in st.session_state:
    st.session_state.ranking_session_id = ""
if "parsed_doc" not in st.session_state:
    st.session_state.parsed_doc = None
if "selected_context" not in st.session_state:
    st.session_state.selected_context = ""
if "_doc_cache_key" not in st.session_state:
    st.session_state._doc_cache_key = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []          # list[Chunk] when auto-chunked
if "current_chunk_idx" not in st.session_state:
    st.session_state.current_chunk_idx = 0


# Default model index for single-pair mode (index into ALL_MODELS)
_DEFAULT_SINGLE_MODEL = "claude-sonnet-4-20250514"

# Context budget (chars) — default 60k (~15k tokens, fits comfortably in
# Claude/GPT-4o/Gemini context windows while leaving room for system
# prompt + generation).  Adjustable via slider in the Control Panel.
DEFAULT_CONTEXT_BUDGET = 60_000


# --- 4. Sidebar ---
with st.sidebar:
    st.header("🧬 Data Management")

    total_pairs = len(st.session_state.dataset)
    single_count = sum(1 for x in st.session_state.dataset if x.get("mode") != "ranking")
    ranking_count = sum(1 for x in st.session_state.dataset if x.get("mode") == "ranking")

    st.metric("Total Pairs", total_pairs)
    if ranking_count > 0:
        col1, col2 = st.columns(2)
        col1.metric("Single", single_count, delta=None)
        col2.metric("Ranked", ranking_count, delta=None)

    if st.session_state.dataset:
        json_str = "\n".join([json.dumps(x) for x in st.session_state.dataset])
        st.download_button(
            label="Download .JSONL",
            data=json_str,
            file_name="dpo_dataset.jsonl",
            mime="application/json"
        )

    st.divider()
    st.markdown("### Instructions")
    st.info(
        "**Single Pair:** Upload PDF → Select Section → Generate → Edit → Commit\n\n"
        "**K-Ranking:** Upload PDF → Select Section → Configure K → Generate → Rank → Commit"
    )
    st.divider()
    st.markdown("### Tech Stack")
    st.markdown("""
    ![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)
    ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
    ![Anthropic](https://img.shields.io/badge/Model-Claude-191919?style=flat&logo=anthropic&logoColor=white)
    ![Gemini](https://img.shields.io/badge/Model-Gemini-4285F4?style=flat&logo=google&logoColor=white)
    ![OpenAI](https://img.shields.io/badge/Model-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
    ![DPO](https://img.shields.io/badge/Method-DPO%20%2F%20RLHF-orange?style=flat)
    """)

# --- 5. Header ---
st.markdown("""
<div class="custom-header">
    <h1>🧬 Financial RLHF Studio</h1>
    <p>Direct Preference Optimization (DPO) Data Engine | Institutional Grade Alignment</p>
</div>
""", unsafe_allow_html=True)

# --- 5a. What This Studio Does ---
st.markdown("""
<div style="background:#1a1c24; border:1px solid #3a3f4b; border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:0.8rem; font-size:0.92em;">
<strong style="color:#ccc;">What is this?</strong>&nbsp;&nbsp;
A human-in-the-loop data engine for financial domain adaptation.
Upload any financial PDF (10-K, 10-Q, earnings call, broker note),
select the relevant section, and generate AI analyses across multiple
models and temperatures. You then <em>correct</em> or <em>rank</em>
the outputs — those preference signals become structured DPO training
pairs for fine-tuning LLMs to institutional standards.
Boilerplate is auto-stripped, oversized sections are auto-chunked,
and a single K-way ranking can produce up to 36 training pairs at once.
</div>
""", unsafe_allow_html=True)

# --- 6. Info Expanders ---
doc_col1, doc_col2 = st.columns(2)

with doc_col1:
    with st.expander("How DPO Training Works", expanded=False):
        st.markdown("""
        **Direct Preference Optimization (DPO)** trains a language model to
        prefer expert-quality outputs over generic ones, without needing a
        separate reward model.

        **The data format** is simple — each training example is a pair:
        * **Chosen:** The expert-corrected or higher-ranked output.
        * **Rejected:** The original AI draft or lower-ranked output.

        **Why it matters for finance:** Generic LLMs hallucinate numbers,
        misapply GAAP vs non-GAAP metrics, miss sector-specific nuance, and
        use the wrong tone for institutional audiences. DPO lets you encode
        these corrections directly as training signal — the model learns
        *your* standards, not just internet-average quality.

        **Two annotation modes:**
        * **Single Pair:** Generate one draft, correct it → 1 preference pair.
        * **K-Ranking:** Generate K outputs (2–9), rank them best→worst →
          up to K(K-1)/2 pairs from one session (e.g. K=4 → 6 pairs).
        """)

with doc_col2:
    with st.expander("Buy-Side Translation (The 'Why')", expanded=False):
        st.markdown("""
        **Think of this as an Automated Analyst Training Program.**

        In traditional finance, a junior analyst writes a draft memo and a
        senior PM red-lines it — correcting tone, fixing GAAP errors, adding
        nuance the junior missed. Over time, the junior learns.

        **This studio digitizes that feedback loop:**
        1. **The AI** generates an analysis from the document section you select.
        2. **You (the expert)** correct the output — fix hallucinations, adjust
           tone, add the context only a domain specialist would know.
        3. **The correction is captured** as a structured preference pair
           (chosen vs rejected) that can train or fine-tune the AI.

        The output is structured DPO training data — chosen/rejected pairs
        that encode your corrections as fine-tuning signal.
        """)

# --- 7. Shared Control Panel ---
with st.container(border=True):
    st.markdown("#### Control Panel")

    input_c1, input_c2 = st.columns([1, 2])

    with input_c1:
        uploaded_file = st.file_uploader("Source Context (PDF)", type="pdf", label_visibility="collapsed")
        if not uploaded_file:
            st.caption("10-K · 10-Q · 8-K · Earnings Transcript · Broker Note · Proxy · Any PDF")
        else:
            st.caption(f"✅ {uploaded_file.name}")

    with input_c2:
        prompt = st.text_input(
            "Analysis prompt",
            placeholder="e.g. Summarize key risks · Extract revenue drivers · Compare margins YoY · Identify red flags",
            value="",
            label_visibility="collapsed",
            help="The instruction sent to the model along with the document context. Leave blank to use a generic analysis prompt.",
        )
        if not prompt:
            prompt = "Analyze the selected section. Identify key themes, material risks, and noteworthy financial metrics."

    # --- Context budget ---
    _BUDGET_OPTIONS = {
        "15k  — Small section (~3.7k tokens)": 15_000,
        "30k  — Medium section (~7.5k tokens)": 30_000,
        "60k  — Standard (default, ~15k tokens)": 60_000,
        "120k — Large section (~30k tokens)": 120_000,
        "200k — Very large (~50k tokens)": 200_000,
        "400k — GPT-4o max (~100k tokens)": 400_000,
        "800k — Claude max (~200k tokens)": 800_000,
    }
    _DEFAULT_BUDGET_LABEL = "60k  — Standard (default, ~15k tokens)"

    budget_label = st.selectbox(
        "Context budget",
        options=list(_BUDGET_OPTIONS.keys()),
        index=list(_BUDGET_OPTIONS.keys()).index(_DEFAULT_BUDGET_LABEL),
        help=(
            "Max chars sent to the model. Larger budgets increase cost and latency. "
            "Claude supports up to ~800k chars; GPT-4o ~500k; Gemini 1.5 Pro ~4M."
        ),
        key="context_budget_select",
    )
    context_budget = _BUDGET_OPTIONS[budget_label]

    # --- Section / Page Selection ---
    if uploaded_file:
        # Parse on upload (cached by filename + size)
        cache_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if cache_key != st.session_state._doc_cache_key:
            with st.spinner("Parsing document structure..."):
                uploaded_file.seek(0)
                st.session_state.parsed_doc = parse_document(uploaded_file)
                st.session_state._doc_cache_key = cache_key
                st.session_state.selected_context = ""
                # Reset file position for any other reads
                uploaded_file.seek(0)

        pdoc = st.session_state.parsed_doc

        # Show cleanup stats if significant
        if pdoc and pdoc.total_chars_removed > 0:
            raw_total = pdoc.total_chars + pdoc.total_chars_removed
            parts = []
            if pdoc.boilerplate_chars_removed > 0:
                bp_pct = pdoc.boilerplate_chars_removed / raw_total * 100
                parts.append(
                    f"**{pdoc.boilerplate_chars_removed:,}** boilerplate ({bp_pct:.0f}%)"
                )
            if pdoc.whitespace_chars_removed > 0:
                ws_pct = pdoc.whitespace_chars_removed / raw_total * 100
                parts.append(
                    f"**{pdoc.whitespace_chars_removed:,}** whitespace ({ws_pct:.0f}%)"
                )
            total_pct = pdoc.total_chars_removed / raw_total * 100
            st.caption(
                f"🧹 Stripped {' + '.join(parts)} "
                f"= **{pdoc.total_chars_removed:,}** total chars removed ({total_pct:.0f}% of raw text)"
            )

        if pdoc and pdoc.has_sections:
            # --- Section mode (structured filing) ---
            st.markdown("##### Document Sections Detected")

            summary = get_section_summary(pdoc.sections, char_budget=context_budget)
            st.dataframe(summary, width="stretch", hide_index=True)

            # Build options for multiselect
            section_options = [sec.title[:80] for sec in pdoc.sections]

            # Auto-select Item 7 (MD&A) if present
            default_selection = []
            for sec in pdoc.sections:
                if sec.item_number in ("7", "7A"):
                    default_selection.append(sec.title[:80])

            selected_titles = st.multiselect(
                "Select sections to include as context",
                options=section_options,
                default=default_selection,
                key="section_select",
            )

            if selected_titles:
                # Map titles back to Section objects
                title_to_section = {sec.title[:80]: sec for sec in pdoc.sections}
                selected_secs = [title_to_section[t] for t in selected_titles if t in title_to_section]

                total_chars = sum(sec.char_count for sec in selected_secs)

                if total_chars <= context_budget:
                    # --- Fits in budget: single context (no chunking) ---
                    st.session_state.chunks = []
                    context_text = assemble_context(
                        pdoc,
                        selected_sections=selected_secs,
                        char_limit=context_budget,
                    )
                    st.session_state.selected_context = context_text

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Selected Chars", f"{total_chars:,}")
                    sc2.metric("~Words", f"{total_chars // 5:,}")
                    sc3.metric("Budget", f"{context_budget:,}", delta=f"{context_budget - total_chars:,} remaining")
                else:
                    # --- Exceeds budget: auto-chunk ---
                    chunks = chunk_context(pdoc, selected_secs, chunk_budget=context_budget)
                    st.session_state.chunks = chunks

                    # Clamp index
                    if st.session_state.current_chunk_idx >= len(chunks):
                        st.session_state.current_chunk_idx = 0

                    st.info(
                        f"📦 Section is **{total_chars:,} chars** — auto-chunked into "
                        f"**{len(chunks)} chunks** of ~{context_budget:,} chars each."
                    )

                    # --- Chunk navigator ---
                    nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])

                    with nav_c1:
                        if st.button("← Prev", key="chunk_prev",
                                     disabled=(st.session_state.current_chunk_idx == 0)):
                            st.session_state.current_chunk_idx -= 1
                            st.rerun()

                    with nav_c2:
                        chunk_idx = st.selectbox(
                            "Chunk",
                            options=list(range(len(chunks))),
                            index=st.session_state.current_chunk_idx,
                            format_func=lambda i: (
                                f"Chunk {i + 1}/{len(chunks)} — "
                                f"{chunks[i].char_count:,} chars"
                                + (f" ({chunks[i].section_title[:40]})"
                                   if len(selected_secs) > 1 else "")
                            ),
                            key="chunk_nav_select",
                        )
                        if chunk_idx != st.session_state.current_chunk_idx:
                            st.session_state.current_chunk_idx = chunk_idx
                            st.rerun()

                    with nav_c3:
                        if st.button("Next →", key="chunk_next",
                                     disabled=(st.session_state.current_chunk_idx >= len(chunks) - 1)):
                            st.session_state.current_chunk_idx += 1
                            st.rerun()

                    # Set selected context to current chunk
                    active_chunk = chunks[st.session_state.current_chunk_idx]
                    st.session_state.selected_context = active_chunk.text

                    # Chunk metrics
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Chunk Chars", f"{active_chunk.char_count:,}")
                    sc2.metric("~Words", f"{active_chunk.char_count // 5:,}")
                    sc3.metric("Budget", f"{context_budget:,}",
                               delta=f"{context_budget - active_chunk.char_count:,} remaining")

            else:
                st.session_state.selected_context = ""
                st.session_state.chunks = []
                st.caption("Select at least one section to use as context.")

        elif pdoc:
            # --- Page-range mode (unstructured document) ---
            st.markdown("##### No section headers detected — select page range")
            st.caption(f"Document has {pdoc.total_pages} pages, {pdoc.total_chars:,} total chars.")

            max_default = min(10, pdoc.total_pages)
            page_range = st.slider(
                "Page range (1-indexed)",
                min_value=1,
                max_value=pdoc.total_pages,
                value=(1, max_default),
                key="page_range_slider",
            )

            # Convert to 0-indexed
            start_page = page_range[0] - 1
            end_page = page_range[1] - 1

            context_text = assemble_context(
                pdoc,
                selected_pages=(start_page, end_page),
                char_limit=context_budget,
            )
            st.session_state.selected_context = context_text

            page_chars = sum(len(pdoc.page_texts[p]) for p in range(start_page, end_page + 1))
            sc1, sc2 = st.columns(2)
            sc1.metric("Selected Chars", f"{page_chars:,}")
            if page_chars > context_budget:
                sc2.metric("Budget", f"{context_budget:,}", delta=f"-{page_chars - context_budget:,} over", delta_color="inverse")
            else:
                sc2.metric("Budget", f"{context_budget:,}", delta=f"{context_budget - page_chars:,} remaining")

    # Context preview (collapsible)
    if st.session_state.selected_context:
        _preview_label = "Preview selected context"
        if st.session_state.chunks:
            ci = st.session_state.current_chunk_idx
            _preview_label = f"Preview — Chunk {ci + 1}/{len(st.session_state.chunks)}"
        with st.expander(_preview_label, expanded=False):
            st.text_area(
                "context_preview",
                value=st.session_state.selected_context[:3000] + (
                    "\n\n[... preview truncated ...]" if len(st.session_state.selected_context) > 3000 else ""
                ),
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )


# ============================================================
# TABS
# ============================================================
tab2, tab1 = st.tabs(["K-Output Ranking Mode", "Single Pair Mode"])

# --- TAB 1: SINGLE PAIR (original workflow) ---
with tab1:
    # Model selector + Generate button
    sp_col1, sp_col2 = st.columns([2, 1])
    with sp_col1:
        single_model = st.selectbox(
            "Model",
            ALL_MODELS,
            index=ALL_MODELS.index(_DEFAULT_SINGLE_MODEL),
            key="single_pair_model",
            help="LLM used to generate the initial draft. You'll correct this output to create the preference pair.",
        )
    with sp_col2:
        st.write("")  # vertical alignment spacer
        gen_clicked = st.button("Generate Draft", type="primary", key="single_gen")

    if gen_clicked:
        if not uploaded_file:
            st.error("Please upload a PDF first.")
        elif not st.session_state.selected_context:
            st.error("Please select sections or a page range first.")
        else:
            with st.spinner("Generating..."):
                config = GenerationConfig(label="draft", model=single_model)
                st.session_state.ai_draft = generate_draft(
                    st.session_state.selected_context, prompt, config,
                    context_limit=context_budget,
                )

    # Workbench
    if st.session_state.ai_draft:
        with st.container(border=True):
            st.markdown("#### DPO Workbench")

            edit_c1, edit_c2 = st.columns(2)

            with edit_c1:
                st.markdown("**Rejected (AI Draft)**")
                draft_text = st.text_area(
                    "Rejected", value=st.session_state.ai_draft,
                    height=200, label_visibility="collapsed", disabled=True
                )

            with edit_c2:
                st.markdown("**Chosen (Expert Rewrite)**")
                expert_text = st.text_area(
                    "Chosen", value=st.session_state.ai_draft,
                    height=200, label_visibility="collapsed"
                )

            st.divider()

            meta_c1, meta_c2 = st.columns([3, 1])

            with meta_c1:
                tags = st.multiselect(
                    "Error Taxonomy",
                    ["Hallucination", "Missed Nuance", "Incorrect Tone", "Math Error", "GAAP vs Non-GAAP", "Outdated Info"],
                    placeholder="Select error types...",
                    key="single_tags"
                )

            with meta_c2:
                st.write("")
                st.write("")
                if st.button("Commit Pair", width="stretch", key="single_commit"):
                    if draft_text == expert_text:
                        st.error("No changes detected.")
                    else:
                        entry = {
                            "timestamp": datetime.now().isoformat(),
                            "prompt": prompt,
                            "chosen": expert_text,
                            "rejected": draft_text,
                            "tags": tags,
                            "source": uploaded_file.name if uploaded_file else "Manual",
                            "mode": "single_pair",
                        }
                        st.session_state.dataset.append(entry)
                        st.toast(f"Saved! Total pairs: {len(st.session_state.dataset)}", icon="✅")
    else:
        st.info("Upload a document, select sections, and click Generate Draft to begin.")


# --- TAB 2: K-OUTPUT RANKING ---
with tab2:

    # ---- Step 1: Configure Generation ----
    with st.container(border=True):
        st.markdown("#### Generation Configuration")
        st.caption(
            "Pick what to vary across outputs, then choose a preset. "
            "Your ranking produces K(K-1)/2 preference pairs for DPO training."
        )

        # --- Row 1: What to compare (category) ---
        category_names = list(PRESET_CATEGORIES.keys())
        category = st.radio(
            "Compare by",
            category_names,
            horizontal=True,
            key="preset_category",
        )

        if category == "Custom":
            # --- Full custom mode ---
            k_count = st.slider("Number of outputs (K)", min_value=2, max_value=9, value=4, key="k_slider")

            custom_configs = []
            for idx in range(k_count):
                lbl = chr(ord("A") + idx)
                with st.expander(f"Output {lbl}", expanded=(idx == 0)):
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        model = st.selectbox("Model", ALL_MODELS, key=f"model_{lbl}")
                    with cc2:
                        temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key=f"temp_{lbl}")

                    persona_names = list(DEFAULT_SYSTEM_PROMPTS.keys())
                    persona = st.selectbox("Persona", persona_names, key=f"persona_{lbl}")
                    sys_prompt = DEFAULT_SYSTEM_PROMPTS[persona]

                    custom_configs.append(GenerationConfig(
                        label=lbl, model=model, temperature=temp, system_prompt=sys_prompt
                    ))

            active_configs = custom_configs
        else:
            # --- Row 2: Specific preset (only if category has multiple) ---
            sub_presets = PRESET_CATEGORIES[category]
            sub_names = list(sub_presets.keys())

            if len(sub_names) == 1:
                preset_key = sub_names[0]
            else:
                preset_key = st.radio(
                    "Preset",
                    sub_names,
                    horizontal=True,
                    key="preset_sub_select",
                )

            active_configs = sub_presets[preset_key]

            # --- Always show the config table ---
            _prompt_to_persona = {v: k for k, v in DEFAULT_SYSTEM_PROMPTS.items()}
            personas = [_prompt_to_persona.get(c.system_prompt, "Custom") for c in active_configs]
            show_persona = len(set(personas)) > 1

            summary_data = []
            for c, persona in zip(active_configs, personas):
                row = {
                    "Output": c.label,
                    "Provider": c.provider,
                    "Model": c.model,
                    "Temperature": c.temperature,
                }
                if show_persona:
                    row["Persona"] = persona
                summary_data.append(row)

            st.dataframe(summary_data, use_container_width=True, hide_index=True)

        k = len(active_configs)
        pair_preview = count_pairs(k)
        st.caption(f"K = {k} outputs → **{pair_preview} pairwise pairs** per ranking")

        # Generate button
        if st.button(f"Generate {k} Outputs", type="primary", key="ranking_gen"):
            if not uploaded_file:
                st.error("Please upload a PDF first.")
            elif not st.session_state.selected_context:
                st.error("Please select sections or a page range first.")
            else:
                progress_bar = st.progress(0, text="Generating outputs...")

                def update_progress(completed, total, status=""):
                    frac = max(0.01, completed / total)  # Avoid 0 so bar is visible
                    text = status if status else f"Generated {completed}/{total}..."
                    progress_bar.progress(frac, text=text)

                # api_key=None → each config resolves its own key via provider
                outputs = generate_k_outputs(
                    st.session_state.selected_context, prompt, active_configs,
                    api_key=None, progress_callback=update_progress,
                    context_limit=context_budget,
                )
                progress_bar.empty()

                st.session_state.ranking_outputs = outputs
                st.session_state.ranking_configs = active_configs
                st.session_state.ranking_session_id = str(uuid.uuid4())
                st.rerun()

    # ---- Step 2: Review & Rank ----
    if st.session_state.ranking_outputs and st.session_state.ranking_configs:
        configs_by_label = {c.label: c for c in st.session_state.ranking_configs}
        outputs = st.session_state.ranking_outputs

        with st.container(border=True):
            st.markdown("#### Review Outputs")
            st.caption("Read each output, then rank them below. Use the scratchpad to take notes while reviewing.")

            for label, text in outputs.items():
                cfg = configs_by_label[label]
                header = f"Output {label}  —  {cfg.provider}/{cfg.model}, temp={cfg.temperature}"
                with st.expander(header, expanded=True):
                    st.text_area(f"output_{label}", value=text, height=350, disabled=True, label_visibility="collapsed")

        # ---- Scratchpad for notes while reviewing ----
        with st.expander("📝 Scratchpad — jot notes while reviewing", expanded=False):
            st.caption(
                "Temporary workspace only — paste snippets, note differences, "
                "track your reasoning. This is not saved or included in the output."
            )
            st.text_area(
                "scratchpad",
                value="",
                height=150,
                placeholder="e.g. Output A nails the revenue breakdown but misses the risk factors. C has better tone but hallucinates margin numbers...",
                label_visibility="collapsed",
                key="ranking_scratchpad",
            )

        with st.container(border=True):
            st.markdown("#### Rank Outputs (Best → Worst)")
            st.caption("Position 1 = best. Drag to reorder, or use manual rank numbers.")

            labels_list = list(outputs.keys())

            use_manual = st.checkbox("Use manual rank numbers", key="manual_rank_toggle")

            if use_manual:
                # Number-input fallback
                rank_assignments = {}
                cols = st.columns(min(len(labels_list), 4))
                for idx, label in enumerate(labels_list):
                    cfg = configs_by_label[label]
                    with cols[idx % len(cols)]:
                        rank = st.number_input(
                            f"{label} ({cfg.model})",
                            min_value=1, max_value=len(labels_list), value=idx + 1,
                            key=f"manual_rank_{label}"
                        )
                        rank_assignments[label] = rank

                # Sort by assigned rank (ties broken by label order)
                sorted_labels = sorted(rank_assignments.keys(), key=lambda lbl: (rank_assignments[lbl], lbl))
                final_order = sorted_labels

                # Validate: check for duplicate ranks
                rank_vals = list(rank_assignments.values())
                if len(set(rank_vals)) != len(rank_vals):
                    st.warning("Duplicate ranks detected — please assign unique ranks.")

            elif sort_items is not None:
                # Drag-and-drop via streamlit-sortables
                display_labels = [
                    f"Output {lbl}  —  {configs_by_label[lbl].model}, t={configs_by_label[lbl].temperature}"
                    for lbl in labels_list
                ]

                sorted_display = sort_items(display_labels, direction="vertical", key="ranking_sort")

                # Map display strings back to labels
                display_to_label = {d: lbl for d, lbl in zip(display_labels, labels_list)}
                final_order = [display_to_label[d] for d in sorted_display]

            else:
                st.warning("streamlit-sortables not installed. Using manual rank numbers.")
                use_manual = True
                # Fall through to manual mode on next rerun
                final_order = labels_list

            # Show current ranking
            st.markdown("**Current ranking:**")
            ranking_display = "  →  ".join([f"**{lbl}** (#{i+1})" for i, lbl in enumerate(final_order)])
            st.markdown(ranking_display)

        # ---- Step 3: Tag & Commit ----
        with st.container(border=True):
            st.markdown("#### Tag & Commit")

            tag_c, commit_c = st.columns([3, 1])

            with tag_c:
                ranking_tags = st.multiselect(
                    "Error Taxonomy",
                    ["Hallucination", "Missed Nuance", "Incorrect Tone", "Math Error", "GAAP vs Non-GAAP", "Outdated Info"],
                    placeholder="Select error types...",
                    key="ranking_tags"
                )

            with commit_c:
                n_pairs = count_pairs(len(final_order))
                st.metric("Pairs", n_pairs)

                if st.button(f"Commit {n_pairs} Pairs", type="primary", width="stretch", key="ranking_commit"):
                    # Validate no duplicate ranks in manual mode
                    if use_manual:
                        rank_vals = list(rank_assignments.values())
                        if len(set(rank_vals)) != len(rank_vals):
                            st.error("Fix duplicate ranks before committing.")
                            st.stop()

                    pairs = extract_pairwise_preferences(
                        ranked_labels=final_order,
                        outputs=outputs,
                        configs_by_label=configs_by_label,
                        prompt=prompt,
                        source=uploaded_file.name if uploaded_file else "Manual",
                        tags=ranking_tags,
                        session_id=st.session_state.ranking_session_id,
                    )
                    st.session_state.dataset.extend(pairs)
                    st.toast(f"Saved {n_pairs} pairs! Total: {len(st.session_state.dataset)}", icon="✅")

    else:
        st.info("Configure your generation settings above, then click Generate to produce K outputs for ranking.")


# --- Footer ---
st.caption("v0.8 | Financial RLHF Studio — Boilerplate Filtering + Auto-Chunking + Multi-Provider + K-Output Ranking")
