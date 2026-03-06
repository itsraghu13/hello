"""
Databricks Pipeline Rewriter Agent
====================================
Streamlit app for Databricks Apps environment.
Rewrites Lakebridge-converted PySpark scripts into clean Databricks notebooks.
"""

import streamlit as st
import anthropic
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Rewriter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert Python and PySpark developer specialising in Databricks pipelines and
Lakebridge-converted ETL code.

When a user provides a PySpark or Databricks pipeline script, analyse the code and
fully correct, restructure, and optimise it according to the standards below.

Return ONLY the fully rewritten Databricks notebook script inside a ```python code block.
Do not include explanations or commentary outside the code block.

══════════════════════════════════════════════════════════
NOTEBOOK OUTPUT FORMAT (DATABRICKS)
══════════════════════════════════════════════════════════

Return the result as a Databricks Python notebook script using cell markers:

# COMMAND ----------

Organise cells in this order:
1. Documentation (Markdown cell)
2. Imports
3. Constants
4. Helper Functions
5. Widgets / Configuration
6. Data Sources
7. Transformations
8. Outputs

The notebook must remain executable as a .py file.

══════════════════════════════════════════════════════════
MODULE DOCUMENTATION (FIRST CELL)
══════════════════════════════════════════════════════════

The first cell must be a Markdown cell with:
- Business Context: Real-world business process this pipeline supports
- Pipeline Purpose: What the notebook does at a high level
- Input Tables: Source tables used (catalog.schema.table_name)
- Output Tables: Tables produced (catalog.schema.table_name)
- Processing Steps: Short ordered list of major transformation steps

══════════════════════════════════════════════════════════
STRUCTURE & ORGANISATION
══════════════════════════════════════════════════════════

- Rewrite as a flat Databricks notebook pipeline.
- Remove the main() function entirely.
- Do NOT include: if __name__ == "__main__", entry-point wrappers, or pipeline orchestration functions.
- Remove unused imports: oracledb, SparkContext, explode (if unused), count (if unused)
- Deduplicate imports.
- Consolidate ALL pyspark.sql.functions imports into a single line.
- Consolidate ALL pyspark.sql.types imports into a single line (if used).

══════════════════════════════════════════════════════════
DATA SOURCES — USE TABLES NOT FILES
══════════════════════════════════════════════════════════

Replace all file-based reads (spark.read.csv, spark.read.parquet, etc.)
with: spark.table("catalog.schema.table_name")

Use fully qualified catalog.schema.table_name format (Unity Catalog).
Avoid filesystem reads unless absolutely necessary.

══════════════════════════════════════════════════════════
LAKEBRIDGE CORRECTIONS
══════════════════════════════════════════════════════════

1. AGGREGATION CORRECTIONS
   Detect any groupBy() without a following .agg() / .sum() / .count() / .min() / .max() / .avg().
   Fix by inferring the correct aggregation from column names and semantics.
   If intent is unclear, use first() with comment: # assumed constant within group — verify with business
   Ensure every non-grouped column is aggregated.

2. DATE FORMAT CORRECTIONS
   Replace invalid tokens:
   "y-MM-dd"  → "yyyy-MM-dd"
   "yMMdd"    → "yyyyMMdd"
   "y/MM/dd"  → "yyyy/MM/dd"
   Single y   → yyyy (unless 2-digit year is explicitly intended)
   Apply in both PySpark functions and SQL expressions.

3. NULL / EMPTY VALUE HANDLING
   Use coalesce() for nullable columns feeding filters, joins, or aggregations.

══════════════════════════════════════════════════════════
OUTPUTS — REPLACE CSV WRITES
══════════════════════════════════════════════════════════

Replace all CSV writes with:
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("catalog.schema.table_name")
)

Add a comment above each write describing the table contents.
Format .write chains with one method per line.

══════════════════════════════════════════════════════════
NAMING & STYLE (PEP 8)
══════════════════════════════════════════════════════════

- UPPER_SNAKE_CASE for module-level constants.
- snake_case for variables, functions, DataFrame names.
- Replace magic literals ('Y', 'N', 'A', '1 ') with named constants.
- Lines under 100 characters.
- Break long chains using parentheses.

══════════════════════════════════════════════════════════
WIDGETS / CONFIGURATION
══════════════════════════════════════════════════════════

Expose all configurable values via dbutils.widgets with inline comments.
Wrap widget reads with try/except for safe local execution.

══════════════════════════════════════════════════════════
JOINS — BEST PRACTICES
══════════════════════════════════════════════════════════

- Always alias DataFrames before joining.
- Use broadcast hints for small lookup/dimension tables.
- Remove duplicate columns after joins using explicit select().

══════════════════════════════════════════════════════════
READABILITY & SECTION COMMENTS
══════════════════════════════════════════════════════════

Add descriptive section comments:
# --- Source Reads ---
# --- Data Cleansing ---
# --- Null / Default Handling ---
# --- Validation ---
# --- Transformations ---
# --- Business Rule Application ---
# --- Aggregations ---
# --- Window Functions ---
# --- Inserts with Key ---
# --- Deletions ---
# --- Final Projection ---
# --- Output Tables ---

If the same select() column list appears more than once, extract it into a helper function.

══════════════════════════════════════════════════════════
HELPER FUNCTION DOCUMENTATION
══════════════════════════════════════════════════════════

Add Google-style docstrings to every helper function with Args and Returns sections.

══════════════════════════════════════════════════════════
PYSPARK BEST PRACTICES
══════════════════════════════════════════════════════════

- Use f-strings for dynamic SQL expressions.
- Replace Python string concat with concat() / lit() in Spark expressions.
- Use display(df) instead of df.show().
- Never use select("*") after a join.
- Avoid: .collect(), .toPandas(), .repartition(1), .coalesce(1)

══════════════════════════════════════════════════════════
PERFORMANCE OPTIMISATION
══════════════════════════════════════════════════════════

- Cache DataFrames reused in multiple downstream steps with explanatory comment.
- Use broadcast() for small tables.
- Apply filter / predicate pushdown as early as possible.

══════════════════════════════════════════════════════════
DELTA LAKE — WRITE PATTERNS
══════════════════════════════════════════════════════════

For upsert/SCD patterns use MERGE INTO SQL, not overwrite.

══════════════════════════════════════════════════════════
OUTPUT REQUIREMENT
══════════════════════════════════════════════════════════

Return the final Databricks notebook script inside a ```python code block ONLY.
Do not include any explanations or text outside the code block.
""".strip()

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  /* Root theme */
  :root {
    --bg-primary:    #0e0e10;
    --bg-card:       #16161a;
    --bg-input:      #1c1c22;
    --border:        #2a2a35;
    --accent:        #ff6b2b;
    --accent-dim:    rgba(255,107,43,0.12);
    --accent-glow:   rgba(255,107,43,0.35);
    --text-primary:  #f0f0f5;
    --text-muted:    #7a7a90;
    --green:         #3dd68c;
    --blue:          #4b9eff;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text-primary) !important;
  }

  [data-testid="stAppViewContainer"] > .main { background: var(--bg-primary) !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  section[data-testid="stSidebar"] { background: var(--bg-card) !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

  /* Typography */
  h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 700 !important; }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, #16161a 0%, #1e1e28 50%, #16161a 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    padding: 32px 36px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2rem; font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 6px;
    letter-spacing: -0.5px;
  }
  .hero-title span { color: var(--accent); }
  .hero-sub {
    font-size: 0.9rem; color: var(--text-muted);
    font-weight: 300; margin: 0;
    font-family: 'IBM Plex Mono', monospace;
  }
  .hero-badge {
    display: inline-block;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 14px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px; text-transform: uppercase;
  }

  /* Cards */
  .panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .panel-title {
    font-size: 0.72rem; font-weight: 600;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 14px;
    font-family: 'IBM Plex Mono', monospace;
    display: flex; align-items: center; gap: 8px;
  }
  .panel-title::before {
    content: ''; display: inline-block;
    width: 3px; height: 14px;
    background: var(--accent); border-radius: 2px;
  }

  /* Stat pills */
  .stat-row { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
  .stat-pill {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1; min-width: 110px;
    text-align: center;
  }
  .stat-pill .val {
    font-size: 1.4rem; font-weight: 700;
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    display: block;
  }
  .stat-pill .lbl {
    font-size: 0.68rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px;
  }

  /* Streamlit widgets */
  .stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
  }
  .stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
  }
  .stButton > button:hover {
    background: #ff8752 !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px);
  }
  .stDownloadButton > button {
    background: transparent !important;
    color: var(--green) !important;
    border: 1px solid var(--green) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(61,214,140,0.1) !important;
    box-shadow: 0 0 12px rgba(61,214,140,0.3) !important;
  }

  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
  }

  /* Code block */
  .stCodeBlock { border-radius: 10px !important; }
  pre {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
  }

  /* Status indicators */
  .status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
  }
  .status-dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.orange { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .status-dot.blue { background: var(--blue); box-shadow: 0 0 6px var(--blue); }

  /* Progress bar */
  .stProgress > div > div > div { background: var(--accent) !important; }

  /* Alerts */
  [data-testid="stAlert"] {
    background: var(--bg-input) !important;
    border-radius: 8px !important;
  }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Input labels */
  label, .stRadio label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">⚡ Pipeline <span>Rewriter</span> Agent</div>
  <p class="hero-sub">databricks · pyspark · lakebridge · unity catalog</p>
  <div class="hero-badge">✦ Powered by Claude</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "rewritten_code" not in st.session_state:
    st.session_state.rewritten_code = ""
if "input_code" not in st.session_state:
    st.session_state.input_code = ""
if "run_stats" not in st.session_state:
    st.session_state.run_stats = {}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_python_code(response_text: str) -> str:
    """Strip ```python fences from the model response.

    Args:
        response_text: Raw text returned by the Anthropic API.

    Returns:
        Clean Python code string without markdown fences.
    """
    text = response_text.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.rsplit("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def count_cells(code: str) -> int:
    """Count Databricks notebook cells by marker.

    Args:
        code: Rewritten notebook source code.

    Returns:
        Number of COMMAND ---------- cells detected.
    """
    return code.count("# COMMAND ----------")


def count_lines(code: str) -> int:
    """Count non-empty lines in code.

    Args:
        code: Source code string.

    Returns:
        Count of non-empty lines.
    """
    return sum(1 for line in code.splitlines() if line.strip())


def call_agent(code: str, model: str) -> tuple[str, float]:
    """Send the input code to the Anthropic API and return the rewritten notebook.

    Args:
        code: Raw PySpark pipeline source code to rewrite.
        model: Anthropic model identifier string.

    Returns:
        Tuple of (rewritten_code, elapsed_seconds).
    """
    client = anthropic.Anthropic()
    t0 = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Please rewrite the following PySpark / Databricks pipeline script "
                    "according to the standards in your instructions.\n\n"
                    f"```python\n{code}\n```"
                ),
            }
        ],
    )
    elapsed = round(time.time() - t0, 1)
    raw = message.content[0].text
    return extract_python_code(raw), elapsed


# ─────────────────────────────────────────────
# LAYOUT — TWO COLUMNS
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT — INPUT ──────────────────────────────
with col_left:
    st.markdown('<div class="panel-title">Input Script</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["📋  Paste code", "📂  Upload .py file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_code = ""

    if input_mode == "📋  Paste code":
        raw_code = st.text_area(
            "Paste your PySpark script",
            height=420,
            placeholder="# Paste your Lakebridge-converted PySpark script here...\n\ndef main():\n    df = spark.read.csv('/mnt/data/input.csv')\n    ...",
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "Upload a .py file",
            type=["py"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw_code = uploaded.read().decode("utf-8")
            st.success(f"✓ Loaded: `{uploaded.name}` — {count_lines(raw_code):,} lines")
            with st.expander("Preview uploaded file", expanded=False):
                st.code(raw_code[:3000] + ("\n... (truncated)" if len(raw_code) > 3000 else ""), language="python")

    st.markdown("<br>", unsafe_allow_html=True)

    # Config row
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        model_choice = st.selectbox(
            "Model",
            [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-5-20251001",
            ],
            index=0,
        )
    with cfg_col2:
        output_catalog = st.text_input(
            "Target catalog (optional)",
            placeholder="my_catalog",
            help="Prepend this catalog to inferred table names",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button("⚡  Rewrite Notebook", use_container_width=True)

    # Input stats
    if raw_code:
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-pill">
            <span class="val">{count_lines(raw_code):,}</span>
            <span class="lbl">Input lines</span>
          </div>
          <div class="stat-pill">
            <span class="val">{len(raw_code):,}</span>
            <span class="lbl">Characters</span>
          </div>
          <div class="stat-pill">
            <span class="val">{raw_code.count("def "):,}</span>
            <span class="lbl">Functions</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT — OUTPUT ────────────────────────────
with col_right:
    st.markdown('<div class="panel-title">Rewritten Notebook</div>', unsafe_allow_html=True)

    # ── RUN AGENT ──
    if run_btn:
        if not raw_code.strip():
            st.error("⚠️  Paste or upload a PySpark script first.")
        else:
            with st.spinner("Agent is rewriting your pipeline…"):
                progress = st.progress(0)
                for pct in [15, 35, 55]:
                    time.sleep(0.3)
                    progress.progress(pct)
                try:
                    result, elapsed = call_agent(raw_code, model_choice)
                    progress.progress(100)
                    time.sleep(0.2)
                    progress.empty()

                    st.session_state.rewritten_code = result
                    st.session_state.input_code = raw_code
                    st.session_state.run_stats = {
                        "elapsed": elapsed,
                        "cells": count_cells(result),
                        "out_lines": count_lines(result),
                        "model": model_choice.split("-")[1].capitalize(),
                    }
                    st.success(f"✓ Rewritten in {elapsed}s")

                except anthropic.APIConnectionError:
                    progress.empty()
                    st.error("❌ Cannot reach Anthropic API. Check your network or ANTHROPIC_API_KEY secret.")
                except anthropic.AuthenticationError:
                    progress.empty()
                    st.error("❌ Invalid API key. Set the ANTHROPIC_API_KEY Databricks secret.")
                except Exception as exc:
                    progress.empty()
                    st.error(f"❌ Unexpected error: {exc}")

    # ── OUTPUT DISPLAY ──
    if st.session_state.rewritten_code:
        stats = st.session_state.run_stats
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-pill">
            <span class="val">{stats.get('cells', 0)}</span>
            <span class="lbl">Cells</span>
          </div>
          <div class="stat-pill">
            <span class="val">{stats.get('out_lines', 0):,}</span>
            <span class="lbl">Output lines</span>
          </div>
          <div class="stat-pill">
            <span class="val">{stats.get('elapsed', 0)}s</span>
            <span class="lbl">Elapsed</span>
          </div>
          <div class="stat-pill">
            <span class="val" style="font-size:0.8rem">{stats.get('model', '')}</span>
            <span class="lbl">Model</span>
          </div>
        </div>
        <br>
        """, unsafe_allow_html=True)

        tab_preview, tab_raw = st.tabs(["🗂  Preview", "📄  Raw"])

        with tab_preview:
            st.code(st.session_state.rewritten_code, language="python")

        with tab_raw:
            st.text_area(
                "Raw output",
                value=st.session_state.rewritten_code,
                height=480,
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇  Download .py Notebook",
                data=st.session_state.rewritten_code,
                file_name="rewritten_notebook.py",
                mime="text/x-python",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="⬇  Download as .txt",
                data=st.session_state.rewritten_code,
                file_name="rewritten_notebook.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.markdown("""
        <div style="
          border: 1.5px dashed #2a2a35;
          border-radius: 10px;
          padding: 80px 24px;
          text-align: center;
          color: #7a7a90;
        ">
          <div style="font-size:2.5rem; margin-bottom:14px;">⚡</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.82rem; margin-bottom:6px;">
            Awaiting input
          </div>
          <div style="font-size:0.75rem;">
            Paste or upload a PySpark script, then click Rewrite Notebook
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER — HOW IT WORKS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

fc1, fc2, fc3, fc4 = st.columns(4)
steps = [
    ("01", "Paste / Upload", "Provide your raw Lakebridge-converted PySpark script"),
    ("02", "Agent Rewrites", "Claude applies Databricks standards, fixes aggregations & dates"),
    ("03", "Review Output", "Inspect cell-by-cell preview with syntax highlighting"),
    ("04", "Download", "Export as .py notebook ready to import into Databricks"),
]
for col, (num, title, desc) in zip([fc1, fc2, fc3, fc4], steps):
    with col:
        st.markdown(f"""
        <div style="padding:16px; border:1px solid #2a2a35; border-radius:8px; background:#16161a;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                      color:#ff6b2b; font-weight:600; margin-bottom:6px;">{num}</div>
          <div style="font-weight:600; font-size:0.85rem; margin-bottom:4px;">{title}</div>
          <div style="font-size:0.75rem; color:#7a7a90; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
