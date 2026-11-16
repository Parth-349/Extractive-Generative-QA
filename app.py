"""
Streamlit Web Application: Extractive vs Generative Question Answering
Side-by-side comparative analysis of extractive QA and Ollama or Groq generative QA.
"""

import html
import os
import streamlit as st
from typing import Tuple, List

from config import (
    EXTRACTIVE_MODEL_NAME,
    GENERATIVE_MODEL_NAME,
    OLLAMA_BASE_URL,
    GROQ_BASE_URL,
    GROQ_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    SAMPLE_PRESETS,
    MAX_ANSWER_LENGTH,
)
from extractive_qa import ExtractiveQAModel, ExtractiveResult
from generative_qa import GenerativeQAModel, GenerativeResult
from comparison import compare_answers, AgreementStatus


def _get_setting(key: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets, falling back to the environment."""
    try:
        value = st.secrets.get(key)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(key, default)


def _format_excerpt_html(passage: str, ext_res: ExtractiveResult) -> str:
    """Return a safely escaped passage excerpt with the extracted span emphasized."""
    start = ext_res.start_char
    end = ext_res.end_char
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(passage):
        excerpt_start = max(0, start - 80)
        excerpt_end = min(len(passage), end + 80)
        prefix = "..." if excerpt_start > 0 else ""
        suffix = "..." if excerpt_end < len(passage) else ""
        before = html.escape(passage[excerpt_start:start])
        matched = html.escape(passage[start:end])
        after = html.escape(passage[end:excerpt_end])
        return f"{prefix}{before}<strong>{matched}</strong>{after}{suffix}"

    fallback = (getattr(ext_res, "context_excerpt", "") or "").replace("**", "")
    return html.escape(fallback)


# --- Page Configuration ---
st.set_page_config(
    page_title="Extractive vs Generative QA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global CSS: glassmorphism dark theme ---
st.markdown(
    """
    <style>
    /* ══════════════════════════════════════════════════════════
       BASE & TYPOGRAPHY
    ══════════════════════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        background-color: #0e0f14;
        color: #d8dce8;
    }

    /* Page title */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a0b4ff 0%, #c8aaff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Section headings */
    h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #c8d0f0 !important;
        border-bottom: 1px solid rgba(100, 110, 180, 0.25);
        padding-bottom: 0.3rem;
        margin-top: 0.5rem !important;
    }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #d0d6f0 !important; }

    /* ══════════════════════════════════════════════════════════
       GLASS SURFACE UTILITY
       Used by cards, metric containers, expanders
    ══════════════════════════════════════════════════════════ */
    .glass {
        background: rgba(18, 20, 32, 0.72);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(100, 110, 200, 0.18);
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0, 0, 10, 0.4);
    }
    /* Fallback when backdrop-filter unsupported */
    @supports not (backdrop-filter: blur(1px)) {
        .glass { background: rgba(18, 20, 36, 0.95); }
    }

    /* ══════════════════════════════════════════════════════════
       ANSWER CARDS
    ══════════════════════════════════════════════════════════ */
    /* Extractive – teal/cyan accent */
    .answer-card-ext {
        background: rgba(13, 30, 30, 0.80);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(33, 199, 178, 0.22);
        border-left: 3px solid #21c7b2;
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        margin: 0.4rem 0 0.9rem;
        box-shadow: 0 2px 16px rgba(33, 199, 178, 0.08);
    }
    @supports not (backdrop-filter: blur(1px)) {
        .answer-card-ext { background: rgba(13, 30, 30, 0.97); }
    }
    .answer-card-ext .label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #21c7b2;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .answer-card-ext .text {
        font-size: 1.0rem;
        font-weight: 600;
        color: #dff5f2;
        line-height: 1.5;
    }

    /* Generative – indigo/violet accent */
    .answer-card-gen {
        background: rgba(18, 12, 30, 0.80);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(140, 90, 230, 0.22);
        border-left: 3px solid #8c5ae6;
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        margin: 0.4rem 0 0.9rem;
        box-shadow: 0 2px 16px rgba(140, 90, 230, 0.08);
    }
    @supports not (backdrop-filter: blur(1px)) {
        .answer-card-gen { background: rgba(18, 12, 30, 0.97); }
    }
    .answer-card-gen .label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9b6ef0;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .answer-card-gen .text {
        font-size: 1.0rem;
        font-weight: 600;
        color: #ece8ff;
        line-height: 1.5;
    }

    /* ══════════════════════════════════════════════════════════
       CONTEXT EXCERPT BOX
    ══════════════════════════════════════════════════════════ */
    .excerpt-box {
        background: rgba(10, 20, 24, 0.75);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(60, 90, 100, 0.35);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.87rem;
        line-height: 1.65;
        color: #b8d4cf;
        margin-top: 0.4rem;
    }
    @supports not (backdrop-filter: blur(1px)) {
        .excerpt-box { background: rgba(10, 20, 24, 0.97); }
    }

    /* ══════════════════════════════════════════════════════════
       MODEL BADGE
    ══════════════════════════════════════════════════════════ */
    .model-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
        background: rgba(28, 36, 48, 0.8);
        border: 1px solid rgba(80, 100, 160, 0.3);
        border-radius: 5px;
        padding: 0.15rem 0.55rem;
        color: #7eb8d4;
        margin-bottom: 0.65rem;
        letter-spacing: 0.03em;
    }

    /* ══════════════════════════════════════════════════════════
       SIDEBAR – MODEL INFO CARD
    ══════════════════════════════════════════════════════════ */
    .sidebar-model-card {
        background: rgba(18, 22, 38, 0.80);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(80, 100, 180, 0.2);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
    }
    @supports not (backdrop-filter: blur(1px)) {
        .sidebar-model-card { background: rgba(18, 22, 38, 0.97); }
    }
    .sidebar-model-card .mc-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.4rem;
    }
    .sidebar-model-card .mc-row:last-child { margin-bottom: 0; }
    .sidebar-model-card .mc-icon {
        font-size: 1rem;
        width: 1.3rem;
        text-align: center;
        flex-shrink: 0;
    }
    .sidebar-model-card .mc-info { flex: 1; min-width: 0; }
    .sidebar-model-card .mc-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6878a8;
        line-height: 1.1;
    }
    .sidebar-model-card .mc-value {
        font-size: 0.78rem;
        font-family: "JetBrains Mono", "Consolas", monospace;
        color: #c0ccee;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sidebar-model-card .mc-divider {
        border: none;
        border-top: 1px solid rgba(80, 100, 180, 0.15);
        margin: 0.45rem 0;
    }

    /* Status dot */
    .status-dot-connected { color: #3ddc84; font-size: 0.65rem; }
    .status-dot-offline    { color: #ff5f5f; font-size: 0.65rem; }

    /* ══════════════════════════════════════════════════════════
       METRIC LABELS (truncation fix)
    ══════════════════════════════════════════════════════════ */
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
        font-size: 0.7rem !important;
        line-height: 1.3 !important;
        color: #8898c8 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #dde4ff !important;
    }

    /* ══════════════════════════════════════════════════════════
       SECTION DIVIDER
    ══════════════════════════════════════════════════════════ */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(80, 90, 150, 0.2);
        margin: 1.6rem 0 1.1rem;
    }

    /* ══════════════════════════════════════════════════════════
       SIDEBAR GLOBAL
    ══════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: rgba(10, 11, 18, 0.92) !important;
        border-right: 1px solid rgba(60, 70, 120, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown hr { margin: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached Model & Health Check Resources ---
@st.cache_resource(show_spinner="Loading Extractive Model (DistilBERT)...")
def get_extractive_model(model_name: str = EXTRACTIVE_MODEL_NAME) -> ExtractiveQAModel:
    """Initializes and caches the extractive QA model in memory."""
    return ExtractiveQAModel(model_name=model_name)


@st.cache_data(ttl=60)
def check_provider_status(
    provider: str, base_url: str, model_name: str
) -> Tuple[bool, str, List[str]]:
    """Cache provider connectivity checks without exposing API keys in cache keys."""
    if provider == "groq":
        api_key = _get_setting("GROQ_API_KEY")
        if not api_key.strip():
            return (
                False,
                "GROQ_API_KEY must be configured in .streamlit/secrets.toml or environment variables.",
                [],
            )
        gen = GenerativeQAModel(
            provider="groq",
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
    else:
        gen = GenerativeQAModel(
            provider="ollama", base_url=base_url, model_name=model_name
        )
    return gen.check_connection()


def _answer_card(answer: str, variant: str) -> None:
    """Render a styled answer card. variant: 'ext' or 'gen'."""
    css_class = f"answer-card-{variant}"
    label = "Extracted Span" if variant == "ext" else "Generated Answer"
    # Escape HTML special chars in the answer text
    safe = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""<div class="{css_class}">
              <div class="label">{label}</div>
              <div class="text">{safe}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def main():
    provider = _get_setting("QA_PROVIDER", "ollama").strip().lower() or "ollama"
    if provider not in {"ollama", "groq"}:
        st.error("Invalid QA_PROVIDER configuration. Set it to 'ollama' or 'groq'.")
        st.stop()

    groq_api_key = _get_setting("GROQ_API_KEY")
    groq_model_name = _get_setting("GROQ_MODEL", GROQ_MODEL_NAME)
    groq_key_missing = provider == "groq" and not groq_api_key.strip()

    # ── Title & subtitle ──────────────────────────────────────────
    st.title("⚖️ Extractive vs Generative Question Answering")
    st.markdown(
        "Compare **extractive** span-based models against **generative** LLMs. "
        "Inspect response latency, span confidence, lexical divergence, and answerability agreement on identical contexts."
    )
    if groq_key_missing:
        st.error(
            "GROQ_API_KEY must be configured in `.streamlit/secrets.toml` or "
            "environment variables. Extractive QA remains available."
        )

    # ═════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═════════════════════════════════════════════════════════════
    sb = st.sidebar
    sb.markdown("### ⚖️ P31 · QA Comparison")

    # ── 1. Compact model + status card ───────────────────────────
    # Persist provider-specific model settings alongside the editable Ollama URL.
    if "ollama_url" not in st.session_state:
        st.session_state["ollama_url"] = OLLAMA_BASE_URL
    if st.session_state.get("gen_provider") != provider:
        st.session_state.pop("latest_comparison", None)
        if (
            "gen_provider" in st.session_state
            or "gen_model_name" not in st.session_state
            or provider == "groq"
        ):
            st.session_state["gen_model_name"] = (
                groq_model_name if provider == "groq" else GENERATIVE_MODEL_NAME
            )
        st.session_state["gen_provider"] = provider
        model_widget_key = "_adv_groq_model" if provider == "groq" else "_adv_gen_model"
        st.session_state[model_widget_key] = st.session_state["gen_model_name"]

    ollama_url = st.session_state["ollama_url"]
    generative_model_name = st.session_state["gen_model_name"]
    active_provider = "Local Ollama" if provider == "ollama" else "Groq Cloud"
    provider_base_url = ollama_url if provider == "ollama" else GROQ_BASE_URL

    is_connected, status_msg, available_models = check_provider_status(
        provider, provider_base_url, generative_model_name
    )

    model_missing = is_connected and "was not found" in status_msg.lower()
    provider_ready = is_connected and not model_missing
    dot_cls = "status-dot-connected" if provider_ready else "status-dot-offline"
    dot_char = "●"
    dot_text = "Connected" if provider_ready else "Model unavailable" if model_missing else "Offline"

    sb.markdown(
        f"""
        <div class="sidebar-model-card">
          <div class="mc-row">
            <span class="mc-icon">🎯</span>
            <div class="mc-info">
              <div class="mc-label">Extractive</div>
              <div class="mc-value">{EXTRACTIVE_MODEL_NAME}</div>
            </div>
          </div>
          <hr class="mc-divider">
          <div class="mc-row">
            <span class="mc-icon">🧠</span>
            <div class="mc-info">
              <div class="mc-label">Generative · {active_provider}</div>
              <div class="mc-value">{generative_model_name}</div>
            </div>
          </div>
          <hr class="mc-divider">
          <div class="mc-row">
            <span class="mc-icon"><span class="{dot_cls}">{dot_char}</span></span>
            <div class="mc-info">
              <div class="mc-label">{active_provider} Status</div>
              <div class="mc-value">{dot_text}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Refresh button — sits just below the card, flush right
    _rcol1, _rcol2 = sb.columns([4, 1])
    with _rcol2:
        if st.button("🔄", help="Re-check generative provider connection status"):
            check_provider_status.clear()
            st.rerun()
    if not provider_ready:
        sb.caption(f"⚠️ {status_msg}")

    sb.markdown("---")

    # ── 2. Sample Scenarios ───────────────────────────────────────
    with sb.expander("💡 Sample Scenarios", expanded=True):
        preset_titles = ["-- Custom / Blank --"] + [p["title"] for p in SAMPLE_PRESETS]
        selected_preset = st.selectbox(
            "Load a preset",
            preset_titles,
            label_visibility="collapsed",
        )

    # ── 3. Advanced Settings (collapsed) ─────────────────────────
    with sb.expander("⚙️ Advanced Settings", expanded=False):
        if provider == "ollama":
            st.caption("Ollama endpoint & generation parameters")
            new_url = st.text_input(
                "Ollama Base URL",
                value=st.session_state["ollama_url"],
                help="OpenAI-compatible Ollama endpoint.",
                key="_adv_ollama_url",
            )
            new_model = st.text_input(
                "Generative Model Tag",
                value=st.session_state["gen_model_name"],
                help="Local Ollama model tag, e.g. qwen3:8b.",
                key="_adv_gen_model",
            )
            if new_url != st.session_state["ollama_url"] or new_model != st.session_state["gen_model_name"]:
                st.session_state["ollama_url"] = new_url
                st.session_state["gen_model_name"] = new_model
                check_provider_status.clear()
                st.rerun()
        else:
            st.caption("Groq model & generation parameters")
            new_model = st.text_input(
                "Groq Model Tag",
                value=st.session_state["gen_model_name"],
                help="Hosted Groq model identifier.",
                key="_adv_groq_model",
            )
            if new_model != st.session_state["gen_model_name"]:
                st.session_state["gen_model_name"] = new_model
                check_provider_status.clear()
                st.rerun()

        st.markdown("---")
        temperature = st.slider(
            "Generative Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.05,
            help="Lower values make the generative model more deterministic and factual.",
        )
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=128,
            max_value=2048,
            value=DEFAULT_MAX_TOKENS,
            step=128,
            help="Maximum generation tokens for the active LLM.",
        )
        max_ans_len = st.slider(
            "Max Extractive Span (tokens)",
            min_value=10,
            max_value=100,
            value=MAX_ANSWER_LENGTH,
            step=5,
            help="Maximum allowable token span for the extractive model.",
        )

    # ═════════════════════════════════════════════════════════════
    # SESSION STATE – preset management
    # ═════════════════════════════════════════════════════════════
    if "current_preset" not in st.session_state:
        st.session_state["current_preset"] = preset_titles[0]
        st.session_state["passage_input"] = ""
        st.session_state["question_input"] = ""

    if selected_preset != st.session_state["current_preset"]:
        st.session_state["current_preset"] = selected_preset
        if selected_preset == "-- Custom / Blank --":
            st.session_state["passage_input"] = ""
            st.session_state["question_input"] = ""
        else:
            preset = next(p for p in SAMPLE_PRESETS if p["title"] == selected_preset)
            st.session_state["passage_input"] = preset["passage"]
            st.session_state["question_input"] = preset["question"]
        st.rerun()

    # ═════════════════════════════════════════════════════════════
    # SECTION 1 – INPUT
    # ═════════════════════════════════════════════════════════════
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("1. Context & Question")

    passage_text = st.text_area(
        "Context Passage",
        value=st.session_state["passage_input"],
        height=165,
        placeholder="Paste or type a context paragraph here...",
        help="Passages longer than 512 tokens are automatically segmented using sliding-window chunking (stride=128).",
    )
    st.session_state["passage_input"] = passage_text

    # Passage stats / long-passage notice
    word_count = len(passage_text.split()) if passage_text.strip() else 0
    char_count = len(passage_text)
    if word_count > 400:
        st.info(
            f"📄 Long passage detected ({word_count} words / {char_count} chars). "
            "Sliding-window chunking will process this across multiple overlapping segments.",
        )
    elif passage_text.strip():
        st.caption(f"Context: {word_count} words · {char_count} characters")

    question_text = st.text_input(
        "Question",
        value=st.session_state["question_input"],
        placeholder="e.g., How much lunar material did Armstrong and Aldrin collect?",
    )
    st.session_state["question_input"] = question_text

    col_run, col_clr, _ = st.columns([1.6, 1, 4])
    with col_run:
        run_button = st.button("🔍 Compare Models", type="primary", use_container_width=True)
    with col_clr:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["passage_input"] = ""
            st.session_state["question_input"] = ""
            st.session_state["current_preset"] = "-- Custom / Blank --"
            st.session_state.pop("latest_comparison", None)
            st.rerun()

    # ═════════════════════════════════════════════════════════════
    # SECTION 2 – MODEL RESPONSES
    # ═════════════════════════════════════════════════════════════
    if run_button:
        if not passage_text.strip() or not question_text.strip():
            st.session_state.pop("latest_comparison", None)
            if not passage_text.strip():
                st.warning("⚠️ Please provide a context passage before running.")
            if not question_text.strip():
                st.warning("⚠️ Please provide a question before running.")
            return

        st.session_state.pop("latest_comparison", None)
        try:
            ext_model = get_extractive_model(EXTRACTIVE_MODEL_NAME)
        except Exception as e:
            st.error(f"❌ Failed to load Extractive QA model: {e}")
            return

        with st.spinner("Extracting answer span..."):
            ext_res: ExtractiveResult = ext_model.answer_question(
                passage=passage_text,
                question=question_text,
                max_answer_length=max_ans_len,
            )

        if groq_key_missing:
            gen_res = GenerativeResult(
                answer="",
                latency_s=0.0,
                success=False,
                error="GROQ_API_KEY must be configured in .streamlit/secrets.toml or environment variables.",
                model_name=generative_model_name,
            )
        else:
            if provider == "ollama":
                gen_model = GenerativeQAModel(
                    provider="ollama",
                    base_url=ollama_url,
                    model_name=generative_model_name,
                )
            else:
                gen_model = GenerativeQAModel(
                    provider="groq",
                    base_url=GROQ_BASE_URL,
                    model_name=generative_model_name,
                    api_key=groq_api_key,
                )
            with st.spinner("Generating synthesized answer..."):
                gen_res = gen_model.answer_question(
                    passage=passage_text,
                    question=question_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        comp = None
        if ext_res.success and gen_res.success:
            comp = compare_answers(
                extractive_ans=ext_res.answer,
                generative_ans=gen_res.answer,
                extractive_latency=ext_res.latency_s,
                generative_latency=gen_res.latency_s,
                extractive_success=ext_res.success,
                generative_success=gen_res.success,
            )

        st.session_state["latest_comparison"] = {
            "passage": passage_text,
            "question": question_text,
            "ext_res": ext_res,
            "gen_res": gen_res,
            "comp": comp,
            "provider": provider,
            "active_provider": active_provider,
            "model_name": generative_model_name,
        }

    latest_run = st.session_state.get("latest_comparison")
    if latest_run is not None and (
        not passage_text.strip()
        or not question_text.strip()
        or latest_run["passage"] != passage_text
        or latest_run["question"] != question_text
    ):
        st.session_state.pop("latest_comparison", None)
        latest_run = None

    if latest_run is None:
        return

    ext_res = latest_run["ext_res"]
    gen_res = latest_run["gen_res"]
    comp = latest_run["comp"]
    result_provider = latest_run["active_provider"]
    result_model_name = latest_run["model_name"]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("2. Model Responses")

    col_ext, col_gen = st.columns(2, gap="large")

    # ── Extractive QA ─────────────────────────────────────────────
    with col_ext:
        st.markdown("### 🎯 Extractive QA")
        st.markdown(
            f'<span class="model-badge">{EXTRACTIVE_MODEL_NAME}</span>',
            unsafe_allow_html=True,
        )

        if not ext_res.success:
            st.error(f"Extractive QA Error: {ext_res.error}")
        else:
            _answer_card(ext_res.answer, "ext")

            m1, m2, m3 = st.columns(3)
            m1.metric("⏱️ Latency (s)", f"{ext_res.latency_s:.3f}")
            m2.metric(
                "📊 Span Confidence",
                f"{ext_res.confidence:.1%}",
                help=(
                    "Softmax probability product for the extracted span. "
                    "NOTE: Uncalibrated heuristic score — not a correctness probability."
                ),
            )
            m3.metric("📑 Chunks", f"{ext_res.num_chunks}")

            excerpt_html = _format_excerpt_html(latest_run["passage"], ext_res)
            if excerpt_html:
                st.markdown("**Supporting Context Span:**")
                st.markdown(
                    f'<div class="excerpt-box">{excerpt_html}</div>',
                    unsafe_allow_html=True,
                )
                if (
                    isinstance(ext_res.start_char, int)
                    and isinstance(ext_res.end_char, int)
                    and 0 <= ext_res.start_char < ext_res.end_char <= len(latest_run["passage"])
                ):
                    st.caption(
                        f"Character range: `[{ext_res.start_char} : {ext_res.end_char}]`"
                    )

    # ── Generative QA ─────────────────────────────────────────────
    with col_gen:
        st.markdown("### 🧠 Generative QA")
        st.markdown(
            f'<span class="model-badge">{result_model_name} &nbsp;·&nbsp; {result_provider}</span>',
            unsafe_allow_html=True,
        )
        if not gen_res.success:
            st.error(f"Generative QA Error: {gen_res.error}")
            if latest_run["provider"] == "ollama" and not is_connected:
                st.info("💡 Ensure Ollama is running (`ollama serve`) and the model is downloaded.")
        else:
            _answer_card(gen_res.answer, "gen")

            g1, g2 = st.columns(2)
            g1.metric("⏱️ Latency (s)", f"{gen_res.latency_s:.3f}")
            g2.metric("🔤 Word Count", f"{len(gen_res.answer.split())}")

    # ═════════════════════════════════════════════════════════════
    # SECTION 3 – COMPARISON
    # ═════════════════════════════════════════════════════════════
    if comp is None:
        st.warning(
            "⚠️ Inter-model comparison could not be performed because one or both models encountered an error."
        )
        return

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("3. Inter-Model Comparison")

    # Agreement status banner
    _status_map = {
        AgreementStatus.STRONG_AGREEMENT:  ("success", "✅"),
        AgreementStatus.PARTIAL_AGREEMENT: ("warning", "⚠️"),
        AgreementStatus.DISAGREEMENT:      ("error",   "🚨"),
        AgreementStatus.UNANSWERABLE:      ("info",    "ℹ️"),
    }
    fn_name, icon = _status_map.get(comp.status, ("error", "❌"))
    getattr(st, fn_name)(f"{icon} **{comp.status.value}** — {comp.summary}")

    st.caption(
        "ℹ️ *Model disagreement reflects divergent phrasing, scope, or factual claims. "
        "Disagreement is not proof of hallucination — verify both outputs against the original context.*"
    )

    # Comparative metric row – four columns with consistent labels
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Token F1 (inter-model)",
        f"{comp.f1_score:.1%}",
        help=(
            "Harmonic mean of directional Precision and Recall between the two outputs. "
            "Measures inter-model lexical similarity — NOT ground-truth correctness."
        ),
    )
    c2.metric(
        "Exact Match",
        "Yes" if comp.exact_match else "No",
        help="Whether the two normalized output strings are identical.",
    )
    c3.metric(
        "Span Contained?",
        "Yes" if comp.is_contained else "No",
        help="Whether the extractive span appears verbatim within the generative response.",
    )
    c4.metric(
        "Extractive Speedup",
        f"{comp.speedup_ratio:.1f}×",
        delta=f"−{comp.latency_difference_s:.2f}s",
        delta_color="normal",
        help="Ratio of generative latency to extractive latency.",
    )

    # Lexical overlap details expander
    with st.expander("🔬 Lexical Overlap & Directional Token Metrics", expanded=True):
        det1, det2, det3 = st.columns(3)

        with det1:
            st.markdown("**Shared Tokens**")
            if comp.shared_tokens:
                st.write(", ".join([f"`{t}`" for t in comp.shared_tokens]))
            else:
                st.caption("None")

        with det2:
            st.markdown("**Extractive-Only Tokens**")
            if comp.extractive_only_tokens:
                st.write(", ".join([f"`{t}`" for t in comp.extractive_only_tokens]))
            else:
                st.caption("None")

        with det3:
            st.markdown("**Generative-Only Tokens**")
            if comp.generative_only_tokens:
                st.write(", ".join([f"`{t}`" for t in comp.generative_only_tokens]))
            else:
                st.caption("None")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"- **Extractive answer length:** {comp.extractive_word_count} words  \n"
                f"- **Generative answer length:** {comp.generative_word_count} words  \n"
                f"- **Token Jaccard Index:** {comp.jaccard_similarity:.1%}"
            )
        with col_b:
            st.markdown(
                f"- **Extractive Span Recall:** {comp.recall:.1%}  \n"
                f"  *(fraction of extractive tokens in generative output)*  \n"
                f"- **Generative Overlap Precision:** {comp.precision:.1%}  \n"
                f"  *(fraction of generative tokens in extractive output)*"
            )

        st.caption(
            "*The metrics above evaluate relative lexical overlap between the two models. "
            "They do not indicate benchmark accuracy against a labeled ground-truth dataset.*"
        )


if __name__ == "__main__":
    main()
