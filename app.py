import streamlit as st
import spacy
from pathlib import Path
import json
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.rule_based import detect_clauses_spacy, predict_bio_tags
from src.clause_labeler import get_clause_spans


st.set_page_config(
    page_title="Clause Boundary Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark/Neon Premium Aesthetics */
    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15) 0%, transparent 50%),
                          radial-gradient(circle at 85% 30%, rgba(29, 78, 216, 0.15) 0%, transparent 50%);
        color: #e2e8f0;
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.39); transform: scale(1); }
        50% { box-shadow: 0 4px 25px 5px rgba(139, 92, 246, 0.6); transform: scale(1.01); }
        100% { box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.39); transform: scale(1); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes scaleIn {
        0% { transform: scale(0.95); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Titles */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #a855f7, #3b82f6, #2dd4bf, #a855f7);
        background-size: 200% auto;
        animation: gradientShift 6s linear infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        animation: fadeInUp 0.8s ease-out forwards;
    }

    /* Boxed Clauses (Tokens) */
    .clause-box {
        display: inline-block;
        padding: 6px 14px;
        margin: 6px 4px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
        cursor: default;
        animation: scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    .clause-box:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.6);
        filter: brightness(1.3);
        z-index: 10;
        position: relative;
    }

    .clause-0 { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); color: #93c5fd; box-shadow: 0 0 15px rgba(59,130,246,0.1); }
    .clause-1 { background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); color: #d8b4fe; box-shadow: 0 0 15px rgba(168,85,247,0.1); }
    .clause-2 { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; box-shadow: 0 0 15px rgba(16,185,129,0.1); }
    .clause-3 { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #fcd34d; box-shadow: 0 0 15px rgba(245,158,11,0.1); }
    .clause-4 { background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.4); color: #f9a8d4; box-shadow: 0 0 15px rgba(236,72,153,0.1); }
    .clause-5 { background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); color: #7dd3fc; box-shadow: 0 0 15px rgba(14,165,233,0.1); }

    /* Metric Cards Glassmorphism */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        animation: fadeInUp 0.6s ease-out forwards;
    }

    .metric-card:hover { 
        border-color: rgba(168, 85, 247, 0.6); 
        box-shadow: 0 15px 30px -5px rgba(168, 85, 247, 0.25);
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* BIO Tags */
    .tag-bio {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        margin: 1px 4px 1px 0px;
        vertical-align: middle;
        letter-spacing: 0.05em;
    }

    .tag-b { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .tag-i { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
    .tag-o { background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

    /* Inputs & Buttons */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }

    /* Streamlit's primary button overrides */
    div.stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        background-size: 200% auto !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.39) !important;
        animation: pulseGlow 3s infinite alternate !important;
    }

    div.stButton > button:hover {
        background-position: right center !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.8) !important;
        filter: brightness(1.2);
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Info Box */
    .info-box {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8b5cf6;
        border-radius: 4px 8px 8px 4px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        color: #e2e8f0;
        backdrop-filter: blur(8px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important;
    }

</style>
""", unsafe_allow_html=True)


# ==================== Helper Functions ====================

@st.cache_resource
def load_spacy_model():
    """Load spaCy model (cached)."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        return None

def prepare_text_for_models(text: str):
    """
    Parses text via spaCy and structures it into sentence tuples for ML inference.
    Returns: list of sentences formatted as list of (word, upos, deprel, xpos, head_dist, "O")
    """
    nlp = load_spacy_model()
    if not nlp: return []
    doc = nlp(text)
    
    parsed_sents = []
    for sent in doc.sents:
        sent_data = []
        for token in sent:
            try:
                head_dist = abs(token.i - token.head.i)
            except Exception:
                head_dist = 0
            # Add dummy "O" label which is required by some feature extractors or pipelines initially
            sent_data.append((token.text, token.pos_, token.dep_, token.tag_, head_dist, "O"))
        parsed_sents.append(sent_data)
    return parsed_sents


def execute_ml_model_inference(model, text: str, model_type: str):
    """
    Executes CRF or BiLSTM inference accurately and returns (clauses, bio_results).
    """
    parsed_sents = prepare_text_for_models(text)
    if not parsed_sents:
        return [], []
        
    bio_results = []
    
    # 1. Infer Labels
    if model_type == "CRF":
        # model.predict takes list of sentences
        predictions = model.predict(parsed_sents)
    elif model_type == "BiLSTM":
        # dataset format naturally accepts the same tuples
        predictions = model.predict(parsed_sents)
    else:
        return [], []
        
    # 2. Format to flat BIO list [(word, POS, label), ...]
    flat_bio_tuples = []
    for sent_idx, sent_data in enumerate(parsed_sents):
        preds = predictions[sent_idx]
        for token_idx, token_tuple in enumerate(sent_data):
            word, upos = token_tuple[0], token_tuple[1]
            label = preds[token_idx]
            flat_bio_tuples.append((word, upos, label))
            bio_results.append((word, label))
            
    # 3. Retrieve clause spans
    clauses_raw = get_clause_spans(flat_bio_tuples)
    
    # Map back to dict format identical to rule-based for rendering parity
    mapped_clauses = []
    for item in clauses_raw:
        mapped_clauses.append({
            "text": item.get('text', ''),
            "type": "AI-detected",
            "tokens": item.get('text', '').split()
        })
        
    return mapped_clauses, bio_results


def render_clauses_html(clauses, original_text=""):
    """Render clauses as colorful HTML boxes."""
    if not clauses:
        return "<p style='color: #64748b; font-style: italic;'>No clauses detected.</p>"

    html = '<div style="line-height: 2.4; margin: 1rem 0;">'
    for i, clause in enumerate(clauses):
        color_class = f"clause-{i % 6}"
        clause_type = clause.get("type", "Clause")
        text = clause.get("text", "")
        html += f'<span class="clause-box {color_class}" title="{clause_type}">{text}</span> '
    html += '</div>'
    return html


def render_bio_tags_html(bio_results):
    """Render BIO tags as colored inline badges."""
    if not bio_results:
        return ""

    html = '<div style="line-height: 2.8; margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">'
    for word, tag in bio_results:
        if tag == "B-CLAUSE":
            tag_class = "tag-b"
        elif tag == "I-CLAUSE":
            tag_class = "tag-i"
        else:
            tag_class = "tag-o"
            
        html += f'<div style="display: inline-block; margin: 2px 4px; text-align: center;">'
        html += f'<div style="font-size: 0.95rem; color: #f1f5f9; margin-bottom: 2px;">{word}</div>'
        html += f'<div class="tag-bio {tag_class}">{tag}</div>'
        html += f'</div>'
    html += '</div>'
    return html


def load_saved_results():
    """Load saved evaluation results if available."""
    results = {}
    results_dir = Path("results/metrics")
    if results_dir.exists():
        for fp in results_dir.glob("*.json"):
            if "comparison.json" in fp.name:
                continue
            with open(fp) as f:
                results[fp.stem] = json.load(f)
    return results


# ==================== Page Definitions ====================

def page_clause_detector():
    """Main clause detection page."""
    st.markdown('<h1 class="hero-title">Clause Boundary Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">High-precision linguistic chunking using Deep Learning & Rules</p>', unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area(
            "Input text sequence:",
            value="While the deep neural network learned the abstract features, the traditional algorithm processed the grammar rules explicitly.",
            height=130,
            key="input_text",
            label_visibility="collapsed"
        )
    with col2:
        model_choice = st.selectbox(
            "Inference Engine:",
            ["Rule-based (spaCy)", "CRF Model", "BiLSTM Model"],
            key="model_select",
        )
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        detect_btn = st.button("🚀 Run Analysis", use_container_width=True)

    if detect_btn and text.strip():
        with st.spinner("Processing Neural Pathways..."):
            if model_choice == "Rule-based (spaCy)":
                clauses = detect_clauses_spacy(text)
                bio_results = predict_bio_tags(text)
            else:
                try:
                    if model_choice == "CRF Model":
                        from src.crf_model import CRFClauseDetector
                        model = CRFClauseDetector()
                        model.load("results/models/crf_model.pkl")
                        st.success("Loaded weights: Conditional Random Field")
                        clauses, bio_results = execute_ml_model_inference(model, text, "CRF")
                        
                    elif model_choice == "BiLSTM Model":
                        from src.bilstm_model import BiLSTMTrainer
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = BiLSTMTrainer(device=device)
                        model.load("results/models/bilstm")
                        st.success("Loaded weights: Bidirectional LSTM")
                        clauses, bio_results = execute_ml_model_inference(model, text, "BiLSTM")
                        
                except Exception as e:
                    st.error(f"Inference Engine failed loading weights. Is it fully trained? Falling back. Error: {e}")
                    clauses = detect_clauses_spacy(text)
                    bio_results = predict_bio_tags(text)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)

        st.markdown("### <span style='color: #a855f7;'>▶</span> Semantic Clause Extraction", unsafe_allow_html=True)
        html = render_clauses_html(clauses, text)
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(clauses)}</div>
                <div class="metric-label">Nodes Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            num_tokens = len(bio_results)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_tokens}</div>
                <div class="metric-label">Vectors Processed</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            clause_types = set(c.get("type", "unknown") for c in clauses)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(clause_types)}</div>
                <div class="metric-label">Grammar Structures</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>### <span style='color: #3b82f6;'>▶</span> Sequence Tokenization", unsafe_allow_html=True)
        bio_html = render_bio_tags_html(bio_results)
        st.markdown(bio_html, unsafe_allow_html=True)


def page_dependency_tree():
    """Dependency tree visualization page."""
    st.markdown('<h1 class="hero-title">Dependency Viewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Map the syntactic structure underpinning clause grammar</p>', unsafe_allow_html=True)

    nlp = load_spacy_model()
    if nlp is None: return

    text = st.text_input(
        "Generate Graph for:",
        value="He realized that the machine worked flawlessly.",
        key="dep_text",
    )

    if text.strip():
        doc = nlp(text)

        st.markdown("### <span style='color: #2dd4bf;'>▶</span> Dependency Extractor", unsafe_allow_html=True)
        from spacy import displacy
        html = displacy.render(doc, style="dep", options={
            "compact": True,
            "bg": "transparent",
            "color": "#cbd5e1",
            "font": "Inter",
        })
        html = html.replace('fill: currentColor', 'fill: #94a3b8')
        st.markdown(f'<div style="overflow-x: auto; background: rgba(15,23,42,0.6); '
                    f'border: 1px solid rgba(255,255,255,0.05); '
                    f'border-radius: 16px; padding: 1.5rem; box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.2);">{html}</div>',
                    unsafe_allow_html=True)

        st.markdown("### <span style='color: #a855f7;'>▶</span> Subordinate Roots Discovered", unsafe_allow_html=True)
        clause_rels = {"advcl", "ccomp", "xcomp", "acl", "relcl", "parataxis", "csubj"}
        found = []
        for token in doc:
            if token.dep_ in clause_rels:
                found.append({
                    "Word": token.text,
                    "Identifier (Relation)": token.dep_,
                    "Parent Head": token.head.text,
                })

        if found:
            st.table(found)
        else:
            st.markdown('<div class="info-box">Zero subordinate markers detected within this structural graph.</div>',
                       unsafe_allow_html=True)


def page_model_comparison():
    """Model comparison page."""
    st.markdown('<h1 class="hero-title">Architecture Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Live Head-to-Head Multi-Agent Evaluation</p>', unsafe_allow_html=True)

    text = st.text_input(
        "Inject evaluation prompt:",
        value="Even though he was tired, he drove home because it was late.",
        key="compare_text",
    )

    if text.strip() and st.button("⚡ Execute Global Comparison", key="compare_btn"):
        with st.spinner("Instantiating AI Models in parallel..."):
            
            # --- Rule Based ---
            rule_clauses = detect_clauses_spacy(text)
            rule_bio = predict_bio_tags(text)
            
            # --- CRF ---
            try:
                from src.crf_model import CRFClauseDetector
                crf_model = CRFClauseDetector()
                crf_model.load("results/models/crf_model.pkl")
                crf_clauses, crf_bio = execute_ml_model_inference(crf_model, text, "CRF")
            except Exception:
                crf_clauses, crf_bio = None, None
                
            # --- BiLSTM ---
            try:
                from src.bilstm_model import BiLSTMTrainer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                bilstm_model = BiLSTMTrainer(device=device)
                bilstm_model.load("results/models/bilstm")
                bilstm_clauses, bilstm_bio = execute_ml_model_inference(bilstm_model, text, "BiLSTM")
            except Exception:
                bilstm_clauses, bilstm_bio = None, None

            st.markdown("<br>", unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["📐 Rule-Based Algorithms", "📊 Conditional Random Fields (ML)", "🧠 BiLSTM Engine (DL)"])
            
            with t1:
                st.markdown("### <span style='color: #60a5fa;'>▍</span> Explicit Rules Engine", unsafe_allow_html=True)
                st.markdown(render_clauses_html(rule_clauses), unsafe_allow_html=True)
                with st.expander("Show BIO Arrays"):
                    st.markdown(render_bio_tags_html(rule_bio), unsafe_allow_html=True)

            with t2:
                st.markdown("### <span style='color: #a855f7;'>▍</span> CRF Statistical Inference", unsafe_allow_html=True)
                if crf_clauses:
                    st.markdown(render_clauses_html(crf_clauses), unsafe_allow_html=True)
                    with st.expander("Show BIO Arrays"):
                        st.markdown(render_bio_tags_html(crf_bio), unsafe_allow_html=True)
                else:
                    st.error("Engine Data Missing: CRF requires pre-training")

            with t3:
                st.markdown("### <span style='color: #34d399;'>▍</span> BiLSTM Neural Execution", unsafe_allow_html=True)
                if bilstm_clauses is not None:
                    st.markdown(render_clauses_html(bilstm_clauses), unsafe_allow_html=True)
                    with st.expander("Show BIO Arrays"):
                        st.markdown(render_bio_tags_html(bilstm_bio), unsafe_allow_html=True)
                else:
                    st.error("Engine Data Missing: BiLSTM requires pre-training")


def page_about():
    """About / Info page."""
    st.markdown('<h1 class="hero-title">System Metrics</h1>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3 style='margin-top: 0;'>Deep Learning Semantic Parsing</h3>
        <p style='margin-bottom: 0;'>A comparative Natural Language Processing suite benchmarking deterministic extraction versus deep neural sequence-to-sequence mappings utilizing Universal Dependencies.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1 style='font-size: 3rem; margin:0;'>📐</h1>
            <div class="metric-label" style='color: #e2e8f0; font-weight:800; margin-top: 10px;'>DETERMINISTIC</div>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Linear syntactic boundary mapping via standard dependency graph traversal.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h1 style='font-size: 3rem; margin:0;'>📊</h1>
            <div class="metric-label" style='color: #e2e8f0; font-weight:800; margin-top: 10px;'>STATISTICAL ML</div>
            <p style='color: #94a3b8; font-size: 0.85rem;'>CRF engine extrapolating boundaries via conditional ±2 localized token probability.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h1 style='font-size: 3rem; margin:0;'>🧠</h1>
            <div class="metric-label" style='color: #e2e8f0; font-weight:800; margin-top: 10px;'>BiLSTM DL</div>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Dual-Embedding Neural mapping isolating profound grammatical structure context.</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== Main App ====================

def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<h2 style="color: #cbd5e1; font-weight:800; margin-bottom: 1rem;">⚙️ DASHBOARD</h2>', unsafe_allow_html=True)
        page = st.radio(
            "Go to:",
            [
                "🔍 Clause Engine",
                "📊 Architecture Comparison",
                "🌳 Dependency Viewer",
                "ℹ️ System Metrics",
            ],
            key="nav",
            label_visibility='collapsed'
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #64748b; font-size: 0.75rem; text-align: center;">
            <p><strong>Powered by PyTorch & CRF</strong></p>
            <p style='margin-top: -10px;'>Universal Dependencies Corpus</p>
            <p style='margin-top: 5px; color: #8b5cf6;'>V2.0 AI Upgrade</p>
        </div>
        """, unsafe_allow_html=True)

    # Route to the selected page
    if page == "🔍 Clause Engine":
        page_clause_detector()
    elif page == "🌳 Dependency Viewer":
        page_dependency_tree()
    elif page == "📊 Architecture Comparison":
        page_model_comparison()
    elif page == "ℹ️ System Metrics":
        page_about()


if __name__ == "__main__":
    main()
