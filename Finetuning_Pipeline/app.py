# ====================
# NEW: MedVision Frontend — Professional Edition
# ====================
#
# Run from the Finetuning_pipeline/ folder:
#   streamlit run app.py
#
# Expects api.py to be running:
#   python api.py                          # local MedGemma (default)
#   python api.py --vlm_backend ollama     # Ollama backend

import streamlit as st
import requests
import io
import json
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="MedVision Retrieval AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:          #f0f4f8;
    --surface:     #ffffff;
    --surface2:    #f8fafc;
    --border:      #dde3ec;
    --border-mid:  #c8d3e0;
    --navy:        #0f2547;
    --navy-mid:    #1a3a6b;
    --navy-light:  #e8eef7;
    --gold:        #b5873a;
    --gold-light:  #d4a84e;
    --gold-bg:     #fdf6e8;
    --gold-border: #e8d09a;
    --teal:        #0d7a72;
    --teal-light:  #e6f4f3;
    --teal-border: #9fd3cf;
    --text:        #1e2d3d;
    --text-mid:    #3d5168;
    --text-muted:  #6b7d93;
    --success:     #0d7a4e;
    --success-bg:  #e8f5ef;
    --warn:        #92580a;
    --warn-bg:     #fef3e2;
    --danger:      #b91c1c;
    --danger-bg:   #fef2f2;
    --font-display: 'Libre Baskerville', Georgia, serif;
    --font-body:    'Outfit', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --radius:       6px;
    --shadow:       0 1px 4px rgba(15,37,71,0.08);
    --shadow-md:    0 4px 16px rgba(15,37,71,0.12);
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }

/* ── Sidebar — dark navy, enterprise style ── */
[data-testid="stSidebar"] {
    background: #0f2547 !important;
    border-right: 1px solid #1a3a6b !important;
}

/* Hide keyboard_double_ text — it is a Material Icons ligature that appears
   as raw text before/when the icon font fails. We make the collapse button's
   span invisible and use overflow:hidden so the icon still renders. */
[data-testid="stSidebarCollapseButton"] {
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] > div,
[data-testid="stSidebarCollapseButton"] svg {
    color: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] .st-emotion-cache-pb6fr7,
[data-testid="stSidebar"] [title*="keyboard"],
[data-testid="stSidebar"] [aria-label*="keyboard"] {
    display: none !important;
}

/* General text on dark sidebar — do NOT include span here to avoid
   overriding input placeholder and button text unintentionally */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div:not([data-testid="stTextInput"]):not(.stTextInput) {
    color: #dce8f5 !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] .mv-sidebar-label {
    color: rgba(255,255,255,0.45) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] span,
[data-testid="stSidebar"] .stSlider [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stSliderTickBarMax"] {
    color: rgba(255,255,255,0.35) !important;
}
[data-testid="stSidebar"] .stCheckbox label { color: #dce8f5 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}

/* Text input — solid dark background at every level, white text */
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stTextInput > div,
[data-testid="stSidebar"] .stTextInput > div > div {
    background: #1a3a6b !important;
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input {
    background: #1a3a6b !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
    caret-color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: rgba(255,255,255,0.4) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.4) !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* Pills on dark sidebar */
[data-testid="stSidebar"] .mv-pill.mv-p-slate {
    background: rgba(255,255,255,0.1) !important;
    color: #dce8f5 !important;
    border-color: rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] .mv-pill.mv-p-online {
    background: rgba(13,122,78,0.3) !important;
    color: #6ee7b7 !important;
    border-color: rgba(13,122,78,0.55) !important;
}
[data-testid="stSidebar"] .mv-pill.mv-p-offline {
    background: rgba(185,28,28,0.3) !important;
    color: #fca5a5 !important;
    border-color: rgba(185,28,28,0.55) !important;
}
[data-testid="stSidebar"] .mv-pill.mv-p-warn {
    background: rgba(146,88,10,0.3) !important;
    color: #fcd34d !important;
    border-color: rgba(146,88,10,0.55) !important;
}
[data-testid="stSidebar"] .mv-pill.mv-p-teal {
    background: rgba(13,122,114,0.3) !important;
    color: #5eead4 !important;
    border-color: rgba(13,122,114,0.55) !important;
}

h1 { font-family: var(--font-display) !important; color: var(--navy) !important;
     font-size: 1.9rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h2 { font-family: var(--font-display) !important; color: var(--navy) !important;
     font-size: 1.05rem !important; font-weight: 700 !important; }
h3 { font-family: var(--font-body) !important; color: var(--text-mid) !important;
     font-size: 0.9rem !important; font-weight: 600 !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--navy);
    color: #ffffff !important;
    border: none;
    border-radius: var(--radius);
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    padding: 0.6rem 1.6rem;
    transition: all 0.18s;
    width: 100%;
    box-shadow: var(--shadow);
}
.stButton > button:hover {
    background: var(--navy-mid) !important;
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

/* Sidebar buttons — ghost outline style, clearly readable on dark navy */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.45) !important;
    box-shadow: none !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.75) !important;
    box-shadow: none !important;
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1.5px dashed var(--border-mid);
    border-radius: var(--radius);
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    box-shadow: var(--shadow);
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-body);
    font-size: 0.83rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    padding: 0.65rem 1.4rem;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom: 2px solid var(--navy) !important;
    font-weight: 600 !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Alerts ── */
.stAlert { font-family: var(--font-body); font-size: 0.82rem; border-radius: var(--radius); }

/* ── Textarea (feedback box) ── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    padding: 0.7rem 0.9rem !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.15s !important;
}
.stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 2px rgba(15,37,71,0.1) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
code {
    background: var(--navy-light);
    color: var(--navy);
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.79rem;
}

/* ── Custom components ── */
.mv-header {
    padding: 1.6rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.mv-logo {
    font-family: var(--font-display);
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.02em;
    line-height: 1;
}
.mv-logo-accent { color: var(--gold); }
.mv-tagline {
    font-family: var(--font-mono);
    font-size: 0.63rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.mv-version {
    font-family: var(--font-mono);
    font-size: 0.63rem;
    color: var(--text-muted);
    text-align: right;
    line-height: 1.7;
}

.mv-disclaimer {
    background: var(--warn-bg);
    border: 1px solid var(--gold-border);
    border-left: 3px solid var(--gold);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.6rem 1rem;
    font-size: 0.74rem;
    color: var(--warn);
    font-family: var(--font-mono);
    letter-spacing: 0.01em;
    margin-bottom: 1.2rem;
}

/* ── Quick-start guide ── */
.mv-quickstart {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow);
}
.mv-quickstart-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.7rem;
}
.mv-quickstart-steps {
    display: flex;
    gap: 0;
    align-items: stretch;
}
.mv-qs-step {
    flex: 1;
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0 1.2rem 0 0;
    border-right: 1px solid var(--border);
    margin-right: 1.2rem;
}
.mv-qs-step:last-child {
    border-right: none;
    margin-right: 0;
    padding-right: 0;
}
.mv-qs-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--navy);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    flex-shrink: 0;
    margin-top: 1px;
}
.mv-qs-text { font-size: 0.8rem; color: var(--text-mid); line-height: 1.5; }
.mv-qs-text strong { color: var(--navy); font-weight: 600; }
.mv-qs-text span   { color: var(--text-muted); font-size: 0.74rem; }

.mv-section-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.mv-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.6rem;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.mv-p-online  { background: var(--success-bg);  color: var(--success); border-color: #9fd3b8; }
.mv-p-offline { background: var(--danger-bg);   color: var(--danger);  border-color: #f5a5a5; }
.mv-p-warn    { background: var(--warn-bg);      color: var(--warn);    border-color: var(--gold-border); }
.mv-p-navy    { background: var(--navy-light);   color: var(--navy);    border-color: #c0cedf; }
.mv-p-teal    { background: var(--teal-light);   color: var(--teal);    border-color: var(--teal-border); }
.mv-p-slate   { background: var(--surface2);     color: var(--text-muted); border-color: var(--border); }
.mv-p-gold    { background: var(--gold-bg);      color: var(--gold);    border-color: var(--gold-border); }

.mv-report-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--navy);
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 1.5rem 1.8rem;
    box-shadow: var(--shadow);
}
.mv-report-card.vlm { border-top-color: var(--teal); }
.mv-report-header {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--navy);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 7px;
}
.mv-report-header.vlm { color: var(--teal); }
.mv-report-text {
    font-family: var(--font-body);
    font-size: 0.88rem;
    line-height: 1.9;
    color: var(--text);
    white-space: pre-wrap;
}

.mv-sim-bar { margin: 0.3rem 0 0; width: 100%; }
.mv-sim-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-muted);
    margin-bottom: 3px;
}
.mv-track { background: var(--border); border-radius: 2px; height: 3px; overflow: hidden; }
.mv-fill  { height: 100%; border-radius: 2px; }

.mv-case-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: border-color 0.18s, box-shadow 0.18s;
}
.mv-case-card:hover {
    border-color: var(--border-mid);
    box-shadow: var(--shadow-md);
}
.mv-case-header {
    padding: 0.45rem 0.7rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface2);
}
.mv-case-footer {
    padding: 0.5rem 0.7rem;
    border-top: 1px solid var(--border);
    background: var(--surface2);
}
.mv-case-caption {
    font-size: 0.71rem;
    line-height: 1.55;
    color: var(--text-muted);
    font-family: var(--font-body);
    margin-top: 0.4rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.mv-query-card {
    background: var(--surface);
    border: 1.5px solid var(--navy);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.mv-query-header {
    padding: 0.5rem 0.8rem;
    background: var(--navy-light);
    border-bottom: 1px solid #c0cedf;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.mv-info-box {
    background: var(--teal-light);
    border: 1px solid var(--teal-border);
    border-radius: var(--radius);
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.6;
}
.mv-info-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--teal);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.mv-step-row {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.28rem 0; font-size: 0.8rem; color: var(--text-muted);
}
.mv-step-num {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted);
    flex-shrink: 0;
}
.mv-step-done   .mv-step-num { background: #e8f5ef; border-color: #9fd3b8; color: #0d7a4e; }
.mv-step-active .mv-step-num { background: var(--gold-bg); border-color: var(--gold-border); color: var(--gold); }
.mv-step-done   { color: var(--text); }
.mv-step-active { color: var(--gold); }

.mv-sidebar-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1rem 0 0.4rem;
}

.mv-conditions {
    background: var(--warn-bg);
    border: 1px solid var(--gold-border);
    border-radius: var(--radius);
    padding: 0.7rem 1rem;
    margin-bottom: 0.8rem;
}
.mv-conditions-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--warn);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.mv-hint {
    background: var(--success-bg);
    border: 1px solid #9fd3b8;
    border-radius: var(--radius);
    padding: 0.6rem 0.9rem;
    margin-top: 1rem;
    font-family: var(--font-mono);
    font-size: 0.66rem;
    color: var(--success);
    letter-spacing: 0.02em;
}

.mv-empty {
    padding: 3.5rem 0;
    text-align: center;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    line-height: 2;
    letter-spacing: 0.04em;
}

.mv-upload-placeholder {
    background: var(--surface);
    border: 1.5px dashed var(--border-mid);
    border-radius: var(--radius);
    padding: 3rem 1rem;
    text-align: center;
    color: var(--text-muted);
}
.mv-upload-placeholder .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.mv-upload-placeholder p {
    font-family: var(--font-mono); font-size: 0.7rem;
    letter-spacing: 0.04em; margin: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULT_BACKEND = "http://localhost:8000"

for _k, _v in [
    ("history", []),
    ("last_result", None),
    ("last_report_result", None),
    ("last_image_bytes", None),
    ("last_image_name", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def sim_bar(score: float) -> str:
    pct = min(max(score * 100, 0), 100)
    if pct > 75:   color, label = "#0d7a4e", "HIGH"
    elif pct > 50: color, label = "#b5873a", "MOD"
    else:          color, label = "#b91c1c", "LOW"
    return (
        f'<div class="mv-sim-bar">'
        f'<div class="mv-sim-label"><span>{pct:.1f}%</span>'
        f'<span style="color:{color};font-weight:600">{label}</span></div>'
        f'<div class="mv-track">'
        f'<div class="mv-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div></div>'
    )

def score_badge(pct: float) -> str:
    if pct > 75:   return "background:#e8f5ef;border:1.5px solid #9fd3b8;color:#0d7a4e"
    elif pct > 50: return "background:#fdf6e8;border:1.5px solid #e8d09a;color:#92580a"
    return "background:#fef2f2;border:1.5px solid #f5a5a5;color:#b91c1c"

def pill(text, cls="mv-p-slate") -> str:
    return f'<span class="mv-pill {cls}">{text}</span>'

def step_html(num, label, state="pending") -> str:
    cls  = {"done": "mv-step-done", "active": "mv-step-active"}.get(state, "")
    icon = "✓" if state == "done" else str(num)
    return (
        f'<div class="mv-step-row {cls}">'
        f'<div class="mv-step-num">{icon}</div>'
        f'<span>{label}</span></div>'
    )

def api_health(url: str) -> dict:
    try:
        r = requests.get(f"{url}/api/health", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_case_image(idx: int, url: str) -> Image.Image | None:
    if idx < 0:
        return None
    try:
        r = requests.get(f"{url}/images/case/{idx}", timeout=10)
        if r.ok:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return None

def do_retrieve(url, img_bytes, fname, top_k, qe, rr, min_score) -> dict | None:
    try:
        r = requests.post(
            f"{url}/api/retrieve",
            files={"image": (fname, img_bytes, "image/jpeg")},
            data={"top_k": top_k, "use_query_expansion": str(qe).lower(),
                  "use_reranking": str(rr).lower(), "min_score": min_score},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Retrieval failed — {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Retrieval error: {e}")
    return None

def do_generate_report(url, img_bytes, fname, method, top_k,
                        min_score, temp, max_tok, num_ex) -> dict | None:
    try:
        r = requests.post(
            f"{url}/api/generate-report",
            files={"image": (fname, img_bytes, "image/jpeg")},
            data={"report_method": method, "top_k": top_k, "min_score": min_score,
                  "temperature": temp, "max_tokens": max_tok, "num_examples": num_ex},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Report generation failed — {e.response.status_code}: {e.response.text}")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The VLM model may need more time.")
    except Exception as e:
        st.error(f"Report error: {e}")
    return None

def do_feedback(url, img_bytes, fname, feedback, caption, qid, method, add_idx) -> dict | None:
    try:
        r = requests.post(
            f"{url}/api/feedback",
            files={"image": (fname, img_bytes, "image/jpeg")},
            data={"feedback": feedback, "generated_caption": caption,
                  "query_id": qid or "", "report_method": method or "",
                  "add_to_index": str(add_idx).lower()},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Feedback error: {e}")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="padding:1rem 0 0.8rem;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:0.2rem;">'
        '<div style="font-family:\'Libre Baskerville\',serif;font-size:1.2rem;font-weight:700;'
        'color:#ffffff;letter-spacing:-0.01em;line-height:1.2;">Med<span style="color:#d4a84e;">Vision</span> Retrieval AI</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.55rem;'
        'color:rgba(255,255,255,0.4);letter-spacing:0.1em;text-transform:uppercase;margin-top:3px;">'
        'Radiology Intelligence Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="mv-sidebar-label">API Endpoint</div>', unsafe_allow_html=True)
    BACKEND_URL = st.text_input("url", value=DEFAULT_BACKEND,
                                 label_visibility="collapsed").rstrip("/")

    health         = api_health(BACKEND_URL)
    is_online      = bool(health)
    pipeline_ready = health.get("pipeline_ready", False)
    vlm_backend    = health.get("vlm_backend", "unknown")
    vlm_cached     = health.get("vlm_model_cached", False)
    vlm_loaded     = health.get("vlm_model_loaded", False)
    ollama_ready   = health.get("ollama_ready", False)

    if is_online:
        st.markdown(
            f'<span class="mv-pill mv-p-online">● Online</span>&nbsp;'
            f'<span class="mv-pill mv-p-slate">{vlm_backend} VLM</span>',
            unsafe_allow_html=True,
        )
        if not pipeline_ready:
            st.warning("Pipeline not initialised.")
    else:
        st.markdown(
            '<span class="mv-pill mv-p-offline">● Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Start api.py at {BACKEND_URL}")

    st.divider()

    st.markdown('<div class="mv-sidebar-label">Retrieval</div>', unsafe_allow_html=True)
    top_k     = st.slider("Similar cases (K)", 1, 10, 3)
    min_score = st.slider("Min similarity", 0.0, 1.0, 0.3, 0.05)
    use_qe    = st.checkbox("Query expansion", value=True)
    use_rr    = st.checkbox("Re-ranking", value=True)

    st.divider()

    st.markdown('<div class="mv-sidebar-label">Report Method</div>', unsafe_allow_html=True)
    report_method = st.selectbox(
        "method",
        ["template", "vlm_few_shot", "vlm_zero_shot",
         "ollama_few_shot", "ollama_zero_shot",
         "weighted", "majority", "concat"],
        label_visibility="collapsed",
        help=(
            "template — Structured FINDINGS/IMPRESSION (instant, no GPU)\n"
            "vlm_few_shot / vlm_zero_shot — Local MedGemma (GPU required)\n"
            "ollama_few_shot / ollama_zero_shot — Ollama remote VLM\n"
            "weighted / majority / concat — Retrieval-based text methods"
        ),
    )

    is_vlm  = report_method.startswith("vlm_") or report_method.startswith("ollama_")
    is_zero = "zero_shot" in report_method

    if is_vlm:
        st.markdown('<div class="mv-sidebar-label">VLM Parameters</div>',
                    unsafe_allow_html=True)
        vlm_temp     = st.slider("Temperature",      0.0, 1.0, 0.3, 0.05)
        vlm_tokens   = st.slider("Max tokens",       128, 1024, 512, 64)
        vlm_examples = st.slider("Few-shot examples", 0, top_k, min(3, top_k),
                                  disabled=is_zero)
    else:
        vlm_temp, vlm_tokens, vlm_examples = 0.3, 512, 3

    # MedGemma controls
    if vlm_backend == "local" and is_online:
        st.divider()
        st.markdown('<div class="mv-sidebar-label">MedGemma Model</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="mv-pill {"mv-p-online" if vlm_cached else "mv-p-warn"}">'
            f'{"✓ Cached" if vlm_cached else "Not cached"}</span>&nbsp;'
            f'<span class="mv-pill {"mv-p-online" if vlm_loaded else "mv-p-slate"}">'
            f'{"✓ Loaded" if vlm_loaded else "Not loaded"}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            if st.button("⬇ Download"):
                with st.spinner("Downloading (~8 GB)…"):
                    try:
                        r = requests.post(f"{BACKEND_URL}/api/vlm/download", timeout=3600)
                        st.success("Done ✓") if r.ok else st.error(r.text)
                    except Exception as e:
                        st.error(str(e))
        with mc2:
            if st.button("↑ Load"):
                with st.spinner("Loading model…"):
                    try:
                        r = requests.post(f"{BACKEND_URL}/api/vlm/load", timeout=300)
                        st.success("Loaded ✓") if r.ok else st.error(r.text)
                    except Exception as e:
                        st.error(str(e))
        if vlm_loaded:
            if st.button("Unload"):
                try:
                    r = requests.post(f"{BACKEND_URL}/api/vlm/unload", timeout=30)
                    st.info("Unloaded.") if r.ok else st.error(r.text)
                except Exception as e:
                    st.error(str(e))

    if vlm_backend == "ollama" and is_online:
        st.divider()
        st.markdown('<div class="mv-sidebar-label">Ollama</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="mv-pill {"mv-p-online" if ollama_ready else "mv-p-offline"}">'
            f'{"● Reachable" if ollama_ready else "● Unreachable"}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="mv-sidebar-label">Pipeline</div>', unsafe_allow_html=True)
    if st.button("↻ Reload checkpoint"):
        with st.spinner("Reloading…"):
            try:
                r = requests.post(f"{BACKEND_URL}/api/load",
                                   data={"checkpoint": "final_model"}, timeout=120)
                st.success("Reloaded ✓") if r.ok else st.error(r.text)
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown('<div class="mv-sidebar-label">Session Progress</div>',
                unsafe_allow_html=True)
    has_img    = st.session_state.last_image_bytes is not None
    has_ret    = st.session_state.last_result is not None
    has_report = st.session_state.last_report_result is not None
    st.markdown(
        step_html("1", "Upload X-ray",          "done"   if has_img    else "active") +
        step_html("2", "Retrieve similar cases", "done"   if has_ret    else ("active" if has_img else "pending")) +
        step_html("3", "Generate report",        "done"   if has_report else ("active" if has_ret else "pending")),
        unsafe_allow_html=True,
    )
    if has_img or has_ret:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ New session"):
            for k in ["last_result", "last_report_result",
                      "last_image_bytes", "last_image_name"]:
                st.session_state[k] = None
            st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mv-header">
  <div>
    <div class="mv-logo">Med<span class="mv-logo-accent">Vision</span> Retrieval AI</div>
    <div class="mv-tagline">Radiology Intelligence Platform · CLIP · MIMIC-CXR · FAISS · MedGemma</div>
  </div>
  <div class="mv-version">
    v2.0 · Research Prototype<br>
    <span style="color:#b5873a;">Authorised use only</span>
  </div>
</div>
<div class="mv-disclaimer">
  ⚠ &nbsp;RESEARCH PROTOTYPE — Not a certified medical device.
  All AI-generated reports require review and sign-off by a licensed radiologist before any clinical use.
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_retrieval, tab_report, tab_history, tab_about = st.tabs([
    "  Retrieval  ", "  Report Generation  ", "  Session History  ", "  Platform Info  "
])

QUICKSTART_HTML = """
<div class="mv-quickstart">
  <div class="mv-quickstart-title">Quick Start Guide</div>
  <div class="mv-quickstart-steps">
    <div class="mv-qs-step">
      <div class="mv-qs-num">1</div>
      <div class="mv-qs-text">
        <strong>Upload X-ray</strong><br>
        <span>Upload a chest X-ray image (JPEG or PNG) using the uploader on the left</span>
      </div>
    </div>
    <div class="mv-qs-step">
      <div class="mv-qs-num">2</div>
      <div class="mv-qs-text">
        <strong>Search Similar Cases</strong><br>
        <span>Click Search to find the most similar cases from the 30,633-image MIMIC-CXR index</span>
      </div>
    </div>
    <div class="mv-qs-step">
      <div class="mv-qs-num">3</div>
      <div class="mv-qs-text">
        <strong>Select Report Method</strong><br>
        <span>Choose a method in the sidebar — use <em>template</em> for instant results or a VLM method for AI-generated narrative</span>
      </div>
    </div>
    <div class="mv-qs-step">
      <div class="mv-qs-num">4</div>
      <div class="mv-qs-text">
        <strong>Generate Report</strong><br>
        <span>Go to the Report Generation tab and click Generate Report to produce a structured radiology report</span>
      </div>
    </div>
  </div>
</div>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Retrieval
# ══════════════════════════════════════════════════════════════════════════════
with tab_retrieval:
    # Quick-start at the very top of the Retrieval tab
    st.markdown(QUICKSTART_HTML, unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 2], gap="large")

    with col_up:
        st.markdown('<div class="mv-section-label">Query Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("xray", type=["png","jpg","jpeg"],
                                     label_visibility="collapsed")
        if uploaded:
            img_bytes = uploaded.read()
            st.session_state.last_image_bytes = img_bytes
            st.session_state.last_image_name  = uploaded.name
            st.image(img_bytes, use_container_width=True)
            st.markdown(
                f'<div style="margin-top:.4rem;">'
                f'{pill(uploaded.name, "mv-p-slate")}&nbsp;'
                f'{pill(f"{len(img_bytes)//1024} KB", "mv-p-slate")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.last_image_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("Search Similar Cases →",
                            disabled=(not is_online or not pipeline_ready))
            if is_online and not pipeline_ready:
                st.caption("Pipeline not initialised — reload checkpoint.")
        else:
            run = False
            st.markdown("""
            <div class="mv-upload-placeholder">
              <div class="icon">🫁</div>
              <p>Upload a chest X-ray to begin</p>
              <p style="margin-top:.3rem;font-size:.65rem;opacity:.5;">JPEG · PNG</p>
            </div>
            """, unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="mv-section-label">Similar Cases</div>',
                    unsafe_allow_html=True)

        if run:
            with st.spinner("Querying FAISS index…"):
                ret = do_retrieve(BACKEND_URL, st.session_state.last_image_bytes,
                                  st.session_state.last_image_name,
                                  top_k, use_qe, use_rr, min_score)
            if ret:
                st.session_state.last_result = ret
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "retrieval", "method": "retrieve",
                    "query_id": ret.get("query_id", "—"),
                    "num_results": ret.get("num_results", 0),
                    "result": ret,
                })
                st.success(f"{ret.get('num_results',0)} similar cases retrieved")

        ret_res = st.session_state.last_result
        if ret_res and ret_res.get("results"):
            cases = ret_res["results"]
            top_s = cases[0]["score"] if cases else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cases Retrieved", len(cases))
            m2.metric("Top Similarity",  f"{top_s:.4f}")
            m3.metric("Query ID",        ret_res.get("query_id", "—"))
            m4.metric("Index Size",      "30,633")

            st.markdown("<br>", unsafe_allow_html=True)

            n_ret    = min(len(cases), 4)
            all_cols = st.columns([1] + [1]*n_ret, gap="small")

            # Query card
            with all_cols[0]:
                st.markdown(
                    '<div class="mv-query-card">'
                    '<div class="mv-query-header">'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
                    'color:#1a3a6b;letter-spacing:.08em;text-transform:uppercase;">Query</span>'
                    '<span class="mv-pill mv-p-navy">Input</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.last_image_bytes:
                    st.image(st.session_state.last_image_bytes, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Retrieved cards
            for i, case in enumerate(cases[:n_ret]):
                sim    = case.get("score", 0)
                pct    = min(max(sim*100, 0), 100)
                cap    = case.get("caption", "")
                orig_i = case.get("original_index", -1)
                bs     = score_badge(pct)

                with all_cols[i+1]:
                    st.markdown(
                        f'<div class="mv-case-card">'
                        f'<div class="mv-case-header">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:.6rem;color:#6b7d93;">Case #{i+1}</span>'
                        f'<span style="width:24px;height:24px;border-radius:50%;'
                        f'display:inline-flex;align-items:center;justify-content:center;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:.58rem;'
                        f'font-weight:600;{bs}">{pct:.0f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    case_img = fetch_case_image(orig_i, BACKEND_URL)
                    if case_img:
                        st.image(case_img, use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="padding:2rem .5rem;text-align:center;'
                            'background:#f8fafc;">'
                            '<p style="font-size:1.4rem;margin:0;opacity:.25">🖼</p>'
                            '<p style="font-family:\'JetBrains Mono\',monospace;'
                            'font-size:.58rem;color:#94a3b8;margin:.2rem 0 0;">'
                            'Unavailable</p></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('<div class="mv-case-footer">', unsafe_allow_html=True)
                    st.markdown(sim_bar(sim), unsafe_allow_html=True)
                    if cap:
                        st.markdown(
                            f'<div class="mv-case-caption">{cap}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div></div>', unsafe_allow_html=True)

            if len(cases) > n_ret:
                with st.expander(f"View {len(cases)-n_ret} additional cases"):
                    for j, case in enumerate(cases[n_ret:]):
                        sim = case.get("score", 0)
                        cap = case.get("caption", "")
                        st.markdown(
                            f'{pill(f"Case #{n_ret+j+1}", "mv-p-slate")}&nbsp;'
                            f'{pill(f"Sim {sim:.4f}", "mv-p-gold")}',
                            unsafe_allow_html=True,
                        )
                        if cap:
                            st.caption(cap[:300])
                        st.divider()

            st.markdown(
                '<div class="mv-hint">'
                '✓ Retrieval complete — proceed to the Report Generation tab'
                '</div>',
                unsafe_allow_html=True,
            )

        elif st.session_state.last_image_bytes and not run:
            st.info("Click **Search Similar Cases** to query the index.")
        else:
            st.markdown(
                '<div class="mv-empty">'
                'Upload a chest X-ray and initiate retrieval<br>'
                'to surface similar cases from the MIMIC-CXR index.'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Report Generation
#  Layout: generate button at the TOP, settings below, report on the right
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    if not st.session_state.last_image_bytes:
        st.info("Upload a chest X-ray in the Retrieval tab first.")
    else:
        # ── Top action bar — always visible without scrolling ─────────────────
        action_left, action_mid, action_right = st.columns([1, 1, 1], gap="medium")

        with action_left:
            st.markdown(
                f'<div style="padding:.4rem 0;">'
                f'{pill(f"Method: {report_method}", "mv-p-navy")}&nbsp;'
                f'{pill(f"K={top_k}", "mv-p-slate")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with action_mid:
            if is_vlm:
                backend_lbl = "MedGemma" if report_method.startswith("vlm_") else "Ollama"
                shot_lbl    = "zero-shot" if is_zero else "few-shot"
                st.markdown(
                    f'<div style="padding:.4rem 0;">'
                    f'{pill(f"{backend_lbl} · {shot_lbl}", "mv-p-teal")}&nbsp;'
                    f'{pill(f"temp {vlm_temp}", "mv-p-slate")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with action_right:
            gen_btn = st.button(
                "🧠 Generate Report →" if is_vlm else "📋 Generate Report →",
                disabled=(not is_online or not pipeline_ready),
                use_container_width=True,
            )

        st.divider()

        # ── Main content: image + settings left, report right ─────────────────
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown('<div class="mv-section-label">Query Image</div>',
                        unsafe_allow_html=True)
            st.image(st.session_state.last_image_bytes,
                     use_container_width=True,
                     caption=st.session_state.last_image_name)

            if is_vlm:
                st.markdown("<br>", unsafe_allow_html=True)
                backend_lbl = "MedGemma (local)" if report_method.startswith("vlm_") else "Ollama"
                shot_lbl    = "Zero-shot — image only" if is_zero else f"Few-shot — {vlm_examples} retrieved examples as context"
                st.markdown(
                    f'<div class="mv-info-box">'
                    f'<div class="mv-info-label">VLM Configuration</div>'
                    f'<strong>{backend_lbl}</strong> · {shot_lbl}<br>'
                    f'<span style="color:#3d5168;font-size:.78rem;">'
                    f'Temperature {vlm_temp} · Max tokens {vlm_tokens}<br>'
                    f'Generation may take 30–90 s on GPU.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown('<div class="mv-section-label">Generated Report</div>',
                        unsafe_allow_html=True)

            if gen_btn:
                msg = (f"Running {report_method} inference…"
                       if is_vlm else "Generating report…")
                with st.spinner(msg):
                    rpt = do_generate_report(
                        BACKEND_URL,
                        st.session_state.last_image_bytes,
                        st.session_state.last_image_name,
                        report_method, top_k, min_score,
                        vlm_temp, vlm_tokens,
                        0 if is_zero else vlm_examples,
                    )
                if rpt:
                    st.session_state.last_report_result = rpt
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "type": "report", "method": report_method,
                        "query_id": rpt.get("query_id", "—"),
                        "num_results": len(rpt.get("retrieval_results", [])),
                        "result": rpt,
                    })

            rpt = st.session_state.last_report_result
            if rpt:
                report_text  = rpt.get("report", "No report generated.")
                success      = rpt.get("success", True)
                method_used  = rpt.get("method", report_method)
                model_used   = rpt.get("model", "")
                backend_used = rpt.get("backend", "")
                gen_time     = rpt.get("generation_time_s", 0)
                num_ex       = rpt.get("num_examples", 0)
                raw_out      = rpt.get("raw_vlm_output", "")
                ret_res      = rpt.get("retrieval_results", [])
                is_vlm_res   = (method_used.startswith("vlm_") or
                                method_used.startswith("ollama_"))

                # Status bar
                if is_vlm_res:
                    model_short = model_used.split("/")[-1] if model_used else method_used
                    st.markdown(
                        f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.9rem;">'
                        f'{pill("✓ Generated" if success else "⚠ Partial", "mv-p-online" if success else "mv-p-warn")}'
                        f'&nbsp;{pill(model_short, "mv-p-teal")}'
                        f'&nbsp;{pill(backend_used, "mv-p-slate")}'
                        f'&nbsp;{pill(f"{num_ex} examples", "mv-p-slate")}'
                        f'&nbsp;{pill(f"{gen_time}s", "mv-p-slate")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="margin-bottom:.9rem;">'
                        f'{pill(method_used.upper(), "mv-p-navy")}'
                        f'&nbsp;{pill(f"{gen_time}s", "mv-p-slate")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Detected conditions
                detected = rpt.get("detected_conditions") or {}
                pathological = {k: v for k, v in detected.items()
                                if k not in ("normal", "support_devices") and isinstance(v, dict)}
                if pathological:
                    top_conds = sorted(pathological.items(),
                                       key=lambda x: x[1].get("avg_score", 0),
                                       reverse=True)[:6]
                    pills_html = "".join(
                        f'<span class="mv-pill mv-p-warn" style="margin-right:.3rem;">'
                        f'{k.replace("_"," ").title()} {v.get("frequency",0)*100:.0f}%</span>'
                        for k, v in top_conds
                    )
                    st.markdown(
                        f'<div class="mv-conditions">'
                        f'<div class="mv-conditions-label">Detected Conditions</div>'
                        f'{pills_html}</div>',
                        unsafe_allow_html=True,
                    )

                # Report card
                card_cls   = "mv-report-card vlm" if is_vlm_res else "mv-report-card"
                header_cls = "mv-report-header vlm" if is_vlm_res else "mv-report-header"
                icon       = "🧠" if is_vlm_res else "📋"
                model_str  = (model_used.split("/")[-1] if model_used and is_vlm_res
                              else method_used.upper())
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<div class="{header_cls}">'
                    f'<span>{icon}</span><span>{model_str} · Radiology Report</span>'
                    f'</div>'
                    f'<div class="mv-report-text">{report_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Expandable details
                if ret_res:
                    with st.expander(f"Retrieved cases used ({len(ret_res)})"):
                        for i, r in enumerate(ret_res):
                            r_score = r.get("score", 0)
                            st.markdown(
                                f'{pill(f"#{i+1}", "mv-p-slate")}&nbsp;'
                                f'{pill(f"Score {r_score:.4f}", "mv-p-gold")}',
                                unsafe_allow_html=True,
                            )
                            if r.get("caption"):
                                st.caption(r["caption"][:350])
                            st.divider()

                if raw_out and raw_out != report_text:
                    with st.expander("Raw model output"):
                        st.markdown(
                            f'<div style="background:#f8fafc;border-radius:4px;'
                            f'padding:.8rem 1rem;font-family:\'JetBrains Mono\',monospace;'
                            f'font-size:.76rem;line-height:1.7;color:#3d5168;'
                            f'white-space:pre-wrap;">{raw_out}</div>',
                            unsafe_allow_html=True,
                        )

                # Downloads
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇ Download Report (.txt)", data=report_text,
                        file_name=f"medvision_report_{ts}.txt",
                        mime="text/plain", use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "⬇ Export JSON",
                        data=json.dumps(rpt, indent=2, default=str),
                        file_name=f"medvision_result_{ts}.json",
                        mime="application/json", use_container_width=True,
                    )

                # Feedback
                st.divider()
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);'
                    'border-top:3px solid var(--teal);border-radius:0 0 6px 6px;'
                    'padding:1.2rem 1.5rem;box-shadow:var(--shadow);">'
                    '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;'
                    'letter-spacing:0.14em;text-transform:uppercase;color:var(--teal);'
                    'margin-bottom:0.8rem;">Radiologist Review &amp; Feedback</div>'
                    '<div style="background:var(--teal-light);border:1px solid var(--teal-border);'
                    'border-radius:4px;padding:0.7rem 1rem;margin-bottom:1rem;'
                    'font-size:0.81rem;color:var(--text);line-height:1.6;">'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
                    'color:var(--teal);letter-spacing:0.1em;text-transform:uppercase;">'
                    'Where this goes</span><br>'
                    'Feedback is <strong>logged to session history</strong> and can be '
                    'downloaded as CSV from the Session History tab. '
                    'Index updates are disabled in this demo — in a clinical deployment, '
                    'verified corrections would be reviewed before being added to the retrieval index.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                feedback_text = st.text_area(
                    "Corrections or clinical notes",
                    placeholder="e.g. 'No pleural effusion. Cardiomegaly is mild, not moderate.'",
                    height=110, label_visibility="collapsed",
                )
                st.checkbox(
                    "Append case to FAISS index",
                    value=False,
                    disabled=True,
                    help="Index updates are disabled in the demo. In a clinical deployment this would require admin review before modifying the retrieval index.",
                )
                if st.button("Submit Feedback", use_container_width=True):
                    if not feedback_text.strip():
                        st.warning("Please enter feedback before submitting.")
                    else:
                        # Log to session history only — no index modification
                        st.session_state.history.append({
                            "timestamp":   datetime.now().strftime("%H:%M:%S"),
                            "type":        "feedback",
                            "method":      method_used,
                            "query_id":    rpt.get("query_id", "—"),
                            "num_results": 0,
                            "result": {
                                "feedback":    feedback_text,
                                "report":      report_text,
                                "image":       st.session_state.last_image_name,
                                "method":      method_used,
                                "query_id":    rpt.get("query_id", "—"),
                                "timestamp":   datetime.now().isoformat(),
                            },
                        })
                        st.success("Feedback recorded in session history ✓ — visible in the Session History tab.")
            elif not gen_btn:
                st.markdown(
                    '<div class="mv-empty">'
                    'Click <strong>Generate Report →</strong> above<br>'
                    'to produce a structured radiology report.'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="mv-section-label">Session History</div>',
                unsafe_allow_html=True)
    history = st.session_state.get("history", [])

with tab_history:
    st.markdown('<div class="mv-section-label">Session History</div>',
                unsafe_allow_html=True)
    history = st.session_state.get("history", [])

    if not history:
        st.markdown(
            '<div class="mv-empty">No queries recorded this session.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── CSV export ────────────────────────────────────────────────────────
        def build_csv(hist: list) -> str:
            import csv, io, re
            buf = io.StringIO(newline="")
            writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "timestamp", "type", "method", "query_id",
                "num_results", "top_score", "report_snippet", "feedback",
            ])

            def safe(text: str, max_len: int = 300) -> str:
                """Clean text for safe CSV/Excel rendering."""
                t = str(text).encode("utf-8", errors="replace").decode("utf-8")
                t = re.sub(r"[\r\n\t]+", " ", t).strip()
                # Remove === separator lines entirely
                t = re.sub(r"=+", "", t)
                t = re.sub(r"-{3,}", "", t)
                # Strip leading formula chars
                t = re.sub(r"^[=+\-@|]+", "", t).strip()
                return t[:max_len]

            for e in hist:
                res = e["result"]
                if e["type"] == "feedback":
                    writer.writerow([
                        e["timestamp"],
                        "feedback",
                        e["method"],
                        e["query_id"],
                        "—",
                        "—",
                        safe(res.get("report", "")),
                        safe(res.get("feedback", "")),
                    ])
                elif e["type"] == "report":
                    cases = res.get("retrieval_results", [])
                    top_s = round(cases[0].get("score", 0), 4) if cases else "—"
                    writer.writerow([
                        e["timestamp"],
                        "report",
                        e["method"],
                        e["query_id"],
                        e["num_results"],
                        top_s,
                        safe(res.get("report", "")),
                        "",
                    ])
                else:
                    cases = res.get("results", [])
                    top_s = round(cases[0].get("score", 0), 4) if cases else "—"
                    writer.writerow([
                        e["timestamp"],
                        "retrieval",
                        e["method"],
                        e["query_id"],
                        e["num_results"],
                        top_s,
                        "",
                        "",
                    ])
            return buf.getvalue().encode("utf-8-sig").decode("utf-8-sig")

        ts_dl = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_data = build_csv(history).encode("utf-8-sig")  # BOM = Excel opens correctly
        dl_col, clr_col = st.columns([2, 1], gap="small")
        with dl_col:
            st.download_button(
                "⬇ Download session history (.csv)",
                data=csv_data,
                file_name=f"medvision_history_{ts_dl}.csv",
                mime="text/csv; charset=utf-8-sig",
                use_container_width=True,
            )
        with clr_col:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        for idx, entry in enumerate(reversed(history)):
            res   = entry["result"]
            type_badge = {
                "report":    "📋 REPORT",
                "retrieval": "🔍 RETRIEVAL",
                "feedback":  "💬 FEEDBACK",
            }.get(entry["type"], entry["type"].upper())
            label = (f'[{entry["timestamp"]}]  {type_badge}  ·  '
                     f'{entry["method"]}  ·  ID {entry["query_id"]}')

            with st.expander(label, expanded=(idx == 0)):
                if entry["type"] == "feedback":
                    st.markdown(
                        f'<div style="background:var(--teal-light);border:1px solid '
                        f'var(--teal-border);border-radius:6px;padding:1rem 1.2rem;">'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
                        f'color:var(--teal);letter-spacing:.1em;text-transform:uppercase;'
                        f'margin-bottom:.5rem;">Radiologist Feedback</div>'
                        f'<div style="font-size:.88rem;color:var(--text);line-height:1.7;">'
                        f'{res.get("feedback", "—")}</div>'
                        f'<div style="margin-top:.6rem;font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:.62rem;color:var(--text-muted);">'
                        f'Image: {res.get("image","—")} · Method: {res.get("method","—")} · '
                        f'Query ID: {res.get("query_id","—")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if res.get("report"):
                        with st.expander("Report this feedback refers to"):
                            st.markdown(
                                f'<div class="mv-report-card">'
                                f'<div class="mv-report-text">{res["report"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                elif entry["type"] == "report" and "report" in res:
                    st.markdown(
                        f'<div class="mv-report-card">'
                        f'<div class="mv-report-header">'
                        f'<span>📋</span>'
                        f'<span>{res.get("method","").upper()} Report</span>'
                        f'</div>'
                        f'<div class="mv-report-text">{res.get("report","—")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif entry["type"] == "retrieval" and "results" in res:
                    for r in res["results"][:3]:
                        r_rank  = r.get("rank", "")
                        r_score = r.get("score", 0)
                        st.markdown(
                            f'{pill(f"#{r_rank}", "mv-p-slate")}&nbsp;'
                            f'{pill(f"Score {r_score:.4f}", "mv-p-gold")}',
                            unsafe_allow_html=True,
                        )
                        if r.get("caption"):
                            st.caption(r["caption"][:200])
                        st.divider()
                if entry["type"] != "feedback":
                    with st.expander("Raw response"):
                        st.json(res)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Platform Info
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="mv-section-label">Platform Overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        "MedVision Retrieval AI is a domain-adaptive medical image retrieval and report generation "
        "system built on fine-tuned CLIP, FAISS vector search, and Vision-Language Model "
        "inference. It enables clinicians to rapidly surface similar cases from a large "
        "chest X-ray corpus and generate structured radiology reports with AI assistance."
    )
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
#### Pipeline Architecture

**Stage 1 · Model Training**
Fine-tunes OpenAI CLIP (`openai/clip-vit-base-patch32`) on the MIMIC-CXR dataset
using symmetric contrastive loss (InfoNCE) with multi-layer improved projection heads
and optional hard negative mining.

**Stage 2 · Index Construction**
Encodes 30,633 chest X-rays through the fine-tuned CLIP model.
Applies PCA whitening (256 → 195 dimensions) and builds a FAISS IVFFlat index
for sub-second approximate nearest neighbour search.

**Stage 3 · Clinical Inference**
Query image → CLIP embedding → optional Rocchio query expansion
→ FAISS search → optional cross-encoder re-ranking → report generation.

#### Report Generation Methods

| Method | Description | GPU Required |
|---|---|---|
| `template` | Keyword analysis → FINDINGS/IMPRESSION | No |
| `vlm_few_shot` | MedGemma with retrieved case context | Yes |
| `vlm_zero_shot` | MedGemma, query image only | Yes |
| `ollama_few_shot` | Ollama LLaVA with retrieved context | Yes |
| `weighted` | Similarity-weighted caption blend | No |
| `majority` | Consensus findings only | No |
        """)

    with c2:
        st.markdown("""
#### API Reference

| Endpoint | Method | Function |
|---|---|---|
| `/api/health` | GET | System health + VLM status |
| `/api/status` | GET | Full pipeline diagnostics |
| `/api/load` | POST | Load checkpoint + index |
| `/api/retrieve` | POST | FAISS similarity search |
| `/api/generate-report` | POST | Report generation |
| `/api/feedback` | POST | Feedback + index update |
| `/api/vlm/download` | POST | Download MedGemma |
| `/api/vlm/load` | POST | Load model to memory |
| `/api/vlm/unload` | POST | Free model memory |

#### Technology Stack

| Component | Detail |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Vision encoder | CLIP ViT-B/32 (fine-tuned on MIMIC-CXR) |
| VLM | MedGemma `google/medgemma-4b-it` |
| Remote VLM | LLaVA-LLaMA3 via Ollama |
| Vector index | FAISS IVFFlat · 30,633 vectors |
| Dataset | MIMIC-CXR · PhysioNet / HuggingFace |

#### Starting the Backend

```bash
# GPU cluster (recommended)
python api.py --host 0.0.0.0 --port 8000

# 4-bit quantized (lower VRAM)
python api.py --vlm_4bit

# Ollama backend
python api.py --vlm_backend ollama
```
        """)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Development Team**")
        st.markdown("Nadeesha Perera · Alli Raittinen")
    with col_b:
        st.markdown("**Classification**")
        st.markdown(
            f'{pill("Research Prototype", "mv-p-warn")}&nbsp;'
            f'{pill("Not for Clinical Use", "mv-p-offline")}',
            unsafe_allow_html=True,
        )
    st.caption(
        f"MedVision Retrieval AI · {datetime.now().year} · "
        "All AI outputs require radiologist review before clinical application."
    )