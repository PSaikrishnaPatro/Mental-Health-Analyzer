import streamlit as st

# ── MUST be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

import Download_model  # ensures model weights exist
from utils import (
    load_model, label_map, label_colors, label_icons,
    DEVICE, clean_and_lemmatize_text, get_text_stats,
    get_label_description, get_resources, CRISIS_INFO,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0A0E1A; color: #E2E8F0; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1629 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stRadio > label { color: #94A3B8 !important; font-size: 13px; }
[data-testid="stSidebar"] .stRadio input:checked + div { color: #818CF8 !important; }

/* ── Typography ── */
h1 { font-size: 2.6rem !important; font-weight: 800 !important; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; color: #C7D2FE !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; color: #A5B4FC !important; }

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    backdrop-filter: blur(12px);
    margin-bottom: 18px;
}
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
}
.metric-title { font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 2rem; font-weight: 800; color: #C7D2FE; }
.metric-sub   { font-size: 12px; color: #64748B; margin-top: 4px; }

/* ── Predicted Result Badge ── */
.result-badge {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 14px 28px; border-radius: 50px;
    font-size: 1.5rem; font-weight: 800;
    letter-spacing: 0.5px;
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
    animation: fadeSlideIn 0.5s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Crisis Box ── */
.crisis-box {
    background: linear-gradient(135deg, rgba(239,35,60,0.18), rgba(239,35,60,0.06));
    border: 1.5px solid rgba(239,35,60,0.55);
    border-radius: 14px;
    padding: 20px 26px;
    animation: fadeSlideIn 0.6s ease;
}
.crisis-title { font-size: 1.15rem; font-weight: 700; color: #FC8181; margin-bottom: 10px; }

/* ── Tips ── */
.tip-item {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.92rem;
    color: #CBD5E1;
}

/* ── Feature Card ── */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.feature-card:hover { transform: translateY(-3px); border-color: rgba(99,102,241,0.4); }
.feature-icon { font-size: 2.2rem; margin-bottom: 10px; }
.feature-title { font-weight: 700; color: #C7D2FE; font-size: 0.95rem; }
.feature-desc  { color: #64748B; font-size: 0.82rem; margin-top: 5px; }

/* ── Step ── */
.step-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.step-num {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white; font-weight: 700; font-size: 0.82rem;
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-text { color: #CBD5E1; font-size: 0.9rem; }

/* ── Label Badge ── */
.label-badge {
    display: inline-block; padding: 5px 14px; border-radius: 50px;
    font-size: 0.82rem; font-weight: 600; margin: 3px;
}

/* ── Text Area ── */
.stTextArea textarea {
    background: #0F1629 !important; color: #E2E8F0 !important;
    font-size: 16px !important; line-height: 1.7 !important;
    border-radius: 12px !important; border: 1.5px solid rgba(99,102,241,0.3) !important;
    padding: 16px !important;
}
.stTextArea textarea:focus { border-color: #6366F1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; }

/* ── Buttons — compact on all screen sizes ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.55em 1.4em !important; font-size: 0.95rem !important;
    width: auto !important;
    min-width: 160px !important;
    max-width: 220px !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stDownloadButton > button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    color: #A5B4FC !important; font-weight: 600 !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
.custom-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent); margin: 28px 0; }

/* ── Warning / info ── */
.stat-pill {
    display: inline-block; padding: 4px 12px; border-radius: 50px;
    font-size: 0.78rem; font-weight: 600; margin-right: 8px;
    background: rgba(99,102,241,0.15); color: #A5B4FC; border: 1px solid rgba(99,102,241,0.25);
}
.warning-pill {
    display: inline-block; padding: 4px 12px; border-radius: 50px;
    font-size: 0.78rem; font-weight: 600;
    background: rgba(234,179,8,0.15); color: #FCD34D; border: 1px solid rgba(234,179,8,0.3);
}
.good-pill {
    display: inline-block; padding: 4px 12px; border-radius: 50px;
    font-size: 0.78rem; font-weight: 600;
    background: rgba(16,185,129,0.15); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3);
}

/* ── History table ── */
.stDataFrame { background: #0F1629 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748B !important; font-weight: 600 !important;
    padding: 10px 22px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
}

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg, #6366F1, #8B5CF6) !important; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

with st.spinner("🧠 Loading AI model… please wait a moment"):
    tokenizer, model = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:3rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:800; color:#C7D2FE; margin-top:6px;">Mental Health<br>Analyzer</div>
        <div style="font-size:0.72rem; color:#475569; margin-top:4px;">Powered by BERT · 90% Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Analyze", "📋 Batch Predict", "📜 History", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:14px; background:rgba(239,35,60,0.08); border:1px solid rgba(239,35,60,0.3); border-radius:10px;">
        <div style="font-size:0.78rem; font-weight:700; color:#FC8181; margin-bottom:6px;">🆘 Crisis Support</div>
        <div style="font-size:0.72rem; color:#94A3B8; line-height:1.6;">
            📞 Call/Text <b style="color:#FCA5A5;">988</b> (US)<br>
            💬 Text <b style="color:#FCA5A5;">HOME</b> to 741741<br>
            🌍 <a href="https://www.iasp.info/resources/Crisis_Centres/" style="color:#93C5FD;" target="_blank">Find help worldwide</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px; font-size:0.68rem; color:#334155; line-height:1.5; text-align:center;">
        ⚠️ For informational use only.<br>Not a substitute for professional care.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Home":
    st.markdown("""
    <div style="padding: 40px 0 20px; text-align:center;">
        <div style="font-size:0.8rem; font-weight:600; color:#818CF8; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
            AI-Powered NLP Tool
        </div>
        <h1 style="background: linear-gradient(135deg,#818CF8,#C084FC,#F472B6);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   background-clip:text; margin-bottom:16px; font-size:3rem !important;">
            Mental Health Analyzer
        </h1>
        <p style="color:#94A3B8; font-size:1.05rem; max-width:560px; margin:0 auto 30px;">
            Understand mental health signals in text using a fine-tuned BERT model trained on 50,000+ real-world samples across 7 conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats row ──
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("🎯", "~90%", "Model Accuracy"),
        ("🗂️", "50K+", "Training Samples"),
        ("🏷️", "7", "Mental Health Classes"),
        ("⚡", "BERT", "NLP Architecture"),
    ]
    for col, (icon, val, label) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.6rem;">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Features ──
    st.markdown("### ✨ Key Features")
    features = [
        ("🔍", "Single Prediction", "Analyze any text statement and get an instant mental health classification with confidence scores."),
        ("📊", "Probability Chart", "Interactive Plotly bar chart showing confidence across all 7 classes."),
        ("📋", "Batch Processing", "Upload a CSV and classify thousands of rows at once with progress tracking."),
        ("📜", "Session History", "Every prediction is saved in your session with timestamps — downloadable as CSV."),
        ("🛡️", "Coping Resources", "Tailored coping strategies and trusted external resources per predicted category."),
        ("🆘", "Crisis Support", "Prominent crisis hotline info auto-shown for high-risk predictions."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 3].markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── How to use ──
    col_how, col_labels = st.columns([1, 1], gap="large")
    with col_how:
        st.markdown("### 📖 How to Use")
        steps = [
            ("1", "Navigate to <b>🔍 Analyze</b> in the sidebar"),
            ("2", "Type or paste your text (2-3 sentences work best)"),
            ("3", "Click <b>Analyze Text</b> to run the model"),
            ("4", "View your result, chart, and coping strategies"),
            ("5", "Save to history or use <b>📋 Batch Predict</b> for CSVs"),
        ]
        for num, text in steps:
            st.markdown(f"""
            <div class="step-item">
                <div class="step-num">{num}</div>
                <div class="step-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_labels:
        st.markdown("### 🏷️ Supported Conditions")
        for lbl, color in label_colors.items():
            icon = label_icons.get(lbl, "")
            st.markdown(f"""
            <span class="label-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                {icon} {lbl}
            </span>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:18px;">
            <div style="font-size:0.82rem; color:#94A3B8; line-height:1.7;">
                💡 <b style="color:#C7D2FE;">Pro tip:</b> Use 2-3 full sentences for best accuracy.
                Short phrases like <i>"I give up"</i> may be classified as Normal.
                Detailed context helps the model understand true intent.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🔍 Analyze":
    st.markdown("## 🔍 Analyze Text")
    st.markdown('<p style="color:#64748B; margin-top:-10px;">Enter a statement and let the AI classify its mental health indicators.</p>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Your text",
        height=160,
        value="I feel restless and anxious all the time. Nothing I do seems to help.",
        placeholder="Type something like 'I feel completely hopeless and exhausted every day...'",
        label_visibility="collapsed",
    )

    # ── Live stats ──
    if user_input.strip():
        stats_data = get_text_stats(user_input)
        wc = stats_data["word_count"]
        sc = stats_data["sentence_count"]
        hint = stats_data["quality_hint"]
        pill_cls = "warning-pill" if wc < 8 else "good-pill"
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <span class="stat-pill">📝 {wc} words</span>
            <span class="stat-pill">📄 {sc} sentences</span>
            <span class="{pill_cls}">{hint}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Compact button — no use_container_width, CSS caps the size ──
    run = st.button("🔍 Analyze Text", use_container_width=False)

    if run:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Running inference…"):
                cleaned = clean_and_lemmatize_text(user_input)
                inputs = tokenizer(cleaned, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                    pred_id = int(probs.argmax())
                    pred_label = label_map[pred_id]
                    confidence = float(probs[pred_id]) * 100

            color = label_colors[pred_label]
            icon  = label_icons.get(pred_label, "")
            desc  = get_label_description(pred_label)
            res   = get_resources(pred_label)

            # ── Result Badge ──
            st.markdown(f"""
            <div style="margin: 20px 0 10px;">
                <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px;">Prediction Result</div>
                <span class="result-badge" style="background:{color}22; color:{color}; border:2px solid {color}55;">
                    {icon} {pred_label}
                    <span style="font-size:1rem; font-weight:500; opacity:0.8; margin-left:4px;">{confidence:.1f}% confidence</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Description ──
            st.markdown(f"""
            <div class="glass-card" style="margin:12px 0;">
                <div style="font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">What this means</div>
                <div style="color:#CBD5E1; font-size:0.92rem; line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Plotly Chart (fixed) ──
            labels_list = list(label_map.values())
            probs_list  = [float(probs[i]) * 100 for i in range(len(labels_list))]
            colors_list = [label_colors[l] for l in labels_list]

            fig = go.Figure(go.Bar(
                x=probs_list,
                y=labels_list,
                orientation='h',
                marker=dict(
                    color=colors_list,
                    opacity=[1.0 if l == pred_label else 0.38 for l in labels_list],
                    line=dict(width=0),
                ),
                text=[f"{p:.1f}%" for p in probs_list],
                textposition='inside',
                insidetextanchor='end',
                textfont=dict(color='#FFFFFF', size=12),
                hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=20, t=10, b=10),
                xaxis=dict(
                    range=[0, 100],
                    showgrid=False, zeroline=False,
                    tickfont=dict(color='#475569'),
                    showticklabels=False,
                ),
                yaxis=dict(
                    tickfont=dict(color='#94A3B8', size=13),
                    gridcolor='rgba(255,255,255,0.04)',
                ),
                height=300,
                bargap=0.35,
                dragmode=False,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "staticPlot": True},
            )

            # ── Crisis Box (if applicable) ──
            if res["is_crisis"]:
                st.markdown("""
                <div class="crisis-box">
                    <div class="crisis-title">🆘 You are not alone — Help is available right now</div>
                    <div style="font-size:0.9rem; color:#FCA5A5; margin-bottom:10px;">
                        If you or someone you know is in crisis, please reach out immediately:
                    </div>
                    <div style="font-size:0.88rem; color:#CBD5E1; line-height:2;">
                        📞 <b>Call or text 988</b> — Suicide & Crisis Lifeline (US, 24/7)<br>
                        💬 <b>Text HOME to 741741</b> — Crisis Text Line (24/7)<br>
                        🌍 <a href="https://www.iasp.info/resources/Crisis_Centres/" style="color:#93C5FD;" target="_blank">Find international crisis centres</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Coping Strategies & Resources ──
            col_tips, col_res = st.columns([1, 1], gap="large")

            with col_tips:
                st.markdown("### 🛠️ Coping Strategies")
                for tip in res.get("tips", []):
                    st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)

            with col_res:
                st.markdown("### 🔗 Helpful Resources")
                for r_item in res.get("resources", []):
                    st.markdown(f"""
                    <a href="{r_item['url']}" target="_blank" style="
                        display:block; padding:10px 16px; margin:6px 0;
                        background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                        border-radius:8px; color:#A5B4FC; font-size:0.88rem;
                        text-decoration:none;">
                        ↗ {r_item['name']}
                    </a>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Save to History ──
            if st.button("💾 Save to History", use_container_width=False):
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": user_input[:120] + ("…" if len(user_input) > 120 else ""),
                    "prediction": pred_label,
                    "confidence": f"{confidence:.1f}%",
                })
                st.success("✅ Saved to history!")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📋 Batch Predict":
    st.markdown("## 📋 Batch Predict")
    st.markdown('<p style="color:#64748B; margin-top:-10px;">Upload a CSV file with a text column to classify multiple entries at once.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="CSV must contain a column named 'text'",
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">File Preview</div>
            <div style="color:#CBD5E1; font-size:0.88rem;">{len(df)} rows · {len(df.columns)} columns detected</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)

        text_col = None
        for c in df.columns:
            if "text" in c.lower():
                text_col = c
                break

        if text_col is None:
            st.error("❌ No column named 'text' found. Please rename your text column to 'text'.")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=False):
                predictions, confidences = [], []
                progress = st.progress(0)
                status   = st.empty()
                total    = len(df)

                for i, row_text in enumerate(df[text_col].astype(str)):
                    cleaned = clean_and_lemmatize_text(row_text)
                    inputs  = tokenizer(cleaned, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                        p = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                    pid = int(p.argmax())
                    predictions.append(label_map[pid])
                    confidences.append(f"{float(p[pid]) * 100:.1f}%")
                    progress.progress((i + 1) / total)
                    status.markdown(f'<span style="color:#94A3B8; font-size:0.82rem;">Processing {i+1}/{total}…</span>', unsafe_allow_html=True)

                df["prediction"]  = predictions
                df["confidence"]  = confidences
                progress.empty()
                status.empty()

                st.success(f"✅ Done! Classified {total} rows.")
                st.dataframe(df, use_container_width=True)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📜 History":
    st.markdown("## 📜 Prediction History")
    st.markdown('<p style="color:#64748B; margin-top:-10px;">All predictions from this session.</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📭</div>
            <div style="color:#64748B; font-size:0.95rem;">No predictions yet. Head to <b style="color:#A5B4FC;">🔍 Analyze</b> to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download History CSV",
            data=csv_bytes,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state.history = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "ℹ️ About":
    st.markdown("## ℹ️ About")

    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">🧠 Mental Health Analyzer</h3>
        <div style="color:#CBD5E1; font-size:0.92rem; line-height:1.8;">
            This tool uses a fine-tuned <b style="color:#C7D2FE;">BERT</b> (Bidirectional Encoder Representations from Transformers)
            model to classify text into one of 7 mental health categories. It was trained on 50,000+ real-world social media
            posts and clinical text samples.<br><br>
            <b style="color:#C7D2FE;">Supported classes:</b> Anxiety, Depression, Bipolar, Suicidal, Stress, Personality Disorder, Normal.<br><br>
            <b style="color:#FC8181;">⚠️ Disclaimer:</b> This tool is for informational and educational purposes only.
            It is not a diagnostic tool and should not replace professional mental health evaluation or treatment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">🔬 Model Details</h3>
        <div style="color:#CBD5E1; font-size:0.92rem; line-height:2;">
            <b style="color:#A5B4FC;">Architecture:</b> bert-base-uncased (fine-tuned)<br>
            <b style="color:#A5B4FC;">Training samples:</b> 50,000+<br>
            <b style="color:#A5B4FC;">Validation accuracy:</b> ~90%<br>
            <b style="color:#A5B4FC;">Input max length:</b> 128 tokens<br>
            <b style="color:#A5B4FC;">Framework:</b> PyTorch + HuggingFace Transformers<br>
            <b style="color:#A5B4FC;">Interface:</b> Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">🆘 Crisis Resources</h3>
        <div style="color:#CBD5E1; font-size:0.92rem; line-height:2;">
            📞 <b>988 Suicide & Crisis Lifeline</b> — Call or text 988 (US, 24/7)<br>
            💬 <b>Crisis Text Line</b> — Text HOME to 741741 (US, 24/7)<br>
            🌍 <a href="https://www.iasp.info/resources/Crisis_Centres/" style="color:#93C5FD;" target="_blank">International Association for Suicide Prevention</a><br>
            🧠 <a href="https://www.nami.org/help" style="color:#93C5FD;" target="_blank">NAMI Helpline</a><br>
            💙 <a href="https://www.mentalhealth.gov/" style="color:#93C5FD;" target="_blank">MentalHealth.gov</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
