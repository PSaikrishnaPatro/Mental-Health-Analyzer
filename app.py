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

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65em 1.8em !important; font-size: 1rem !important;
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
    for col, (icon, val, label) in zip([c1,c2,c3,c4], stats):
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
    f1, f2, f3 = st.columns(3)
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

    col_btn, col_clear = st.columns([2, 8])
    with col_btn:
        run = st.button("🔍 Analyze Text", use_container_width=True)

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

            color   = label_colors[pred_label]
            icon    = label_icons.get(pred_label, "")
            desc    = get_label_description(pred_label)
            res     = get_resources(pred_label)

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

            # ── Plotly Chart ──
            labels_list  = list(label_map.values())
            probs_list   = [float(probs[i]) * 100 for i in range(len(labels_list))]
            colors_list  = [label_colors[l] for l in labels_list]

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
                textposition='outside',
                textfont=dict(color='#94A3B8', size=12),
                hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=60, t=10, b=10),
                xaxis=dict(
                    range=[0, max(probs_list) + 12],
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
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
                st.markdown("")

            # ── Coping Strategies & Resources ──
            col_tips, col_links = st.columns([3, 2], gap="large")
            with col_tips:
                st.markdown(f"### 🛠️ Coping Strategies")
                for tip in res["tips"]:
                    st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)

            with col_links:
                st.markdown("### 🔗 Helpful Resources")
                for name, url in res["links"]:
                    st.markdown(f"""
                    <a href="{url}" target="_blank" style="
                        display:block; padding:12px 16px; margin:6px 0;
                        background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                        border-radius:10px; color:#A5B4FC; font-size:0.85rem;
                        text-decoration:none; transition:all 0.2s;
                    ">↗ {name}</a>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Save to History ──
            save_col, _ = st.columns([2, 8])
            with save_col:
                if st.button("💾 Save to History", use_container_width=True):
                    snippet = user_input[:80] + ("…" if len(user_input) > 80 else "")
                    st.session_state.history.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Text Snippet": snippet,
                        "Prediction": pred_label,
                        "Confidence": f"{confidence:.1f}%",
                    })
                    st.success(f"✅ Saved to History! ({len(st.session_state.history)} entries)")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📋 Batch Predict":
    st.markdown("## 📋 Batch Prediction")
    st.markdown('<p style="color:#64748B; margin-top:-10px;">Upload a CSV with a <code>text</code> column to classify all rows at once.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <b style="color:#C7D2FE;">CSV Format</b>
        <div style="margin-top:8px; font-family:monospace; font-size:0.85rem; color:#94A3B8;">
            text<br>
            "I have been feeling hopeless for weeks."<br>
            "Today was a great day, I feel fantastic!"<br>
            "The pressure at work is unbearable."
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("❌ The CSV must contain a column named **text**.")
        else:
            st.info(f"📄 {len(df)} rows detected. Running predictions…")
            progress = st.progress(0, text="Analyzing…")
            preds, confidences = [], []
            total = len(df)
            for i, txt in enumerate(df['text']):
                cleaned = clean_and_lemmatize_text(str(txt))
                inp = tokenizer(cleaned, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.no_grad():
                    out = model(**inp)
                    prb = F.softmax(out.logits, dim=1).cpu().numpy()[0]
                    pid = int(prb.argmax())
                    preds.append(label_map[pid])
                    confidences.append(f"{prb[pid]*100:.1f}%")
                progress.progress((i + 1) / total, text=f"Analyzed {i+1}/{total} rows…")

            df['Prediction'] = preds
            df['Confidence'] = confidences

            st.success(f"✅ Done! Classified **{total}** rows.")
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Label Distribution Pie ──
            dist = pd.Series(preds).value_counts().reset_index()
            dist.columns = ['Label', 'Count']
            pie_colors = [label_colors.get(l, "#888") for l in dist['Label']]
            fig_pie = px.pie(
                dist, names='Label', values='Count',
                color_discrete_sequence=pie_colors,
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#94A3B8')),
                margin=dict(l=0, r=0, t=20, b=0),
                font=dict(color='#94A3B8'),
            )
            col_pie, col_tbl = st.columns([1, 2], gap="large")
            with col_pie:
                st.markdown("#### Label Distribution")
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            with col_tbl:
                st.markdown("#### Results Preview")
                st.dataframe(
                    df[['text', 'Prediction', 'Confidence']].head(20),
                    use_container_width=True,
                    hide_index=True,
                )

            csv_buf = BytesIO()
            df[['text', 'Prediction', 'Confidence']].to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Full Predictions CSV",
                data=csv_buf.getvalue(),
                file_name="mental_health_predictions.csv",
                mime="text/csv",
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📜 History":
    st.markdown("## 📜 Session History")
    st.markdown('<p style="color:#64748B; margin-top:-10px;">All predictions saved during this session.</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📂</div>
            <div style="color:#475569;">No predictions saved yet.</div>
            <div style="color:#334155; font-size:0.82rem; margin-top:6px;">
                Go to <b>🔍 Analyze</b>, run a prediction, then click <b>💾 Save to History</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            hist_csv = BytesIO()
            df_hist.to_csv(hist_csv, index=False)
            st.download_button(
                "⬇️ Download History CSV",
                data=hist_csv.getvalue(),
                file_name="analysis_history.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

        # Label distribution
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Distribution of Saved Predictions")
        hist_dist = df_hist['Prediction'].value_counts().reset_index()
        hist_dist.columns = ['Label', 'Count']
        hbar_colors = [label_colors.get(l, "#888") for l in hist_dist['Label']]
        fig_hist = go.Figure(go.Bar(
            x=hist_dist['Count'],
            y=hist_dist['Label'],
            orientation='h',
            marker_color=hbar_colors,
            text=hist_dist['Count'],
            textposition='outside',
            textfont=dict(color='#94A3B8'),
        ))
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(color='#94A3B8', size=13)),
            margin=dict(l=0, r=40, t=10, b=10),
            height=250,
            bargap=0.4,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">🤖 Model Details</h3>
            <table style="width:100%; font-size:0.88rem; color:#CBD5E1; border-collapse:collapse;">
                <tr><td style="color:#64748B; padding:6px 0;">Architecture</td><td>BERT (bert-base-uncased)</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Task</td><td>Sequence Classification</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Accuracy</td><td>~90%</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Max Input Length</td><td>128 tokens</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Output Classes</td><td>7 mental health categories</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Training Data</td><td>50,000+ social media posts</td></tr>
                <tr><td style="color:#64748B; padding:6px 0;">Split</td><td>70% train / 15% val / 15% test</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">🧹 Preprocessing Pipeline</h3>
            <div style="font-size:0.87rem; color:#CBD5E1; line-height:2;">
                1. Lowercase & HTML entity decode<br>
                2. Remove URLs, @mentions, hashtags<br>
                3. Strip special characters & normalize whitespace<br>
                4. POS-aware lemmatization via WordNet<br>
                5. Class balancing with oversampling & augmentation
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">⚠️ Limitations</h3>
            <div style="font-size:0.87rem; color:#CBD5E1; line-height:2;">
                🌐 <b>Language:</b> English only — may degrade on informal/slang text<br>
                📌 <b>Domain:</b> Trained on social media; may differ for clinical notes<br>
                🎭 <b>Sarcasm:</b> Partial detection only<br>
                📝 <b>Length:</b> Needs 2-3 sentences for meaningful context<br>
                🏥 <b>Scope:</b> Only covers 7 predefined classes<br>
                🤖 <b>Not clinical:</b> Cannot replace a licensed professional
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">📊 Dataset</h3>
            <div style="font-size:0.87rem; color:#CBD5E1; line-height:1.8;">
                Source: Kaggle Mental Health Sentiment Dataset<br>
                Classes: Anxiety · Bipolar · Depression · Normal · Personality Disorder · Stress · Suicidal<br><br>
                <a href="https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health"
                   target="_blank" style="color:#818CF8;">↗ View Dataset on Kaggle</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="border-color:rgba(239,35,60,0.25); background:rgba(239,35,60,0.04); margin-top:4px;">
        <b style="color:#FC8181;">🛡️ Disclaimer</b>
        <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px; line-height:1.7;">
            This tool is <b>for informational purposes only</b> and does not constitute medical advice,
            diagnosis, or treatment. Always seek the guidance of a qualified mental health professional.
            If you are experiencing a mental health crisis, call <b>988</b> (US) or contact your local
            emergency services immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)
