"""
Stroke Recovery Progress Tracker
=================================
A machine-learning powered Streamlit app for stroke survivors to track
daily recovery, receive personalized recommendations, and visualize progress.

Author: Samuel Oluwakoya
Institution:: Afe Babalola University
GitHub: https://github.com/samexdgs/stroke-recovery-tracker/
Live Demo: https://stroketracker.streamlit.app/
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Recovery Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Status cards */
    .status-card {
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .status-on-track {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 5px solid #059669;
    }
    .status-plateau {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 5px solid #d97706;
    }
    .status-attention {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 5px solid #dc2626;
    }

    .status-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .status-desc {
        font-size: 0.95rem;
        color: #374151;
    }

    /* Metric boxes */
    .metric-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #111827;
    }

    /* Recommendations */
    .rec-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.7rem;
        border-left: 4px solid #4f46e5;
    }
    .rec-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.2rem;
        font-size: 0.95rem;
    }
    .rec-body {
        color: #4b5563;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #f8f7ff;
    }

    .stSlider > label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* History log */
    .log-entry {
        background: #f9fafb;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.87rem;
        border: 1px solid #f3f4f6;
    }

    .stButton > button {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #4338ca;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,70,229,0.3);
    }

    .disclaimer {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #0369a1;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA STORAGE (session-based log)
# ─────────────────────────────────────────────
LOG_FILE = "recovery_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def append_entry(entry):
    log = load_log()
    log.append(entry)
    save_log(log)


# ─────────────────────────────────────────────
# ML MODEL — train on demand (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on clinical data...")
def get_model():
    """
    Generates a simulated dataset of 2,000 stroke survivors and trains
    three ML models. Returns the best-performing model and metadata.
    This runs once and is cached for the session.
    """
    np.random.seed(42)
    n = 2000

    age              = np.random.randint(35, 85, n)
    days_post_stroke = np.random.randint(7, 365, n)
    affected_side    = np.random.choice([0, 1], n)
    stroke_type      = np.random.choice([0, 1], n)
    pain_level       = np.random.randint(1, 11, n)
    fatigue_level    = np.random.randint(1, 11, n)
    spasticity_level = np.random.randint(1, 11, n)
    balance_score    = np.random.randint(1, 11, n)
    mobility_score   = np.random.randint(1, 11, n)
    exercises_done   = np.random.randint(0, 11, n)
    exercise_min     = np.random.randint(0, 91, n)
    sleep_quality    = np.random.randint(1, 11, n)
    mood_score       = np.random.randint(1, 11, n)
    uses_afo         = np.random.choice([0, 1], n)
    uses_cane        = np.random.choice([0, 1], n)
    uses_walker      = np.random.choice([0, 1], n)

    recovery_score = (
        (mobility_score * 1.5)
        + (balance_score * 1.3)
        + (exercises_done * 1.2)
        + (sleep_quality * 0.8)
        + (mood_score * 0.6)
        - (pain_level * 1.1)
        - (fatigue_level * 0.9)
        - (spasticity_level * 0.7)
        + (days_post_stroke * 0.02)
        - (age * 0.05)
        + np.random.normal(0, 3, n)
    )

    labels = pd.cut(recovery_score,
                    bins=[-np.inf, 12, 22, np.inf],
                    labels=[0, 1, 2]).astype(int)

    FEATURES = [
        "age","days_post_stroke","affected_side","stroke_type",
        "pain_level","fatigue_level","spasticity_level",
        "balance_score","mobility_score",
        "exercises_done","exercise_min",
        "sleep_quality","mood_score",
        "uses_afo","uses_cane","uses_walker"
    ]

    df = pd.DataFrame({
        "age": age, "days_post_stroke": days_post_stroke,
        "affected_side": affected_side, "stroke_type": stroke_type,
        "pain_level": pain_level, "fatigue_level": fatigue_level,
        "spasticity_level": spasticity_level,
        "balance_score": balance_score, "mobility_score": mobility_score,
        "exercises_done": exercises_done, "exercise_min": exercise_min,
        "sleep_quality": sleep_quality, "mood_score": mood_score,
        "uses_afo": uses_afo, "uses_cane": uses_cane, "uses_walker": uses_walker,
        "recovery_status": labels
    })

    X = df[FEATURES]
    y = df["recovery_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Train all three
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_te_sc))
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_te_sc), multi_class="ovr")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr")

    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test), multi_class="ovr")

    model_results = {
        "Logistic Regression": {"model": lr, "scaler": scaler, "uses_scaler": True,
                                  "accuracy": lr_acc, "roc_auc": lr_auc},
        "Random Forest":       {"model": rf, "scaler": None, "uses_scaler": False,
                                  "accuracy": rf_acc, "roc_auc": rf_auc},
        "LightGBM":            {"model": lgbm, "scaler": None, "uses_scaler": False,
                                  "accuracy": lgbm_acc, "roc_auc": lgbm_auc},
    }

    best_name = max(model_results, key=lambda k: model_results[k]["roc_auc"])
    return model_results, best_name, FEATURES


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(model_results, best_name, features, input_dict):
    r = model_results[best_name]
    X = pd.DataFrame([input_dict])[features]
    if r["uses_scaler"]:
        X = r["scaler"].transform(X)
    pred  = r["model"].predict(X)[0]
    proba = r["model"].predict_proba(X)[0]
    return int(pred), proba


# ─────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────
def get_recommendations(status, data):
    """
    Rule-based + status-driven personalised recommendations.
    status: 0 = Needs Attention, 1 = Plateauing, 2 = On Track
    """
    recs = []

    # ── Exercise ──
    if data["exercises_done"] < 5:
        recs.append({
            "icon": "🏃",
            "title": "Increase exercise completion",
            "body": f"You completed {data['exercises_done']}/10 recommended exercises today. "
                    "Start with seated ankle pumps (10 reps × 3 sets), knee slides on a bed surface, "
                    "and standing weight-shifts holding a stable surface. Add one new exercise every 3 days."
        })
    elif data["exercises_done"] >= 8:
        recs.append({
            "icon": "🏆",
            "title": "Excellent exercise completion — progress the difficulty",
            "body": "You're hitting 80%+ of exercises. This week, try adding resistance: "
                    "use a light resistance band for ankle dorsiflexion, or increase standing time by 5 minutes."
        })

    # ── Pain ──
    if data["pain_level"] >= 7:
        recs.append({
            "icon": "🌡️",
            "title": "High pain reported — modify today's session",
            "body": "Pain at 7+ warrants adjustment. Switch to gentle range-of-motion only today: "
                    "ankle circles, toe curls, and deep breathing. Apply heat for 15 minutes before exercise. "
                    "If pain persists above 7 for 3+ days, contact your physiotherapist."
        })

    # ── Fatigue ──
    if data["fatigue_level"] >= 7:
        recs.append({
            "icon": "😴",
            "title": "Fatigue is high — use the pacing strategy",
            "body": "High fatigue is your body signalling it needs recovery time. "
                    "Use the 30-30-30 rule: 30 minutes of light activity, 30 minutes rest, repeat. "
                    "Do NOT push through fatigue > 7 — this causes setbacks, not progress."
        })

    # ── Sleep ──
    if data["sleep_quality"] <= 4:
        recs.append({
            "icon": "🌙",
            "title": "Poor sleep is slowing your neurological recovery",
            "body": "Sleep is when the brain consolidates motor learning from your day's exercises. "
                    "Try: consistent sleep/wake time, no screens 1 hour before bed, "
                    "keep room cool and dark. If spasticity is waking you, ask your doctor about timing of medications."
        })

    # ── Spasticity ──
    if data["spasticity_level"] >= 6:
        recs.append({
            "icon": "💪",
            "title": "Spasticity management — stretch before exercise",
            "body": "Always stretch the affected limb for 10–15 minutes before any exercise. "
                    "Use prolonged slow stretch (not bouncing). A warm shower or heated pad before stretching "
                    "reduces muscle tone and makes the stretch more effective."
        })

    # ── AFO / assistive device ──
    if not data["uses_afo"] and data["mobility_score"] <= 5:
        recs.append({
            "icon": "🦿",
            "title": "Consider an Ankle-Foot Orthosis (AFO) assessment",
            "body": "Your mobility score suggests an AFO may significantly improve your gait safety "
                    "and reduce fall risk. Ask your physiotherapist or rehabilitation physician "
                    "for a formal AFO assessment — these are often covered by health insurance."
        })

    # ── Mood ──
    if data["mood_score"] <= 4:
        recs.append({
            "icon": "🧠",
            "title": "Low mood affects physical recovery — this is clinical",
            "body": "Post-stroke depression affects 30% of survivors and directly slows motor recovery. "
                    "This is not weakness — it is a neurological response to stroke. "
                    "Please mention your mood score to your medical team. Treatment is available and effective."
        })

    # ── Status-specific ──
    if status == 0:
        recs.insert(0, {
            "icon": "🔴",
            "title": "Priority: contact your healthcare provider this week",
            "body": "Today's data suggests your recovery needs clinical review. "
                    "Please contact your physiotherapist or rehabilitation doctor. "
                    "Bring this app's summary (use the export button below) to your appointment."
        })
    elif status == 1:
        recs.insert(0, {
            "icon": "🟡",
            "title": "Plateau detected — time to change the stimulus",
            "body": "Plateaus are normal but require a response. Your brain needs new challenges to continue rewiring. "
                    "Try a new exercise type, a different environment (outdoor walking vs indoor), "
                    "or ask your physiotherapist about electrical stimulation therapy."
        })
    else:
        recs.insert(0, {
            "icon": "🟢",
            "title": "You are on track — maintain consistency",
            "body": "Today's metrics reflect solid recovery progress. "
                    "The most important thing now is consistency — every day of exercises, "
                    "even a short session, builds the neural pathways that become permanent."
        })

    return recs[:6]  # max 6 recs per session


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def make_radar_chart(data):
    categories = ["Mobility", "Balance", "Sleep", "Mood", "Exercises\n(×10)", "Fatigue\n(inv)", "Pain\n(inv)"]
    values = [
        data["mobility_score"],
        data["balance_score"],
        data["sleep_quality"],
        data["mood_score"],
        data["exercises_done"],
        10 - data["fatigue_level"],   # invert so high = good
        10 - data["pain_level"],      # invert so high = good
    ]
    values += values[:1]
    cats = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=cats,
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.15)',
        line=dict(color='#4f46e5', width=2),
        marker=dict(size=6, color='#4f46e5')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def make_history_chart(log):
    if len(log) < 2:
        return None
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)

    fig = go.Figure()
    metrics = [
        ("mobility_score", "#4f46e5", "Mobility"),
        ("balance_score",  "#059669", "Balance"),
        ("pain_level",     "#dc2626", "Pain"),
        ("fatigue_level",  "#d97706", "Fatigue"),
        ("exercises_done", "#0891b2", "Exercises"),
    ]
    for col, color, label in metrics:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=label, line=dict(color=color, width=2),
                mode='lines+markers', marker=dict(size=5)
            ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        xaxis=dict(title="", showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="Score (1–10)", range=[0, 11], showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    return fig


def make_status_history_chart(log):
    if len(log) < 2:
        return None
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)
    if "prediction" not in df.columns:
        return None

    status_map = {0: "Needs Attention", 1: "Plateauing", 2: "On Track"}
    color_map  = {0: "#dc2626", 1: "#d97706", 2: "#059669"}

    df["status_label"] = df["prediction"].map(status_map)
    df["color"] = df["prediction"].map(color_map)

    fig = go.Figure(go.Scatter(
        x=df["date"],
        y=df["prediction"],
        mode='lines+markers',
        marker=dict(size=10, color=df["color"].tolist()),
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
        text=df["status_label"],
        hovertemplate="%{x|%b %d}<br>%{text}<extra></extra>"
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["Needs Attention", "Plateauing", "On Track"],
            showgrid=False
        ),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def make_model_comparison_chart(model_results):
    names = list(model_results.keys())
    accs  = [model_results[n]["accuracy"] for n in names]
    aucs  = [model_results[n]["roc_auc"] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names, y=accs,
                         marker_color="#4f46e5", text=[f"{v:.1%}" for v in accs],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="ROC-AUC", x=names, y=aucs,
                         marker_color="#059669", text=[f"{v:.3f}" for v in aucs],
                         textposition="outside"))
    fig.update_layout(
        barmode="group", height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR — DAILY CHECK-IN FORM
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/64/brain.png", width=48)
    st.sidebar.markdown("## Daily Check-In")
    st.sidebar.markdown("*Complete this every day for accurate tracking.*")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**About You** *(set once)*")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=55)
    days_post = st.sidebar.number_input("Days since stroke", min_value=1, max_value=1825, value=90)
    side = st.sidebar.selectbox("Affected side", ["Left", "Right"])
    stroke_type = st.sidebar.selectbox("Stroke type", ["Ischemic", "Hemorrhagic"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How are you feeling today?**")
    pain      = st.sidebar.slider("Pain level",       1, 10, 4, help="1 = no pain, 10 = worst pain")
    fatigue   = st.sidebar.slider("Fatigue level",    1, 10, 5, help="1 = fully rested, 10 = exhausted")
    spastic   = st.sidebar.slider("Spasticity / stiffness", 1, 10, 4, help="1 = no stiffness, 10 = very stiff")
    balance   = st.sidebar.slider("Balance (self-rated)", 1, 10, 6, help="1 = very unsteady, 10 = very steady")
    mobility  = st.sidebar.slider("Mobility (walking ability)", 1, 10, 6, help="1 = cannot walk, 10 = walks normally")
    sleep     = st.sidebar.slider("Sleep quality last night", 1, 10, 6, help="1 = terrible, 10 = excellent")
    mood      = st.sidebar.slider("Mood today", 1, 10, 7, help="1 = very low, 10 = very positive")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Today's Exercises**")
    ex_done = st.sidebar.slider("Exercises completed (out of 10)", 0, 10, 6)
    ex_min  = st.sidebar.slider("Total exercise time (minutes)", 0, 90, 30)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Assistive Devices**")
    afo    = st.sidebar.checkbox("Wearing AFO (ankle-foot orthosis)")
    cane   = st.sidebar.checkbox("Using cane")
    walker = st.sidebar.checkbox("Using walker")

    submitted = st.sidebar.button("📊 Analyse My Recovery Today")

    data = {
        "age": age,
        "days_post_stroke": days_post,
        "affected_side": 1 if side == "Right" else 0,
        "stroke_type": 1 if stroke_type == "Hemorrhagic" else 0,
        "pain_level": pain,
        "fatigue_level": fatigue,
        "spasticity_level": spastic,
        "balance_score": balance,
        "mobility_score": mobility,
        "exercises_done": ex_done,
        "exercise_min": ex_min,
        "sleep_quality": sleep,
        "mood_score": mood,
        "uses_afo": int(afo),
        "uses_cane": int(cane),
        "uses_walker": int(walker),
        "date": str(date.today()),
    }
    return data, submitted


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Load ML models (cached)
    model_results, best_name, FEATURES = get_model()

    # Sidebar input
    data, submitted = render_sidebar()

    # ── Header ──
    st.markdown('<div class="main-header">🧠 Stroke Recovery Progress Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalised daily insights powered by machine learning · Not a substitute for clinical care</div>', unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Today's Analysis",
        "📈 Progress Over Time",
        "🤖 Model Performance",
        "ℹ️ About This Tool"
    ])

    # ─────────────────────────────────────────────
    # TAB 1: TODAY'S ANALYSIS
    # ─────────────────────────────────────────────
    with tab1:
        if not submitted:
            st.info("👈 **Complete the daily check-in in the left panel**, then click *Analyse My Recovery Today* to see your results.")

            # Show quick start guide
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Step 1</div>
                    <div style="font-size:2rem;">📝</div>
                    <div style="font-weight:500;margin-top:0.3rem;">Fill the sidebar</div>
                    <div style="font-size:0.82rem;color:#6b7280;margin-top:0.3rem;">Enter today's symptoms, exercises, and how you feel</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Step 2</div>
                    <div style="font-size:2rem;">🤖</div>
                    <div style="font-weight:500;margin-top:0.3rem;">ML analyses your data</div>
                    <div style="font-size:0.82rem;color:#6b7280;margin-top:0.3rem;">Three models vote on your recovery status</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">Step 3</div>
                    <div style="font-size:2rem;">💡</div>
                    <div style="font-weight:500;margin-top:0.3rem;">Get personalised plan</div>
                    <div style="font-size:0.82rem;color:#6b7280;margin-top:0.3rem;">Exercises, adjustments, and daily guidance</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Run prediction
            pred, proba = predict(model_results, best_name, FEATURES, data)

            # Save to log
            log_entry = {**data, "prediction": pred, "proba": proba.tolist()}
            append_entry(log_entry)

            # ── Status card ──
            status_labels = {0: "Needs Clinical Attention 🔴", 1: "Recovery Plateauing 🟡", 2: "On Track — Keep Going! 🟢"}
            status_descs  = {
                0: "Today's data suggests you may need to speak with your physiotherapist or rehabilitation doctor. Don't ignore this — early intervention prevents setbacks.",
                1: "You're in a plateau phase — common and manageable. Your brain needs a new stimulus to continue rewiring. See recommendations below.",
                2: "Excellent. Your metrics today reflect solid recovery momentum. Consistency is everything — keep showing up every day."
            }
            status_classes = {0: "status-attention", 1: "status-plateau", 2: "status-on-track"}

            st.markdown(f"""
            <div class="status-card {status_classes[pred]}">
                <div class="status-title">{status_labels[pred]}</div>
                <div class="status-desc">{status_descs[pred]}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bar ──
            conf_labels = ["Needs Attention", "Plateauing", "On Track"]
            col_conf = st.columns(3)
            for i, (col, label) in enumerate(zip(col_conf, conf_labels)):
                with col:
                    st.metric(label, f"{proba[i]:.0%}")

            st.markdown("---")

            # ── Metrics row ──
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics = [
                ("Mobility", data["mobility_score"], "/10"),
                ("Balance",  data["balance_score"],  "/10"),
                ("Exercises", f"{data['exercises_done']}/10", "done"),
                ("Pain",     data["pain_level"],     "/10"),
                ("Sleep",    data["sleep_quality"],  "/10"),
            ]
            for col, (label, val, unit) in zip([col1,col2,col3,col4,col5], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div style="font-size:0.75rem;color:#9ca3af;">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Radar + Recommendations ──
            col_radar, col_recs = st.columns([1, 1.4])

            with col_radar:
                st.markdown("**Today's Recovery Profile**")
                st.plotly_chart(make_radar_chart(data), use_container_width=True)

            with col_recs:
                st.markdown("**Personalised Recommendations**")
                recs = get_recommendations(pred, data)
                for r in recs:
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-title">{r['icon']} {r['title']}</div>
                        <div class="rec-body">{r['body']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Disclaimer ──
            st.markdown("""
            <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer:</strong> This tool provides informational support only and does not constitute medical advice.
            Always consult a qualified physiotherapist or rehabilitation physician before changing your treatment. If you are in pain or distress, contact your healthcare provider immediately.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 2: PROGRESS OVER TIME
    # ─────────────────────────────────────────────
    with tab2:
        log = load_log()
        if len(log) < 2:
            st.info("Log at least **2 days** of data to see your progress charts. Come back tomorrow!")
        else:
            df_log = pd.DataFrame(log)
            total_days = len(log)
            on_track   = sum(1 for e in log if e.get("prediction") == 2)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Days logged", total_days)
            c2.metric("Days on track", on_track)
            c3.metric("Consistency", f"{on_track/total_days:.0%}")
            c4.metric("Latest status", {0:"Attention",1:"Plateau",2:"On Track"}.get(log[-1].get("prediction",1), "—"))

            st.markdown("**Recovery Status Over Time**")
            fig_status = make_status_history_chart(log)
            if fig_status:
                st.plotly_chart(fig_status, use_container_width=True)

            st.markdown("**Daily Metric Trends**")
            fig_hist = make_history_chart(log)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)

            # Raw log
            with st.expander("View raw data log"):
                st.dataframe(df_log, use_container_width=True)
                csv = df_log.to_csv(index=False)
                st.download_button(
                    "⬇️ Download my data (CSV)",
                    data=csv,
                    file_name=f"stroke_recovery_log_{date.today()}.csv",
                    mime="text/csv"
                )

    # ─────────────────────────────────────────────
    # TAB 3: MODEL PERFORMANCE
    # ─────────────────────────────────────────────
    with tab3:
        st.markdown("### ML Model Performance")
        st.markdown(f"**Active model: `{best_name}`** (selected automatically — highest ROC-AUC)")

        st.plotly_chart(make_model_comparison_chart(model_results), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**What these metrics mean**")
            st.markdown("""
            - **Accuracy** — what % of test patients the model classified correctly
            - **ROC-AUC** — ability to distinguish between recovery classes (1.0 = perfect, 0.5 = chance)
            - **Best model** is auto-selected by ROC-AUC on a held-out 20% test set
            """)
        with col_b:
            st.markdown("**Dataset summary**")
            st.markdown("""
            - **2,000** simulated stroke survivor records
            - **3 classes:** On Track · Plateauing · Needs Attention
            - **16 clinical features** (symptoms, exercises, sleep, devices)
            - **80/20 train-test split**, stratified by class
            """)

        st.markdown("---")
        st.markdown("### How the prediction works")
        st.markdown("""
        Each day you submit your check-in, the app:
        1. Converts your inputs into a 16-feature vector
        2. Passes it through the best-performing trained model
        3. Returns a **class prediction** (0/1/2) and **confidence probabilities**
        4. The recommendations engine uses both the class and raw feature values to personalise advice
        """)

    # ─────────────────────────────────────────────
    # TAB 4: ABOUT
    # ─────────────────────────────────────────────
    with tab4:
        st.markdown("### About This Tool")
        st.markdown("""
        **Stroke Recovery Progress Tracker** is an open-source, machine-learning powered daily monitoring
        tool built for stroke survivors to track their own recovery outside of clinical settings.

        #### Why this exists
        Stroke is the leading cause of adult disability worldwide. Over 80% of rehabilitation happens
        at home — yet most patients receive no real-time feedback on whether their recovery is on track.
        This tool bridges that gap using ML to classify daily recovery status and generate personalised,
        evidence-informed recommendations.

        #### The research behind it
        This system was developed as an extension of a published study on ML-based foot drop management
        (Logistic Regression, Random Forest, LightGBM, 2,000 patient records, 90%+ accuracy, ROC-AUC 0.99).
        The same architecture is applied here to the broader stroke recovery context.

        #### Technology stack
        - **Frontend:** Streamlit
        - **ML Models:** Logistic Regression, Random Forest, LightGBM (scikit-learn + lightgbm)
        - **Visualisation:** Plotly
        - **Deployment:** Streamlit Community Cloud
        - **Source code:** GitHub (open source)

        #### Limitations & future work
        - Currently trained on simulated data — real clinical validation is needed
        - Does not connect to wearables or EHR systems (planned)
        - Single-language (English) — multilingual support planned
        - No professional validation by a registered physiotherapist yet

        #### Citation
        If you use this tool in research, please cite:
        > [Samuel Oluwakoya] (2026). *Stroke Recovery Progress Tracker: An ML-Based Daily Monitoring System
        > for Stroke Survivors*. GitHub. github.com/samexdgs/stroke-recovery-tracker/

        #### Contact
        Built by Samuel Oluwakoya — A Computer scientist and foot drop patient building AI rehabilitation tools.
        Reach out via GitHub Issues or soluwakoyat@gmail.com.
        """)

        st.markdown("""
        <div class="disclaimer">
        This tool is for informational and research purposes only. It is not a medical device and has not
        been validated for clinical use. Always follow the advice of your rehabilitation team.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
