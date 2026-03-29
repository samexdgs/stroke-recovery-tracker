"""
model.py — Stroke Recovery Progress Tracker
============================================
Trains three ML models on a simulated clinical dataset of stroke survivors.
Saves the best model and scaler to disk for use by the Streamlit app.

Author: [Your Name]
Project: Stroke Recovery Progress Tracker
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import pickle
import os

# ─────────────────────────────────────────────
# 1. GENERATE SIMULATED CLINICAL DATASET
# ─────────────────────────────────────────────
# In a real deployment this would be replaced with actual patient data.
# Simulating 2,000 stroke survivor records with clinically plausible features.

def generate_dataset(n=2000, seed=42):
    np.random.seed(seed)

    # Patient background features
    age = np.random.randint(35, 85, n)
    days_since_stroke = np.random.randint(7, 365, n)
    affected_side = np.random.choice([0, 1], n)           # 0=left, 1=right
    stroke_type = np.random.choice([0, 1], n)             # 0=ischemic, 1=hemorrhagic

    # Daily self-reported symptoms (scale 1–10)
    pain_level        = np.random.randint(1, 11, n)
    fatigue_level     = np.random.randint(1, 11, n)
    spasticity_level  = np.random.randint(1, 11, n)
    balance_score     = np.random.randint(1, 11, n)       # higher = better balance
    mobility_score    = np.random.randint(1, 11, n)       # higher = better mobility

    # Exercises completed today (out of recommended set)
    exercises_done    = np.random.randint(0, 11, n)       # 0 = none, 10 = all
    exercise_duration = np.random.randint(0, 91, n)       # minutes

    # Sleep and mood
    sleep_quality     = np.random.randint(1, 11, n)
    mood_score        = np.random.randint(1, 11, n)

    # Assistive device usage
    uses_afo          = np.random.choice([0, 1], n)       # ankle-foot orthosis
    uses_cane         = np.random.choice([0, 1], n)
    uses_walker       = np.random.choice([0, 1], n)

    # ── Recovery status label (clinically informed rules + noise) ──
    # 0 = Needs attention, 1 = Plateauing, 2 = On track / improving
    recovery_score = (
        (mobility_score * 1.5)
        + (balance_score * 1.3)
        + (exercises_done * 1.2)
        + (sleep_quality * 0.8)
        + (mood_score * 0.6)
        - (pain_level * 1.1)
        - (fatigue_level * 0.9)
        - (spasticity_level * 0.7)
        + (days_since_stroke * 0.02)          # longer since stroke → further along
        - (age * 0.05)                         # older → slightly harder
        + np.random.normal(0, 3, n)            # clinical noise
    )

    # Bin into 3 classes
    labels = pd.cut(
        recovery_score,
        bins=[-np.inf, 12, 22, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    df = pd.DataFrame({
        "age": age,
        "days_since_stroke": days_since_stroke,
        "affected_side": affected_side,
        "stroke_type": stroke_type,
        "pain_level": pain_level,
        "fatigue_level": fatigue_level,
        "spasticity_level": spasticity_level,
        "balance_score": balance_score,
        "mobility_score": mobility_score,
        "exercises_done": exercises_done,
        "exercise_duration": exercise_duration,
        "sleep_quality": sleep_quality,
        "mood_score": mood_score,
        "uses_afo": uses_afo,
        "uses_cane": uses_cane,
        "uses_walker": uses_walker,
        "recovery_status": labels
    })
    return df


# ─────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────

def train_and_save():
    print("Generating dataset...")
    df = generate_dataset(2000)

    FEATURES = [
        "age", "days_since_stroke", "affected_side", "stroke_type",
        "pain_level", "fatigue_level", "spasticity_level",
        "balance_score", "mobility_score",
        "exercises_done", "exercise_duration",
        "sleep_quality", "mood_score",
        "uses_afo", "uses_cane", "uses_walker"
    ]

    X = df[FEATURES]
    y = df["recovery_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (needed for Logistic Regression)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ──
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)
    results["Logistic Regression"] = {
        "model": lr,
        "accuracy": accuracy_score(y_test, lr_pred),
        "roc_auc": roc_auc_score(y_test, lr_prob, multi_class="ovr"),
        "uses_scaler": True
    }

    # ── Random Forest ──
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    results["Random Forest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, rf_pred),
        "roc_auc": roc_auc_score(y_test, rf_prob, multi_class="ovr"),
        "uses_scaler": False
    }

    # ── LightGBM ──
    print("Training LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)
    results["LightGBM"] = {
        "model": lgbm,
        "accuracy": accuracy_score(y_test, lgbm_pred),
        "roc_auc": roc_auc_score(y_test, lgbm_prob, multi_class="ovr"),
        "uses_scaler": False
    }

    # ── Print results ──
    print("\n── Model Performance ──")
    for name, r in results.items():
        print(f"{name:25s}  Accuracy: {r['accuracy']:.3f}  ROC-AUC: {r['roc_auc']:.3f}")

    # ── Save best model (by ROC-AUC) ──
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")

    os.makedirs("model_artifacts", exist_ok=True)

    with open("model_artifacts/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    with open("model_artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata so the app knows which model won and whether to scale
    metadata = {
        "best_model_name": best_name,
        "uses_scaler": best["uses_scaler"],
        "features": FEATURES,
        "accuracy": best["accuracy"],
        "roc_auc": best["roc_auc"],
        "all_results": {
            k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"]}
            for k, v in results.items()
        }
    }
    with open("model_artifacts/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Saved to model_artifacts/")
    return metadata


if __name__ == "__main__":
    train_and_save()
