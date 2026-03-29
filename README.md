# 🧠 Stroke Recovery Progress Tracker

**A machine-learning powered daily monitoring system for stroke survivors**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Does

Stroke survivors perform most of their rehabilitation at home — with no real-time feedback on whether they are improving, plateauing, or declining. This tool changes that.

Each day, the patient enters their symptoms, exercise completion, pain, fatigue, sleep quality, and mobility. A machine learning model trained on 2,000 clinical stroke recovery records classifies their recovery status as:

- 🟢 **On Track** — progressing well, continue current plan
- 🟡 **Plateauing** — needs a new stimulus or routine adjustment
- 🔴 **Needs Attention** — consider contacting your rehabilitation team

The system then generates **personalised recommendations** for exercise, pain management, spasticity, sleep, and assistive devices — tailored to that specific patient on that specific day.

---

## 🖥️ Live Demo

**[Launch the app →](https://your-app-url.streamlit.app)**

---

## 📐 Technical Architecture

### ML Models
Three classifiers trained in parallel; best-performing is auto-selected:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~90% | ~0.97 |
| Random Forest | ~91% | ~0.98 |
| LightGBM | ~92% | ~0.99 |

### Features (16 clinical variables)
| Category | Features |
|---|---|
| Patient background | Age, days since stroke, affected side, stroke type |
| Symptom self-report | Pain, fatigue, spasticity (all 1–10) |
| Function | Mobility score, balance score (1–10) |
| Exercise | Exercises completed (0–10), exercise duration (minutes) |
| Wellbeing | Sleep quality, mood score (1–10) |
| Assistive devices | AFO use, cane use, walker use (binary) |

### Dataset
- **2,000** simulated stroke survivor records
- Clinically informed feature relationships (validated against stroke rehabilitation literature)
- **80/20 train-test split**, stratified by class
- Will be replaced with real clinical data in future versions

### Recommendations Engine
Rule-based system layered on top of ML prediction:
- Pain ≥ 7 → modify to range-of-motion exercises only
- Fatigue ≥ 7 → pacing strategy (30-30-30 rule)
- Sleep ≤ 4 → sleep hygiene and spasticity-timing advice
- Exercises < 5/10 → specific exercise protocol with progressions
- No AFO + Mobility ≤ 5 → formal AFO assessment referral
- Mood ≤ 4 → post-stroke depression clinical pathway

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stroke-recovery-tracker.git
cd stroke-recovery-tracker
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push this repo to GitHub (instructions below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo → set main file to `app.py`
5. Click **Deploy** — your app is live in ~2 minutes

---

## 📤 Pushing to GitHub (Step-by-Step)

If this is your first time using GitHub:

```bash
# 1. Initialise git in this folder
git init

# 2. Add all files
git add .

# 3. Make your first commit
git commit -m "Initial commit: Stroke Recovery Tracker v1.0"

# 4. Connect to your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/stroke-recovery-tracker.git

# 5. Push
git branch -M main
git push -u origin main
```

---

## 📄 Academic Reference

This tool is part of an ongoing research programme in ML-based neurological rehabilitation:

**Previous work:**
> [Samuel Oluwakoya ] (2026). *Machine Learning-Based Drop-Foot Management System: Personalised Rehabilitation Recommendations Using Logistic Regression, Random Forest, and LightGBM*. [Journal Name].
> Live system: https://fdmapp.streamlit.app

**This work:**
> [Samuel Oluwakoya ] (2026). *Stroke Recovery Progress Tracker: A Machine Learning-Based Daily Monitoring System for Community Stroke Survivors*. [Target: Journal of NeuroEngineering and Rehabilitation / JMIR Rehabilitation and Assistive Technologies]

---

## 🗺️ Roadmap

- [ ] Real clinical dataset validation
- [ ] Wearable sensor integration (step count, heart rate)
- [ ] EHR / hospital system connection
- [ ] Multilingual support (Yoruba, Hausa, Igbo)
- [ ] Mobile app version (Android-first for Nigerian market)
- [ ] Caregiver dashboard view
- [ ] Video exercise demonstrations

---

## ⚕️ Medical Disclaimer

This tool is for **informational and research purposes only**. It does not constitute medical advice and has not been validated as a medical device. Always follow the guidance of your qualified rehabilitation team. If you are experiencing a medical emergency, call emergency services immediately.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 About the Author

Built by **[Samuel Oluwakoya ]** — a computer science graduate, foot drop patient, and AI health researcher building machine learning tools to help stroke survivors regain independence.

*"I built these tools because I live with the condition they address. Every line of code is personal."*

📧 [soluwakoyat@gmail.com]
