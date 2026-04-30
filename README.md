https://stroketracker.streamlit.app/
# Stroke Recovery Progress Tracker

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--2126--0254-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-2126-0254)


A daily monitoring tool for stroke survivors built with machine learning. Patients log how they feel each day and the system tells them whether their recovery is on track, plateauing, or needs attention — then gives specific recommendations based on what they reported.

Built by Samuel Oluwakoya. This is the first tool in what has become a series of rehabilitation systems built from lived experience. Samuel has foot drop from a neurological condition and started building these tools because the feedback gap in home rehabilitation is real and largely unsolved by existing software.

---

## Why this exists

Most stroke survivors do the bulk of their rehabilitation at home, alone, with no way of knowing if what they are doing is working. Their next physiotherapy appointment might be weeks away. By the time a plateau is identified, weeks of recovery time have been lost.

This tool gives patients a daily signal. It is not a replacement for clinical care. It is the thing that sits between appointments and keeps both the patient and their family informed.

---

## What it does

The patient fills in a short daily form covering symptoms, exercises completed, blood pressure, sleep, mood, and mobility. Three machine learning classifiers analyse the input and produce a recovery classification:

- On Track, continue current plan
- Plateauing, something needs to change
- Needs Attention, contact your rehabilitation team

On top of the ML output, a rules-based recommendation engine generates specific advice tailored to that day's data. If pain is high it suggests modified exercise. If sleep is poor it gives spasticity-timing guidance. If mood is low it flags the post-stroke depression pathway.

---

## Models

Three classifiers are trained in parallel. The one with the highest ROC-AUC on the held-out test set is selected automatically.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~90% | ~0.97 |
| Random Forest | ~91% | ~0.98 |
| LightGBM | ~92% | ~0.99 |

Training data is 2,000 simulated stroke survivor records with clinically informed feature relationships, validated against stroke rehabilitation literature. The 80/20 train-test split is stratified by class.

16 clinical features are used: age, days since stroke, affected side, stroke type, pain, fatigue, spasticity, mobility, balance, exercises completed, exercise duration, sleep quality, mood, and whether the patient uses an AFO, cane, or walker.

---

## Tech stack

- Python 3.10
- Streamlit (web interface)
- scikit-learn (Logistic Regression, Random Forest)
- LightGBM
- Plotly (charts)
- pandas, numpy


## Where this fits in the wider project

This is the first tool in a series:

1. Foot Drop Management App (published, live at fdmapp.streamlit.app) — ML classification of foot drop severity with personalised rehabilitation guidance
2. Stroke Recovery Progress Tracker (this tool) — daily monitoring and recovery classification
3. Stroke Recovery Monitor v2 — adds family dashboard and email alerts for caregivers abroad
4. AFO Clinical Management Platform — dual-dashboard system for physiotherapists and patients with ML-based AFO prescription
5. NeuroKinetics — camera-based upper limb motor tracking using MediaPipe, no wearable hardware needed

Each builds on the previous one. The clinical gaps that motivated each tool came from reading the literature, from personal experience with foot drop, and from talking to patients and physiotherapists.

---

## Academic reference

Samuel Oluwakoya (2026). Stroke Recovery Progress Tracker: A Machine Learning System for Daily Rehabilitation Monitoring in Community Stroke Survivors. GitHub. https://github.com/samexdgs/stroke-recovery-tracker

Target journals: Journal of NeuroEngineering and Rehabilitation, JMIR Rehabilitation and Assistive Technologies.

---
## Disclaimer

This is a research tool. It has not been validated as a medical device and does not constitute clinical advice. Always follow the guidance of your physiotherapist or rehabilitation physician. If you are experiencing a medical emergency, call emergency services.

---
Samuel Oluwakoya — computer science graduate, foot drop patient, AI health researcher.

- Email: [soluwakoyat@gmail.com](mailto:soluwakoyat@gmail.com),
- ORCID: [0009-0008-2126-0254](https://orcid.org/0009-0008-2126-0254)
- GitHub: [github.com/samexdgs](https://github.com/samexdgs)
- LinkedIn: [linkedin.com/in/samueloluwakoya](https://linkedin.com/in/samueloluwakoya)
- Portfolio: [samueloluwakoya.netlify.app](https://samueloluwakoya.netlify.app)
