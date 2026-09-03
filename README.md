# PMOS Intelligence Platform

*What if a screening tool could tell you not just "yes" or "no," but which parts of your health actually need attention — and be upfront about how sure it is?*

**Try it live:** [pmosproject-fskjdpg8r9nidgzocyzhmu.streamlit.app](https://pmosproject-fskjdpg8r9nidgzocyzhmu.streamlit.app/)
**Code:** [github.com/annnx7172-cell/PMOS_PROJECT](https://github.com/annnx7172-cell/PMOS_PROJECT)

> Not a medical device. Every number this platform produces is a model-based estimate on a 541-patient research dataset, not a diagnosis. Please don't use it to make real medical decisions.

## PCOS, PMOS — what's going on with the name?

You'll see both names in this repo. PCOS (Polycystic Ovary Syndrome) is the name almost everyone knows. PMOS — Polyendocrine Metabolic Ovarian Syndrome — is the name this project builds toward, reflecting a 2026 renaming discussion that better captures what the condition actually is: not just an ovary problem, but a metabolic and hormonal one that happens to show up in the ovaries too. Code, models and file names in this repo mostly use "PMOS"; think of it as the same condition under a name that's trying to describe it more honestly.

## The problem

PCOS/PMOS is genuinely hard to pin down. It doesn't announce itself with one clean symptom — it shows up as a *pattern*: a cycle that's a bit irregular, a hormone slightly outside range, some weight gain, some acne, follicle counts on an ultrasound that could mean something or could mean nothing on their own. No single test settles it. A clinician has to weigh the whole picture, and two clinicians looking at the same chart don't always land in the same place.

That's exactly the kind of problem a model is good at — not replacing the clinician's judgment, but doing the pattern-matching across dozens of features at once, and being explicit about *why* it landed where it did. And a "yes, PMOS" answer isn't the end of the story either. One patient with PMOS is mainly facing a metabolic risk down the road; another is mainly worried about fertility; another is carrying the psychological weight of visible symptoms nobody's addressing. Treating all of them the same misses most of what actually matters for what she should do next.

So this project isn't just "does she have it." It's: does she have it, what specifically does that put at risk, why does the model think so, and what's a reasonable next step to raise with a doctor.

## What it actually does

| Capability | What it answers |
|---|---|
| PMOS diagnosis | How likely is this patient to be PMOS positive? |
| Subtype exploration | Do PMOS patients cluster into distinct types, or is it more of a spectrum? |
| Risk scoring (×4) | Of metabolic, cardiovascular, reproductive and psychological risk, which ones does this profile actually put at stake? |
| Explainability | Which specific features pushed the diagnosis probability up or down? |
| Recommendations | Given the risk profile, what's worth raising with a doctor? |
| Ultrasound classification | Does this ovarian ultrasound look normal, PCO, or dominant-follicle? |

All served through one Streamlit dashboard — you fill in a form, not a spreadsheet.

## The data problem nobody mentions

Before any of the modeling happened, there was a quieter decision that mattered just as much: which dataset to actually trust. A few didn't make the cut, and it's worth saying why, because "we found more data" isn't automatically good news.

- **A synthetic dataset (PCOSGen) was rejected outright.** Generated data can look clean and still teach a model patterns that don't exist in real patients.
- **A 5-class Kaggle dataset was rejected** after turning up 2,284 exact duplicate images — about a third of the dataset was the same pictures counted twice. Training on that would have meant the model's reported accuracy was partly just memorizing repeats.
- **A binary ultrasound dataset was rejected** because a model hit AUC = 1.0 by epoch one. Perfect scores that arrive instantly are almost never a sign of a great model — they're a sign the task was accidentally made trivial (often because of near-duplicate images leaking between train and test).

None of these are exciting rejections. They're the unglamorous, unpaid work of not fooling yourself before you even start training — and they're a big part of why the numbers below are trustworthy rather than just impressive-looking.

## Datasets

- **Tabular clinical data (541 patients, 45 features):**  
  https://www.kaggle.com/prasoonkottarathil/polycystic-ovary-syndrome-pcos  
  Collected from 10 hospitals in Kerala, India.

- **Ovarian ultrasound images (Normal / PCO / Dominant Follicle):**  
  https://github.com/HananSaadat/ovarian_ultrasound_dataset  
  Introduced in Borna et al. (2025), *Frontiers in Physiology*, DOI: 10.3389/fphys.2025.1520898.

## How it fits together

```
Excel workbook (541 patients)
    |
Data ingestion -> cleaned CSV
    |
Feature selection -- chi2 / ANOVA / mutual information (diagnostic only),
                      then LassoCV on standardized features, then a VIF prune
    |
Class balancing -- SMOTETomek, training half only
    |
Diagnosis models -- Logistic Regression / Random Forest / SVM / XGBoost /
                     soft-voting ensemble
    |
Risk scoring -- 4 rule-derived labels, each learned by XGBoost with
                cross-validated out-of-sample predictions
    |
Clustering -- exploring whether PMOS patients form distinct subtypes
    |
SHAP explainer + recommendation engine
    |
(separate track) Ovarian ultrasounds -> MobileNetV2 CNN -> follicle classification
    |
Streamlit app -- Input -> Diagnosis -> Risk Dashboard -> Recommendations -> SHAP
```

Same modular philosophy throughout: ingestion, feature engineering, model training and risk scoring each live in their own file under `src/components/`. The app itself contains zero model code — it only ever calls into a serving-side pipeline that has zero Streamlit code. Swap the front end out entirely and the models underneath don't need to change.

## Choosing honesty over a better-looking number

A pattern runs through this project: several places where the "correct" answer looks worse on paper than the alternative, and the alternative was rejected anyway.

- **Reproductive Risk's AUC is the lowest of the four risk models (0.71 vs. 0.90–0.98 elsewhere) — on purpose.** Its label is partly defined by AMH and cycle regularity, so those two features were removed from what the model is allowed to see. Leaving them in would have pushed the score higher by letting the model read its own answer key. The lower number is the honest one.
- **PMOS subtypes are reported as a continuum, not discrete clusters.** The clustering analysis was run in good faith looking for distinct patient subtypes; the honest finding was that patients spread out more than they cleanly separate. Reporting "three clean subtypes" would have been a more exciting slide and a less true one.
- **The CNN's 80.4% accuracy is reported next to a published benchmark (76.2%), with the comparison flagged as approximate** — different studies split and preprocess their data differently, so a few points of difference shouldn't be read as "definitively better."
- **Marriage duration is a real model feature, but it's hidden from the SHAP explanation chart.** The dashboard has no sensible question to ask for it, so it's sent as a fixed placeholder — showing its "contribution" to a specific patient's prediction would be showing noise, not signal.
- **Pregnancy status was dropped as an input entirely.** It's a downstream consequence of PMOS-related fertility issues, not a cause — including it as a predictor would have let the model quietly cheat off an outcome instead of learning the underlying pattern.

None of this is about being modest for its own sake. A model that hides its own weak spots is a model you can't trust anywhere, including in the places it's actually strong.

## Results

### PMOS diagnosis

| Model | Accuracy | F1 | ROC-AUC | Recall (PMOS+) |
|---|---|---|---|---|
| SVM | 0.8899 | 0.8378 | 0.9587 | 0.8611 |
| Random Forest | 0.9174 | 0.8732 | 0.9585 | 0.8611 |
| XGBoost | 0.9266 | 0.8857 | 0.9650 | 0.8611 |
| Voting Ensemble | 0.9266 | 0.8889 | 0.9684 | 0.8889 |
| **Logistic Regression** | 0.9174 | 0.8800 | 0.9654 | **0.9167** ✅ |

**Why Logistic Regression, when the ensemble beats it on almost every other number?** Because in a screening tool, a missed positive costs more than a false alarm. A patient waved through as "probably fine" doesn't get the follow-up she needs; a patient flagged unnecessarily just has a conversation with her doctor that turns out reassuring. Logistic Regression catches 91.7% of true PMOS-positive patients against the ensemble's 88.9% — three more out of every hundred who don't slip through. That's the number worth protecting, even at a small cost elsewhere.

### Risk scoring (cross-validated, out-of-sample)

| Dimension | CV AUC | F1 | Accuracy |
|---|---|---|---|
| Metabolic Risk | 0.9066 | 0.8395 | 0.8466 |
| CVD Risk | 0.8991 | 0.5549 | 0.8577 |
| **Reproductive Risk** | **0.7055** | 0.6571 | 0.6470 |
| Psychological Risk | 0.9773 | 0.8564 | 0.9039 |

### Ultrasound classification

MobileNetV2, transfer-learned in two phases (frozen feature extraction, then fine-tuning the last 30 layers) on 304 images across three classes — Normal, PCO, Dominant Follicle.

- **Accuracy:** 80.4%
- **Macro AUC:** 0.9298
- Compared cautiously against a published ResNet18 benchmark of 76.2% accuracy on a similar task.

These numbers are as recorded from the original training run; the image set itself isn't part of this repo, so the run hasn't been reproduced locally.

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, NumPy |
| Feature selection | scikit-learn (LassoCV, chi2, ANOVA, mutual information), statsmodels (VIF) |
| Class balancing | imbalanced-learn (SMOTETomek) |
| Machine learning | scikit-learn (LR, RF, SVM), XGBoost, soft-voting ensembles |
| Explainability | SHAP (TreeExplainer) |
| Deep learning | TensorFlow / Keras, MobileNetV2 transfer learning (ultrasound track only) |
| Web app | Streamlit |
| Serialization | pickle |
| Version control | Git & GitHub |

## Run it yourself

```bash
# Dashboard — run from the repo root, src/ must be importable
venv/bin/streamlit run app.py

# Retrain the diagnosis + risk-scoring models against the raw data.
# Writes to artifacts/retrained/ by default and never touches the shipped models.
venv/bin/python -m src.pipeline.train_pipeline
venv/bin/python -m src.pipeline.train_pipeline --overwrite   # replace the shipped ones in place

# Quick sanity check against the shipped artifacts
venv/bin/python -m src.pipeline.predict_pipeline

# The exploratory notebooks
venv/bin/jupyter lab
```

A full training run reproduces the shipped diagnosis models and all four risk models bit-for-bit — verified by checksum, not just "it ran without errors."

## Where this could go next

- **Make the ultrasound track fully reproducible.** The CNN notebook exists, but its training images aren't in this repo. Committing them (or documenting how to source them) and adding TensorFlow to the deployment requirements would let the live dashboard actually classify ultrasounds instead of quietly showing "no classification."
- **Finish porting the notebook-only stages.** Clustering, the SHAP explainer and the recommendation engine still live only in notebooks. Diagnosis and risk scoring already made the jump to runnable, reproducible code — extending that to the rest would mean the entire platform, not just half of it, can be rebuilt from raw data on demand.
- **Validate the risk labels against real outcomes.** They're honestly derived given what's available, but they're still clinical rules standing in for ground truth. If longitudinal outcome data ever became available, checking these labels against what actually happened to real patients would be the real test of whether they hold up.

## Author

**Ananya Singh**
MSc Statistics and Computing (Machine Learning)
GitHub: [@annnx7172-cell](https://github.com/annnx7172-cell)
