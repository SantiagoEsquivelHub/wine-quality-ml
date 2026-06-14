# Video Script (max. 10 minutes) — Wine Quality ML

Total length: 10:00, split proportionally across 3 presenters (~3:20 each).
Each presenter owns one continuous block so transitions are clean.

---

## 👤 Presenter 1 — Introduction + Data (0:00 – 3:20)

**0:00 – 0:40 · Team and problem**
- Introduce the team and the course.
- State the goal: build a full ML pipeline on the Wine Quality dataset, from raw
  data to a working prediction app.
- Justify the dataset: 6.497 wines, 11 physicochemical features, real and public
  (UCI / Cortez et al. 2009).

**0:40 – 2:00 · Exploratory Data Analysis (Phase 1a)**
- Show `eda_histograms.png` and `eda_correlation.png`.
- Point out the strongest correlations (free/total SO₂ 0.72; density/alcohol -0.69).
- Show `eda_red_vs_white.png`: reds and whites differ structurally.

**2:00 – 3:20 · Preprocessing (Phase 1b)**
- Explain the cleaning decisions: 6.497 → 5.320 rows (removed ~18% duplicates).
- Show `prep_winsorize_before_after.png`: outliers clipped at the 99th percentile.
- Show `prep_scaling_before_after.png`: StandardScaler makes features comparable.
- Hand off: "Now [Presenter 2] will explain how we labeled the data."

---

## 👤 Presenter 2 — Clustering + Models (3:20 – 6:40)

**3:20 – 4:40 · Unsupervised labeling (Phase 2)**
- Explain why: the dataset has no "profile" label, so we discover it with K-Means.
- Show `clustering_k_selection.png`: how we chose k=4 (best Davies-Bouldin, good
  silhouette).
- Show `clustering_pca_2d.png`: the 4 clusters visualized in 2D.
- Show `clustering_profiles_heatmap.png`: name each profile from its centroid
  (sweet whites, light reds, bold reds, dry high-alcohol).

**4:40 – 6:40 · Supervised models (Phase 3)**
- Task: predict the cluster_label from the features.
- The three models: Logistic Regression, Random Forest, Neural Network.
- Show the NN learning curves (`nn_learning_curves.png`) — EarlyStopping at epoch 23.
- Show the comparison table: LogReg wins (F1-macro 0.993).
- Show `confusion_matrix.png`.
- Be honest: accuracy is high because the label derives from the same features
  (mention this as a limitation — shows critical thinking).
- Hand off: "Now [Presenter 3] will show the final application."

---

## 👤 Presenter 3 — Application + Conclusions (6:40 – 10:00)

**6:40 – 9:00 · Live app demo (Phase 4)**
- Run `streamlit run src/app.py`.
- Enter a wine's properties with the sliders (e.g. high alcohol, low density).
- Show the predicted profile and the probability bar chart.
- Explain how the app reuses the SAME scaler and model trained earlier (the
  consistency that makes predictions valid).

**9:00 – 10:00 · Conclusions**
- One reflection per model (LogReg / RF / NN) as the rubric asks.
- Why Logistic Regression was chosen as the final model (best F1, fastest, smallest).
- Limitations and future work: predict the real `quality` (a genuinely hard task).
- Close: the pipeline runs end-to-end and is fully reproducible (random_state=42).

---

## Recording tips
- Record with OBS Studio or Loom; share screen for figures and the live demo.
- Each presenter rehearses their block; keep strict to the time budget.
- Keep the app already running before the demo to avoid dead air.
