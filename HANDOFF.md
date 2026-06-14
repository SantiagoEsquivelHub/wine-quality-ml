# 🗂️ HANDOFF — Bitácora del Proyecto 2 (Wine Quality ML)

> Documento vivo. Se actualiza al cerrar cada fase. Sirve para retomar el trabajo
> en cualquier momento sin perder contexto.

**Última actualización:** Los 4 notebooks (01–04) ejecutados; 13 figuras generadas.

---

## 0. Resumen rápido

- **Curso:** Inteligencia Artificial — Univalle. Proyecto 2 (35% de la nota).
- **Dataset:** Wine Quality (UCI), tintos + blancos combinados.
- **Idea central:** descubrir "perfiles de vino" con clustering (no supervisado),
  etiquetar con ellos, y entrenar modelos supervisados que predigan el perfil de
  un vino nuevo. App final en Streamlit.
- **Convención:** todo el CÓDIGO en inglés; explicaciones/documentos en español.
- **Reproducibilidad:** `random_state = 42` en todo el código.

---

## 1. Estado por fase

| Fase | Descripción | Código | Ejecutado | Artefacto generado |
|---|---|---|---|---|
| **1** | Preprocesamiento (limpieza, outliers, escalado, split) | ✅ | ✅ | `data/processed/wine_combined.csv` |
| **2** | Clustering + etiquetado (`cluster_label`) | ✅ | ✅ | `wine_labeled.csv`, `models/kmeans.pkl`, `models/cluster_scaler.pkl` |
| **3** | Modelos supervisados (LogReg, RF, Red Neuronal) + evaluación | ✅ | ✅ | `models/*.pkl`, `models/nn_model.h5`, `models/scaler.pkl`, figuras |
| **4** | App Streamlit | ✅ | ✅ | `models/label_mapping.json` (consume artefactos) |
| Notebooks | `01`–`04` completos | ✅ | ✅ | 13 figuras en `reports/figures/` |
| Informe | Documento escrito | ⬜ | — | `reports/informe.pdf` |
| Video | Demo ≤ 10 min | ⬜ | — | — |

**Avance global estimado:** ~93% (código + notebooks + figuras listos).

### Figuras generadas (reports/figures/) — 13 en total
- EDA (6): wine_type_counts, histograms, correlation, quality_distribution,
  red_vs_white, alcohol_vs_quality
- Preprocesamiento (2): prep_winsorize_before_after, prep_scaling_before_after
- Clustering (3): k_selection, pca_2d, profiles_heatmap
- Modelos (2): nn_learning_curves, confusion_matrix

---

## 2. Entorno de ejecución

- **Python correcto:** 3.12 vía entorno virtual. El `python` global del sistema es
  3.6 (NO usar).
- **Entorno virtual:** `wine-quality-ml/venv/` (creado con `py -3.12 -m venv venv`).
- **Ejecutar siempre con:** `.\venv\Scripts\python.exe <script>` desde la raíz
  `wine-quality-ml/`.

### Dependencias instaladas hasta ahora
- pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.0, joblib 1.4.2
- tensorflow 2.16.1, keras 3.14.1, matplotlib 3.8.4, seaborn 0.13.2
- ipykernel, nbconvert, nbformat (para ejecutar notebooks)

- streamlit 1.35.0 (Fase 4)

### Dependencias PENDIENTES de instalar
- plotly, shap (creatividad), imbalanced-learn (SMOTE) — opcionales
- NOTA: el metapaquete `jupyter` falla por límite de rutas largas de Windows.
  Usar `nbconvert` para ejecutar notebooks (ya funciona).

### Orden de ejecución del pipeline completo
1. `python src/preprocessing.py`         -> wine_combined.csv
2. `python src/clustering.py`             -> wine_labeled.csv + kmeans/cluster_scaler
3. `python src/make_label_mapping.py`     -> label_mapping.json
4. ejecutar notebook 04 (entrena modelos) -> *.pkl, *.h5, scaler.pkl, figuras
5. `streamlit run src/app.py`             -> app de predicción

---

## 3. Decisiones tomadas (justificables en el informe)

1. **Combinar tintos + blancos** en un solo dataset, añadiendo `wine_type`.
2. **Eliminar duplicados:** se quitaron 1.177 filas (18%). 6.497 → 5.320.
3. **Outliers por winsorización** (clip al percentil 99), no eliminación.
   Conserva todas las filas. Aplicado a `residual sugar` y `total sulfur dioxide`.
4. **`wine_type` codificado** como binario 0/1 (red=0, white=1), no One-Hot.
5. **Escalado StandardScaler**, `fit` solo en train, guardado con joblib.
6. **Split 70/15/15** (train/val/test), estratificado. Se mantienen 3 particiones
   (no 2) porque la red neuronal usa EarlyStopping sobre validation.
7. **Target del modelo supervisado = `cluster_label`** (de Fase 2), NO `quality`.
8. **Clustering sobre features ESCALADAS** (StandardScaler ajustado a todo el
   dataset). Sin escalar, el SO₂ dominaba las distancias. Scaler de clustering
   guardado aparte (`cluster_scaler.pkl`) del scaler de Fase 3.
9. **k = 4 clusters.** Mejor Davies-Bouldin (1.486), silueta cercana al máximo,
   4 perfiles interpretables. (k=2 solo separaría tinto/blanco.)

---

## 4. Resultados reales registrados

### Fase 1
- Filas tras combinar: **6.497**
- Filas tras quitar duplicados: **5.320**
- Distribución `wine_type`: 3.961 blancos / 1.359 tintos (desbalanceado).
- Valores faltantes: **0** (validado).
- Winsorización: `residual sugar` max 65.8 → **18.15**;
  `total sulfur dioxide` max 440 → **240.0**.

### Fase 2
- Clustering sobre 11 features escaladas. k elegido = **4**.
- Scores: silueta k=2 0.27, k=3 0.23, k=4 0.24; Davies-Bouldin mínimo en k=4 (1.486).
- Perfiles (nombres por centroides):
  - 0 = `sweet low-alcohol whites` (1.409)
  - 1 = `light acidic reds` (852)
  - 2 = `bold structured reds` (590)
  - 3 = `dry high-alcohol wines` (2.469)
- **Target desbalanceado** (~4:1) → justifica `class_weight="balanced"` y F1-macro.

### Fase 4
- App verificada: arranca, health check 200, predice y traduce el cluster a nombre
  legible vía `label_mapping.json`. Modelo cargado: `logreg_model.pkl`.
- Smoke test: vino con alcohol 13.5 / densidad baja -> "dry high-alcohol wines".

### Fase 3
- Split: train 3.724 / val 798 / test 798. Features = 11 + wine_type (12 total).
- Resultados en test (ordenado por F1-macro):
  - **Logistic Regression**: acc 0.995, F1-macro **0.993**, AUC 0.99998 — MEJOR.
  - Neural Network: acc 0.984, F1-macro 0.980 (paró en época 23 por EarlyStopping).
  - Random Forest: acc 0.965, F1-macro 0.959.
- **Modelo final = Logistic Regression** (mejor F1, más rápido y liviano).
  `app.py` ya apunta a `models/logreg_model.pkl`.
- ⚠️ OJO INFORME: accuracy ~99% porque el target deriva de las mismas features
  (K-Means). Tarea "fácil" → mencionarlo en Discusión/Limitaciones. Trabajo
  futuro: predecir `quality` real (problema difícil).

---

## 5. Próximo paso

➡️ **Redactar el informe escrito.** Toda la base está lista: usar la sección 3
(decisiones) como metodología, la sección 4 (resultados) para las tablas, y las 11
figuras de `reports/figures/`. Estructura mínima del informe en el plan original.

Pendientes finales:
- Informe escrito (PDF en `reports/`).
- Video ≤10 min (guion por tiempos en el plan original).

---

## 6. Estructura de archivos (referencia)

```
wine-quality-ml/
├── data/raw/            winequality-red.csv, winequality-white.csv  [descargados]
├── data/processed/      wine_combined.csv  [generado en Fase 1]
├── src/
│   ├── preprocessing.py   Fase 1
│   ├── clustering.py      Fase 2
│   ├── models.py          Fase 3 (definición + entrenamiento)
│   ├── evaluation.py      Fase 3 (métricas + comparación)
│   └── app.py             Fase 4
├── models/              .pkl / .h5  [se generan al entrenar]
├── notebooks/           01–04  [pendientes]
├── reports/             informe + figuras  [pendientes]
├── requirements.txt
├── README.md
└── HANDOFF.md           este archivo
```
