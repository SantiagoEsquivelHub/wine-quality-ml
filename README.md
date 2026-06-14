# 🍷 Wine Quality ML — Proyecto 2 Inteligencia Artificial

Aplicación de Machine Learning que integra el ciclo completo de un proyecto de ML
sobre el dataset **Wine Quality (UCI)**: desde el preprocesamiento de datos hasta una
aplicación funcional de predicción.

**Curso:** Inteligencia Artificial — Escuela de Ingeniería de Sistemas y Computación
**Universidad del Valle**

---

## 📋 Descripción

El proyecto descubre **perfiles de vino** mediante clustering (aprendizaje no
supervisado) y luego entrena modelos supervisados para predecir el perfil de un vino
nuevo a partir de sus 11 propiedades fisicoquímicas.

### Fases
1. **Comprensión y preparación** del dataset (EDA + limpieza + escalado).
2. **Etiquetado no supervisado** con clustering (K-Means / jerárquico / DBSCAN).
3. **Modelado supervisado**: Regresión Logística, Random Forest y Red Neuronal.
4. **Aplicación** en Streamlit que predice el perfil de un vino.

---

## 🚀 Instalación y uso

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd wine-quality-ml

# 2. Crear y activar entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar los datos crudos en data/raw/
#    winequality-red.csv  y  winequality-white.csv
#    desde https://archive.ics.uci.edu/dataset/186/wine+quality

# 5. Ejecutar los notebooks en orden (01 -> 04) o la app:
streamlit run src/app.py
```

---

## 📁 Estructura del proyecto

```
wine-quality-ml/
├── data/
│   ├── raw/            # CSV originales (NO se modifican)
│   └── processed/      # datasets limpios y etiquetados (generados)
├── notebooks/          # exploración interactiva (EDA, clustering, modelos)
├── src/                # código de producción (pipeline + app)
├── models/             # modelos entrenados serializados
├── reports/            # informe y figuras
├── requirements.txt    # dependencias con versión fija
└── README.md
```

---

## 👥 Equipo

| Rol | Responsabilidad |
|---|---|
| Data Engineer | Fase 1 (EDA + preprocesamiento) |
| Unsupervised ML | Fase 2 (clustering + etiquetado) |
| Supervised ML | Fase 3 (modelos + red neuronal) |
| App + Reporte | Fase 4 (Streamlit + informe + video) |

---

## 📚 Referencia del dataset

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, 47(4), 547-553.
