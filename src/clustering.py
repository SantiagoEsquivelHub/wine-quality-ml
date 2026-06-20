"""
clustering.py
=============
Phase 2: unsupervised labeling of the Wine Quality dataset.

We don't have a "wine profile" label in the raw data, so we DISCOVER it with
clustering. K-Means groups wines by their physicochemical similarity; we then
interpret each group and turn it into the `cluster_label` column that Phase 3
(the supervised models) will try to predict.

Pipeline:
    1. Pick the number of clusters k (elbow + silhouette + Davies-Bouldin).
    2. Fit K-Means and (optionally) compare with Agglomerative / DBSCAN.
    3. Interpret clusters from their centroids and give them readable names.
    4. Attach `cluster_label` and persist the fitted model + labeled dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Reuse the same feature list and seed defined for Phase 1.
from preprocessing import FEATURE_COLUMNS, RANDOM_STATE


def evaluate_k_range(X: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """Try several values of k and score each one.

    Returns a table with three signals to choose k:
      - inertia: within-cluster sum of squares (used for the "elbow" plot;
        always decreases, so we look for the bend, not the minimum).
      - silhouette: how well-separated the clusters are (higher is better, max 1).
      - davies_bouldin: average similarity between each cluster and its closest
        one (lower is better).
    """
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X: pd.DataFrame, n_clusters: int,
               model_path: str = "models/kmeans.pkl") -> tuple[KMeans, np.ndarray]:
    """Fit the final K-Means model with the chosen k and persist it.

    Returns the fitted model and the cluster assignment for each row. The model
    is saved so the app can assign a cluster to brand-new wines later.
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, model_path)
    return km, labels


def compare_algorithms(X: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Compare K-Means against Agglomerative and DBSCAN on the same data.

    This is for the report: it shows we didn't just trust one algorithm. We score
    each by silhouette (DBSCAN may produce a different number of clusters and a
    noise group labeled -1, which is expected and worth discussing).
    """
    results = []

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X)
    results.append({"algorithm": "KMeans", "n_clusters": n_clusters,
                    "silhouette": silhouette_score(X, km_labels)})

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X)
    results.append({"algorithm": "Agglomerative", "n_clusters": n_clusters,
                    "silhouette": silhouette_score(X, agg_labels)})

    db = DBSCAN(eps=1.5, min_samples=10)
    db_labels = db.fit_predict(X)
    # DBSCAN labels noise as -1; count clusters excluding that group.
    n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    # Silhouette needs at least 2 clusters and is undefined if everything is noise.
    db_sil = silhouette_score(X, db_labels) if n_db >= 2 else float("nan")
    results.append({"algorithm": "DBSCAN", "n_clusters": n_db,
                    "silhouette": db_sil})

    return pd.DataFrame(results)


def describe_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute the mean of every feature per cluster (the cluster 'profile').

    These per-cluster averages are what let us NAME the clusters instead of
    leaving them as 0, 1, 2. A cluster with high alcohol and low density is a
    'powerful' profile; one with high residual sugar is a 'sweet' profile, etc.
    """
    profile = df[FEATURE_COLUMNS].copy()
    profile["cluster"] = labels
    return profile.groupby("cluster").mean()


def assign_labels(df: pd.DataFrame, labels: np.ndarray,
                  names: dict[int, str]) -> pd.DataFrame:
    """Map numeric cluster ids to human-readable names and attach the column.

    `names` is built by the analyst AFTER reading describe_clusters(), e.g.
        {0: "sweet low-alcohol whites", 1: "structured dry reds", ...}
    The resulting `cluster_label` column is the target for Phase 3.
    """
    df = df.copy()
    df["cluster"] = labels
    df["cluster_label"] = df["cluster"].map(names)
    return df


def project_2d(X: pd.DataFrame) -> np.ndarray:
    """Reduce the features to 2 dimensions with PCA for plotting the clusters.

    Clustering happens in 11-dimensional space, which we can't draw. PCA finds the
    2 directions that keep the most variance so we can visualize the groups on a
    flat scatter plot.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X)


if __name__ == "__main__":
    # Example Phase 2 flow (assumes Phase 1 already produced the combined dataset).
    df = pd.read_csv("data/processed/wine_combined.csv")

    # Cluster on the 11 features only (exclude quality and wine_type).
    # CRITICAL: scale first. K-Means uses distances, so without scaling the
    # large-magnitude features (e.g. total sulfur dioxide) would dominate and the
    # clusters would ignore the rest. We fit on the full dataset because clustering
    # is unsupervised exploration over all the data, and we save this scaler so the
    # app can assign clusters to brand-new wines consistently.
    cluster_scaler = StandardScaler()
    X = cluster_scaler.fit_transform(df[FEATURE_COLUMNS])
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(cluster_scaler, "models/cluster_scaler.pkl")

    scores = evaluate_k_range(X)
    print("Scores per k:")
    print(scores.to_string(index=False))

    # The chosen k comes from inspecting the table above. k=4 gives the best
    # Davies-Bouldin score, a silhouette close to the maximum, and 4 interpretable
    # profiles (k=2 would only separate red vs white).
    chosen_k = 4
    _, labels = fit_kmeans(X, n_clusters=chosen_k)

    print("\nCluster profiles (feature means, in ORIGINAL units):")
    # describe_clusters uses the unscaled df on purpose: real units are what let us
    # interpret and name the clusters (e.g. "alcohol = 12" is meaningful, a z-score
    # of 1.3 is not).
    print(describe_clusters(df, labels).to_string())

    # Human-readable names derived from the centroids above. These turn the numeric
    # clusters into the supervised target `cluster_label` for Phase 3.
    cluster_names = {
        0: "sweet low-alcohol whites",
        1: "light acidic reds",
        2: "bold structured reds",
        3: "dry high-alcohol wines",
    }
    df_labeled = assign_labels(df, labels, cluster_names)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv("data/processed/wine_labeled.csv", index=False)
    print("\nLabeled dataset saved to data/processed/wine_labeled.csv")
    print("\nLabel distribution:")
    print(df_labeled["cluster_label"].value_counts().to_string())
