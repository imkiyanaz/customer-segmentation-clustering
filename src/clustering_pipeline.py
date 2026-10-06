"""Reusable clustering utilities extracted from the project notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame):
    """Drop incomplete rows, separate CUST_ID, and standardize numeric features."""
    clean = df.dropna(subset=["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]).copy()
    customer_id = clean.pop("CUST_ID") if "CUST_ID" in clean.columns else None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    return clean, customer_id, scaled, scaler


def evaluate_kmeans(scaled, k_values=range(2, 11)) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(scaled, labels)})
    return pd.DataFrame(rows)


def evaluate_gmm(scaled, k_values=range(2, 11)) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        labels = model.fit_predict(scaled)
        rows.append({"n_components": k, "bic": model.bic(scaled), "silhouette": silhouette_score(scaled, labels)})
    return pd.DataFrame(rows)


def evaluate_dbscan(scaled, eps_values=(0.4, 0.6, 0.8, 1.0, 1.2), min_samples_values=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            core_mask = labels != -1
            score = np.nan
            if n_clusters >= 2 and core_mask.sum() > 0:
                score = silhouette_score(scaled[core_mask], labels[core_mask])
            rows.append({"eps": eps, "min_samples": min_samples, "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": score})
    return pd.DataFrame(rows)


def fit_final_kmeans(scaled, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
    return model, model.fit_predict(scaled)
