"""
models.py — K-Means Clustering dan Linear Regression untuk Proyek Puskesmas Sumatra 2022.

Modul ini berisi:
1. train_kmeans()          : Melatih K-Means dari data referensi.
2. save_kmeans() / load_kmeans() : Persistensi model K-Means.
3. predict_cluster()       : Prediksi cluster untuk input baru (pakai model terlatih).
4. elbow_method_data()     : Data untuk Elbow Method (inertia per K).
5. silhouette_scores_data(): Silhouette Score per K.
6. train_linear_regression(): Melatih Linear Regression dengan split 80:20.
7. get_cluster_profile()   : Profil karakteristik tiap cluster dari nilai aktual.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent
_KMEANS_PATH = _ROOT / "models" / "kmeans_model.pkl"
_LR_PATH = _ROOT / "models" / "lr_model.pkl"

# Kolom dataset untuk Regresi
REG_FEATURE_COL = "Jumlah Penduduk"
REG_TARGET_COL = "Jumlah Puskesmas"
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# K-Means
# ─────────────────────────────────────────────

def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 3) -> KMeans:
    """
    Melatih K-Means pada data referensi yang sudah distandarisasi.

    Parameters
    ----------
    X_scaled : np.ndarray
        Data terstandarisasi, shape (154, 3).
    n_clusters : int
        Jumlah cluster yang diinginkan.

    Returns
    -------
    KMeans
        Model K-Means yang sudah di-fit.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )
    kmeans.fit(X_scaled)
    return kmeans


def save_kmeans(model: KMeans) -> None:
    """Simpan model K-Means ke disk."""
    _KMEANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(_KMEANS_PATH))


def load_kmeans() -> KMeans | None:
    """Muat model K-Means dari disk. Kembalikan None jika belum ada."""
    if _KMEANS_PATH.exists():
        return joblib.load(str(_KMEANS_PATH))
    return None


def predict_cluster(X_new_scaled: np.ndarray, model: KMeans) -> int:
    """
    Prediksi cluster untuk satu baris input menggunakan model yang sudah dilatih.

    TIDAK melatih ulang model. Model dan scaler harus sudah di-load dari disk.

    Parameters
    ----------
    X_new_scaled : np.ndarray
        Array shape (1, 3) yang sudah distandarisasi.
    model : KMeans
        Model K-Means terlatih.

    Returns
    -------
    int
        Nomor cluster (0-indexed).
    """
    return int(model.predict(X_new_scaled)[0])


def elbow_method_data(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Hitung inertia untuk berbagai nilai K (Elbow Method).

    Returns
    -------
    dict
        {'k_values': [...], 'inertia': [...]}
    """
    k_values = list(k_range)
    inertia = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return {"k_values": k_values, "inertia": inertia}


def silhouette_scores_data(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Hitung Silhouette Score untuk berbagai nilai K.

    Returns
    -------
    dict
        {'k_values': [...], 'silhouette': [...]}
    """
    k_values = list(k_range)
    scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    return {"k_values": k_values, "silhouette": scores}


def get_cluster_labels(
    df: pd.DataFrame, X_scaled: np.ndarray, model: KMeans
) -> pd.DataFrame:
    """
    Tambahkan label cluster ke DataFrame dan kembalikan dengan kolom cluster.

    Label cluster diinterpretasikan dari nilai aktual (bukan diasumsikan).

    Returns
    -------
    pd.DataFrame
        Salinan df dengan kolom 'Cluster' (int) dan 'Label Cluster' (str).
    """
    df = df.copy()
    df["Cluster"] = model.predict(X_scaled)
    return df


def get_cluster_profile(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung profil karakteristik tiap cluster dari nilai aktual.

    Label ditetapkan berdasarkan nilai aktual rasio dan log penduduk,
    BUKAN diasumsikan dari nomor cluster.

    Returns
    -------
    pd.DataFrame
        Tabel profil cluster, diurutkan dari rasio tertinggi ke terendah.
    """
    profile = (
        df_clustered.groupby("Cluster")
        .agg(
            n_wilayah=("Kabupaten/Kota", "count"),
            avg_penduduk=("Jumlah Penduduk", "mean"),
            avg_puskesmas=("Jumlah Puskesmas", "mean"),
            avg_rasio=("Rasio Puskesmas per 100.000 Penduduk", "mean"),
            avg_proporsi_ri=("Proporsi Puskesmas Rawat Inap (%)", "mean"),
            avg_log_penduduk=("Log Jumlah Penduduk", "mean"),
        )
        .reset_index()
    )

    # Beri label berdasarkan rasio aktual (bukan asumsi)
    profile = profile.sort_values("avg_rasio", ascending=False).reset_index(drop=True)
    labels = []
    for rank, row in profile.iterrows():
        if rank == 0:
            label = "🔵 Rasio Tinggi"
        elif rank == len(profile) - 1:
            label = "🔴 Rasio Rendah"
        else:
            label = "🟡 Rasio Sedang"
        labels.append(label)
    profile["Label"] = labels

    return profile


def get_cluster_interpretation(cluster_id: int, profile_df: pd.DataFrame) -> dict:
    """
    Kembalikan interpretasi untuk satu cluster berdasarkan profil aktual.

    Parameters
    ----------
    cluster_id : int
        Nomor cluster yang ingin diinterpretasikan.
    profile_df : pd.DataFrame
        Hasil dari get_cluster_profile().

    Returns
    -------
    dict
        Dictionary berisi label, deskripsi, dan statistik cluster.
    """
    row = profile_df[profile_df["Cluster"] == cluster_id]
    if row.empty:
        return {"label": f"Cluster {cluster_id}", "deskripsi": "Data tidak tersedia."}

    row = row.iloc[0]
    return {
        "cluster_id": cluster_id,
        "label": row["Label"],
        "n_wilayah": int(row["n_wilayah"]),
        "avg_penduduk": float(row["avg_penduduk"]),
        "avg_puskesmas": float(row["avg_puskesmas"]),
        "avg_rasio": float(row["avg_rasio"]),
        "avg_proporsi_ri": float(row["avg_proporsi_ri"]),
    }


# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────

def train_linear_regression(df: pd.DataFrame) -> dict:
    """
    Melatih Linear Regression: Penduduk → Jumlah Puskesmas.

    Split: 80% training, 20% testing, random_state=42.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset referensi dengan kolom 'Jumlah Penduduk' dan 'Jumlah Puskesmas'.

    Returns
    -------
    dict
        Berisi:
        - 'model'     : LinearRegression terlatih
        - 'X_test'    : Array fitur test
        - 'y_test'    : Array target test
        - 'y_pred'    : Array prediksi test
        - 'mae'       : Mean Absolute Error
        - 'rmse'      : Root Mean Squared Error
        - 'r2'        : R² Score
        - 'intercept' : Intercept regresi
        - 'coef'      : Koefisien regresi
        - 'n_train'   : Jumlah data training
        - 'n_test'    : Jumlah data testing
    """
    X = df[[REG_FEATURE_COL]].values
    y = df[REG_TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Simpan model
    _LR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(_LR_PATH))

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "intercept": float(model.intercept_),
        "coef": float(model.coef_[0]),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def predict_puskesmas(penduduk: float, model: LinearRegression) -> float:
    """
    Estimasi jumlah Puskesmas berdasarkan jumlah penduduk.

    BUKAN ramalan masa depan — ini estimasi berdasarkan hubungan statistik
    antarwilayah di dataset 2022.

    Parameters
    ----------
    penduduk : float
        Jumlah penduduk.
    model : LinearRegression
        Model yang sudah dilatih.

    Returns
    -------
    float
        Estimasi jumlah Puskesmas.
    """
    x = np.array([[penduduk]])
    return float(model.predict(x)[0])


def load_lr_model() -> LinearRegression | None:
    """Muat model Linear Regression dari disk."""
    if _LR_PATH.exists():
        return joblib.load(str(_LR_PATH))
    return None
