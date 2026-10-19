"""
preprocessing.py — Preprocessing dan feature engineering untuk K-Means Clustering.

Alur:
1. Ekstraksi tiga fitur dari dataset: log_penduduk, rasio, proporsi_ri.
2. Fit StandardScaler pada data referensi 154 wilayah.
3. Simpan scaler menggunakan joblib agar input baru dapat ditransformasi secara konsisten.
4. DILARANG melakukan fit ulang scaler terhadap satu baris input pengguna.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Path penyimpanan scaler
_ROOT = Path(__file__).parent.parent
_SCALER_PATH = _ROOT / "models" / "scaler.pkl"

# Nama kolom fitur K-Means (sesuai COL di data_loader)
FEATURE_COLS = {
    "log_penduduk": "Log Jumlah Penduduk",
    "rasio": "Rasio Puskesmas per 100.000 Penduduk",
    "proporsi_ri": "Proporsi Puskesmas Rawat Inap (%)",
}

FEATURE_DISPLAY_NAMES = [
    "ln(1 + Jumlah Penduduk)",
    "Rasio Puskesmas per 100.000 Penduduk",
    "Proporsi Puskesmas Rawat Inap (%)",
]


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Ekstraksi tiga fitur K-Means dari DataFrame dataset.

    Kolom yang digunakan:
    - 'Log Jumlah Penduduk'      : ln(1 + Jumlah Penduduk), sudah ada di dataset.
    - 'Rasio Puskesmas per 100.000 Penduduk'
    - 'Proporsi Puskesmas Rawat Inap (%)'

    Returns
    -------
    np.ndarray
        Array shape (n_samples, 3).
    """
    return df[list(FEATURE_COLS.values())].values.astype(float)


def compute_derived_features(
    penduduk: float,
    non_ri: int,
    ri: int,
) -> dict:
    """
    Hitung variabel turunan dari input pengguna (satu baris).

    Parameters
    ----------
    penduduk : float
        Jumlah penduduk wilayah.
    non_ri : int
        Jumlah Puskesmas non rawat inap.
    ri : int
        Jumlah Puskesmas rawat inap.

    Returns
    -------
    dict
        Dictionary berisi semua variabel turunan.
    """
    total = non_ri + ri
    rasio = (total / penduduk) * 100_000 if penduduk > 0 else 0.0
    proporsi_ri = (ri / total * 100) if total > 0 else 0.0
    # Gunakan rumus yang sama dengan dataset: ln(1 + penduduk)
    log_penduduk = np.log1p(penduduk)

    return {
        "total_puskesmas": total,
        "rasio": rasio,
        "proporsi_ri": proporsi_ri,
        "log_penduduk": log_penduduk,
    }


def compute_derived_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung variabel turunan untuk DataFrame upload pengguna.
    Kolom wajib: penduduk, non_ri, ri.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom turunan ditambahkan.
    """
    df = df.copy()
    df["total_puskesmas"] = df["non_ri"] + df["ri"]
    df["rasio"] = np.where(
        df["penduduk"] > 0,
        (df["total_puskesmas"] / df["penduduk"]) * 100_000,
        0.0,
    )
    df["proporsi_ri"] = np.where(
        df["total_puskesmas"] > 0,
        (df["ri"] / df["total_puskesmas"]) * 100,
        0.0,
    )
    df["log_penduduk"] = np.log1p(df["penduduk"])
    return df


def fit_and_save_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler pada data referensi dan simpan ke disk.

    PENTING: Fungsi ini hanya dipanggil sekali saat inisialisasi model.
    Scaler yang tersimpan digunakan untuk semua transformasi input berikutnya.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset referensi 154 wilayah (Data_Gabungan).

    Returns
    -------
    StandardScaler
        Scaler yang sudah di-fit.
    """
    X = extract_features(df)
    scaler = StandardScaler()
    scaler.fit(X)
    _SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(_SCALER_PATH))
    return scaler


def load_scaler() -> StandardScaler | None:
    """
    Muat scaler yang sudah tersimpan.

    Returns
    -------
    StandardScaler or None
        None jika file scaler belum ada.
    """
    if _SCALER_PATH.exists():
        return joblib.load(str(_SCALER_PATH))
    return None


def transform_reference_data(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Transformasi data referensi menggunakan scaler yang sudah di-fit.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset referensi.
    scaler : StandardScaler
        Scaler yang sudah di-fit (dari fit_and_save_scaler atau load_scaler).

    Returns
    -------
    np.ndarray
        Data terstandarisasi, shape (154, 3).
    """
    X = extract_features(df)
    return scaler.transform(X)


def transform_single_input(features_dict: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Transformasi satu baris input pengguna menggunakan scaler referensi.

    TIDAK melakukan fit ulang. Scaler harus sudah di-load dari disk.

    Parameters
    ----------
    features_dict : dict
        Dictionary dengan kunci 'log_penduduk', 'rasio', 'proporsi_ri'.
    scaler : StandardScaler
        Scaler referensi yang sudah di-fit.

    Returns
    -------
    np.ndarray
        Array shape (1, 3) yang sudah distandarisasi.
    """
    x = np.array([[
        features_dict["log_penduduk"],
        features_dict["rasio"],
        features_dict["proporsi_ri"],
    ]])
    return scaler.transform(x)


def transform_upload_dataframe(df_upload: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Transformasi DataFrame upload menggunakan scaler referensi.

    Parameters
    ----------
    df_upload : pd.DataFrame
        DataFrame sudah ada kolom log_penduduk, rasio, proporsi_ri.
    scaler : StandardScaler
        Scaler referensi.

    Returns
    -------
    np.ndarray
        Array terstandarisasi shape (n, 3).
    """
    X = df_upload[["log_penduduk", "rasio", "proporsi_ri"]].values.astype(float)
    return scaler.transform(X)
