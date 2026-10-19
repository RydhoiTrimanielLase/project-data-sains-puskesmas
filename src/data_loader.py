"""
data_loader.py — Modul pemuatan dataset untuk Proyek Puskesmas Sumatra 2022.

Fungsi utama:
- Membaca Data_Gabungan dari dataset utama (tahap 4 revisi).
- Menyediakan cache Streamlit agar tidak membaca ulang setiap interaksi.
- Memvalidasi schema kolom wajib.
- Melaporkan metadata dataset untuk keperluan audit.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

# Path relatif terhadap root proyek (satu tingkat di atas src/)
_ROOT = Path(__file__).parent.parent
_MAIN_DATA_PATH = _ROOT / "Data" / "hasil_tahap4_normalisasi_puskesmas_2022_REVISI_TAHAP3_EDA.xlsx"
_MAIN_SHEET = "Data_Gabungan"

# Kolom wajib yang harus ada di Data_Gabungan
_REQUIRED_COLS = [
    "Provinsi",
    "Kode Kab/Kota",
    "Kabupaten/Kota",
    "Puskesmas Non Rawat Inap",
    "Puskesmas Rawat Inap",
    "Jumlah Puskesmas",
    "Provinsi_BPS",
    "Kode BPS",
    "Kabupaten/Kota BPS",
    "Jumlah Penduduk",
    "Rasio Puskesmas per 100.000 Penduduk",
    "Proporsi Puskesmas Rawat Inap (%)",
    "Log Jumlah Penduduk",
]

# Alias kolom untuk aplikasi (nama pendek → nama asli)
COL = {
    "provinsi": "Provinsi",
    "kode_kab": "Kode Kab/Kota",
    "kab_kota": "Kabupaten/Kota",
    "non_ri": "Puskesmas Non Rawat Inap",
    "ri": "Puskesmas Rawat Inap",
    "total_puskesmas": "Jumlah Puskesmas",
    "provinsi_bps": "Provinsi_BPS",
    "kode_bps": "Kode BPS",
    "kab_kota_bps": "Kabupaten/Kota BPS",
    "laki": "Penduduk Laki-laki",
    "perempuan": "Penduduk Perempuan",
    "penduduk": "Jumlah Penduduk",
    "url": "URL Sumber",
    "rasio": "Rasio Puskesmas per 100.000 Penduduk",
    "proporsi_ri": "Proporsi Puskesmas Rawat Inap (%)",
    "log_penduduk": "Log Jumlah Penduduk",
}


def _validate_schema(df: pd.DataFrame, source_name: str = "dataset") -> list[str]:
    """Kembalikan daftar kolom wajib yang tidak ditemukan."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"[data_loader] Kolom wajib tidak ditemukan di {source_name}: {missing}",
            stacklevel=2,
        )
    return missing


def load_main_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Muat Data_Gabungan dari dataset utama.

    Returns
    -------
    pd.DataFrame
        DataFrame 154 baris × 16 kolom, sudah divalidasi.

    Raises
    ------
    FileNotFoundError
        Jika file dataset utama tidak ditemukan.
    ValueError
        Jika kolom wajib tidak lengkap setelah pemuatan.
    """
    if not _MAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"File dataset utama tidak ditemukan: {_MAIN_DATA_PATH}\n"
            "Pastikan folder Data/ berisi file hasil_tahap4_normalisasi_puskesmas_2022_REVISI_TAHAP3_EDA.xlsx"
        )

    df = pd.read_excel(str(_MAIN_DATA_PATH), sheet_name=_MAIN_SHEET, engine="openpyxl")

    missing_cols = _validate_schema(df)
    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di dataset: {missing_cols}. "
            "Periksa file sumber atau kontak pengembang."
        )

    # Pastikan tipe numerik sudah benar
    num_cols = [
        COL["non_ri"], COL["ri"], COL["total_puskesmas"],
        COL["penduduk"], COL["rasio"], COL["proporsi_ri"], COL["log_penduduk"],
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_dataset_metadata(df: pd.DataFrame) -> dict:
    """
    Kembalikan metadata ringkasan dataset untuk ditampilkan di Dashboard.

    Returns
    -------
    dict
        Dictionary berisi statistik utama dataset.
    """
    total_non_ri = int(df[COL["non_ri"]].sum())
    total_ri = int(df[COL["ri"]].sum())
    total_puskesmas = int(df[COL["total_puskesmas"]].sum())
    total_penduduk = int(df[COL["penduduk"]].sum())
    n_provinsi = df[COL["provinsi"]].nunique()
    n_wilayah = len(df)

    rasio_col = COL["rasio"]
    rasio_agregat_nasional = total_puskesmas / total_penduduk * 100_000

    idx_max = df[rasio_col].idxmax()
    idx_min = df[rasio_col].idxmin()

    return {
        "n_wilayah": n_wilayah,
        "n_provinsi": n_provinsi,
        "total_penduduk": total_penduduk,
        "total_puskesmas": total_puskesmas,
        "total_non_ri": total_non_ri,
        "total_ri": total_ri,
        "rasio_agregat_nasional": rasio_agregat_nasional,
        "wilayah_rasio_tertinggi": df.loc[idx_max, COL["kab_kota"]],
        "rasio_tertinggi": df.loc[idx_max, rasio_col],
        "provinsi_rasio_tertinggi": df.loc[idx_max, COL["provinsi"]],
        "wilayah_rasio_terendah": df.loc[idx_min, COL["kab_kota"]],
        "rasio_terendah": df.loc[idx_min, rasio_col],
        "provinsi_rasio_terendah": df.loc[idx_min, COL["provinsi"]],
    }


def get_province_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung ringkasan agregat per provinsi.

    Rasio agregat dihitung sebagai:
        (Total Puskesmas Provinsi / Total Penduduk Provinsi) × 100.000
    BUKAN rata-rata sederhana rasio tiap kab/kota.

    Returns
    -------
    pd.DataFrame
        Tabel ringkasan provinsi dengan kolom agregat.
    """
    summary = (
        df.groupby(COL["provinsi"])
        .agg(
            jumlah_wilayah=(COL["kab_kota"], "count"),
            total_penduduk=(COL["penduduk"], "sum"),
            total_puskesmas=(COL["total_puskesmas"], "sum"),
            total_non_ri=(COL["non_ri"], "sum"),
            total_ri=(COL["ri"], "sum"),
        )
        .reset_index()
    )
    summary["rasio_agregat"] = (
        summary["total_puskesmas"] / summary["total_penduduk"] * 100_000
    )
    summary = summary.rename(columns={COL["provinsi"]: "Provinsi"})
    return summary.sort_values("rasio_agregat", ascending=False).reset_index(drop=True)


def validate_upload_dataframe(df_upload: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Validasi DataFrame dari upload pengguna.

    Parameters
    ----------
    df_upload : pd.DataFrame
        DataFrame hasil upload CSV/Excel.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        - df_valid : baris yang lolos validasi
        - errors   : daftar pesan error per baris
        - warnings : daftar pesan peringatan
    """
    errors = []
    row_warnings = []

    # Kolom wajib upload
    required_upload_cols = ["provinsi", "kab_kota", "penduduk", "non_ri", "ri"]
    missing = [c for c in required_upload_cols if c not in df_upload.columns]
    if missing:
        errors.append(f"Kolom wajib tidak ditemukan: {missing}")
        return pd.DataFrame(), errors, row_warnings

    df = df_upload.copy()

    # Konversi tipe
    for col in ["penduduk", "non_ri", "ri"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = []
    for i, row in df.iterrows():
        row_errors = []

        if pd.isna(row["penduduk"]) or pd.isna(row["non_ri"]) or pd.isna(row["ri"]):
            row_errors.append(f"Baris {i+1} ({row.get('kab_kota','?')}): Ada nilai kosong/tidak valid")

        elif row["penduduk"] <= 0:
            row_errors.append(f"Baris {i+1} ({row.get('kab_kota','?')}): Penduduk harus > 0")

        elif row["non_ri"] < 0 or row["ri"] < 0:
            row_errors.append(f"Baris {i+1} ({row.get('kab_kota','?')}): Jumlah Puskesmas tidak boleh negatif")

        elif int(row["non_ri"]) != row["non_ri"] or int(row["ri"]) != row["ri"]:
            row_errors.append(f"Baris {i+1} ({row.get('kab_kota','?')}): non_ri dan ri harus bilangan bulat")

        if row_errors:
            errors.extend(row_errors)
            bad_rows.append(i)

    df_valid = df.drop(index=bad_rows).copy()

    if df_valid.duplicated(subset=["kab_kota"]).any():
        dups = df_valid[df_valid.duplicated(subset=["kab_kota"], keep=False)]["kab_kota"].unique()
        row_warnings.append(f"Ditemukan nama wilayah duplikat: {list(dups)}")

    return df_valid, errors, row_warnings
