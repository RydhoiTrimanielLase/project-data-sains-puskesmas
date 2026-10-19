"""
tests/test_data_loader.py — Pengujian modul data_loader.py

Menguji:
1. Pembacaan dataset utama.
2. Skema kolom dan tipe data.
3. Integritas data (null, duplikat, jumlah baris/provinsi).
4. Fungsi get_dataset_metadata().
5. Fungsi get_province_summary().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest

from src.data_loader import (
    load_main_data, get_dataset_metadata, get_province_summary,
    validate_upload_dataframe, COL, _REQUIRED_COLS,
)


class TestLoadMainData:
    """Pengujian load_main_data()."""

    @pytest.fixture(scope="class")
    def df(self):
        return load_main_data()

    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame), "Harus mengembalikan DataFrame"

    def test_row_count(self, df):
        assert len(df) == 154, f"Harus ada 154 baris, ditemukan {len(df)}"

    def test_required_columns_exist(self, df):
        for col in _REQUIRED_COLS:
            assert col in df.columns, f"Kolom '{col}' tidak ditemukan"

    def test_no_null_values(self, df):
        nulls = df[_REQUIRED_COLS].isnull().sum()
        cols_with_null = nulls[nulls > 0]
        assert len(cols_with_null) == 0, f"Nilai null ditemukan: {cols_with_null.to_dict()}"

    def test_no_duplicate_rows(self, df):
        assert df.duplicated().sum() == 0, "Ditemukan baris duplikat"

    def test_province_count(self, df):
        n_prov = df[COL["provinsi"]].nunique()
        assert n_prov == 10, f"Harus ada 10 provinsi, ditemukan {n_prov}"

    def test_numeric_columns_dtype(self, df):
        numeric_cols = [COL["penduduk"], COL["total_puskesmas"], COL["rasio"], COL["log_penduduk"]]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Kolom '{col}' bukan numerik"

    def test_no_negative_values(self, df):
        for col in [COL["non_ri"], COL["ri"], COL["total_puskesmas"], COL["penduduk"]]:
            assert (df[col] >= 0).all(), f"Nilai negatif ditemukan di kolom '{col}'"

    def test_total_puskesmas_consistency(self, df):
        """Jumlah Puskesmas = Non RI + RI."""
        computed = df[COL["non_ri"]] + df[COL["ri"]]
        diff = (computed - df[COL["total_puskesmas"]]).abs().max()
        assert diff == 0, f"Jumlah Puskesmas tidak konsisten, selisih max={diff}"

    def test_penduduk_positive(self, df):
        assert (df[COL["penduduk"]] > 0).all(), "Semua penduduk harus > 0"


class TestGetDatasetMetadata:
    """Pengujian get_dataset_metadata()."""

    @pytest.fixture(scope="class")
    def meta(self):
        df = load_main_data()
        return get_dataset_metadata(df)

    def test_returns_dict(self, meta):
        assert isinstance(meta, dict)

    def test_required_keys(self, meta):
        required_keys = [
            "n_wilayah", "n_provinsi", "total_penduduk", "total_puskesmas",
            "total_non_ri", "total_ri", "rasio_agregat_nasional",
            "wilayah_rasio_tertinggi", "rasio_tertinggi",
            "wilayah_rasio_terendah", "rasio_terendah",
        ]
        for k in required_keys:
            assert k in meta, f"Kunci '{k}' tidak ada di metadata"

    def test_n_wilayah(self, meta):
        assert meta["n_wilayah"] == 154

    def test_n_provinsi(self, meta):
        assert meta["n_provinsi"] == 10

    def test_total_puskesmas_nonzero(self, meta):
        assert meta["total_puskesmas"] > 0

    def test_rasio_tertinggi_greater_than_terendah(self, meta):
        assert meta["rasio_tertinggi"] > meta["rasio_terendah"]

    def test_total_puskesmas_equals_ri_plus_nonri(self, meta):
        assert meta["total_ri"] + meta["total_non_ri"] == meta["total_puskesmas"]


class TestGetProvinceSummary:
    """Pengujian get_province_summary()."""

    @pytest.fixture(scope="class")
    def summary(self):
        df = load_main_data()
        return get_province_summary(df)

    def test_returns_dataframe(self, summary):
        assert isinstance(summary, pd.DataFrame)

    def test_row_count(self, summary):
        assert len(summary) == 10, "Harus ada 10 provinsi"

    def test_rasio_agregat_column_exists(self, summary):
        assert "rasio_agregat" in summary.columns

    def test_total_provinces_match(self, summary):
        assert summary["total_puskesmas"].sum() > 0

    def test_rasio_agregat_computed_correctly(self, summary):
        """Verifikasi rasio = total_pkm / total_penduduk * 100000."""
        computed = summary["total_puskesmas"] / summary["total_penduduk"] * 100_000
        diff = (computed - summary["rasio_agregat"]).abs().max()
        assert diff < 1e-6, f"Rasio agregat tidak konsisten, selisih={diff}"


class TestValidateUploadDataframe:
    """Pengujian validate_upload_dataframe()."""

    def test_valid_data(self):
        df_valid = pd.DataFrame({
            "provinsi": ["Aceh"],
            "kab_kota": ["Test Kab"],
            "penduduk": [100000],
            "non_ri": [5],
            "ri": [3],
        })
        df_ok, errors, warnings = validate_upload_dataframe(df_valid)
        assert len(df_ok) == 1
        assert len(errors) == 0

    def test_missing_required_column(self):
        df_bad = pd.DataFrame({
            "provinsi": ["Aceh"],
            "kab_kota": ["Test"],
            "penduduk": [100000],
            "non_ri": [5],
            # 'ri' missing
        })
        df_ok, errors, warnings = validate_upload_dataframe(df_bad)
        assert len(errors) > 0

    def test_zero_penduduk(self):
        df_bad = pd.DataFrame({
            "provinsi": ["Aceh"],
            "kab_kota": ["Test Kab"],
            "penduduk": [0],  # invalid
            "non_ri": [5],
            "ri": [3],
        })
        df_ok, errors, warnings = validate_upload_dataframe(df_bad)
        assert len(df_ok) == 0
        assert len(errors) > 0

    def test_negative_puskesmas(self):
        df_bad = pd.DataFrame({
            "provinsi": ["Aceh"],
            "kab_kota": ["Test Kab"],
            "penduduk": [100000],
            "non_ri": [-1],  # invalid
            "ri": [3],
        })
        df_ok, errors, warnings = validate_upload_dataframe(df_bad)
        assert len(df_ok) == 0

    def test_mixed_valid_invalid(self):
        df_mix = pd.DataFrame({
            "provinsi": ["Aceh", "Sumut"],
            "kab_kota": ["Valid", "Invalid"],
            "penduduk": [100000, 0],  # baris ke-2 invalid
            "non_ri": [5, 3],
            "ri": [3, 2],
        })
        df_ok, errors, warnings = validate_upload_dataframe(df_mix)
        assert len(df_ok) == 1  # hanya 1 yang valid
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
