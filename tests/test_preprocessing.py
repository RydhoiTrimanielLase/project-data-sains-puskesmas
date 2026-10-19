"""
tests/test_preprocessing.py — Pengujian modul preprocessing.py

Menguji:
1. Konsistensi perhitungan variabel turunan.
2. Fit dan save scaler.
3. Transformasi data referensi.
4. Transformasi input baru (tidak fit ulang).
5. Transformasi DataFrame upload.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_main_data
from src.preprocessing import (
    extract_features,
    compute_derived_features,
    compute_derived_for_dataframe,
    fit_and_save_scaler,
    load_scaler,
    transform_reference_data,
    transform_single_input,
    transform_upload_dataframe,
    FEATURE_COLS,
)


@pytest.fixture(scope="module")
def df():
    return load_main_data()


@pytest.fixture(scope="module")
def scaler(df):
    return fit_and_save_scaler(df)


class TestExtractFeatures:
    def test_shape(self, df):
        X = extract_features(df)
        assert X.shape == (154, 3), f"Shape harus (154,3), ditemukan {X.shape}"

    def test_no_nan(self, df):
        X = extract_features(df)
        assert not np.isnan(X).any(), "Ada NaN di fitur"

    def test_correct_columns(self, df):
        """Verifikasi urutan kolom sesuai FEATURE_COLS."""
        X = extract_features(df)
        for i, col in enumerate(FEATURE_COLS.values()):
            expected = df[col].values.astype(float)
            np.testing.assert_array_almost_equal(X[:, i], expected, decimal=5)


class TestComputeDerivedFeatures:
    def test_basic_case(self):
        result = compute_derived_features(penduduk=200_000, non_ri=8, ri=5)
        assert result["total_puskesmas"] == 13
        assert abs(result["rasio"] - 13 / 200_000 * 100_000) < 1e-6
        assert abs(result["proporsi_ri"] - 5 / 13 * 100) < 1e-6
        assert abs(result["log_penduduk"] - np.log1p(200_000)) < 1e-6

    def test_no_rawat_inap(self):
        result = compute_derived_features(penduduk=100_000, non_ri=10, ri=0)
        assert result["proporsi_ri"] == 0.0
        assert result["total_puskesmas"] == 10

    def test_all_rawat_inap(self):
        result = compute_derived_features(penduduk=100_000, non_ri=0, ri=5)
        assert result["proporsi_ri"] == 100.0

    def test_large_population(self):
        """Test dengan populasi besar (seperti Medan)."""
        result = compute_derived_features(penduduk=2_494_512, non_ri=10, ri=31)
        assert result["total_puskesmas"] == 41
        assert result["rasio"] > 0

    def test_log_formula_matches_dataset(self, df):
        """Verifikasi log formula konsisten dengan Log Jumlah Penduduk di dataset."""
        for _, row in df.head(5).iterrows():
            computed = np.log1p(row["Jumlah Penduduk"])
            stored = row["Log Jumlah Penduduk"]
            assert abs(computed - stored) < 1e-4, (
                f"Log mismatch untuk {row['Kabupaten/Kota']}: "
                f"computed={computed:.4f}, stored={stored:.4f}"
            )


class TestComputeDerivedForDataframe:
    def test_returns_dataframe(self):
        df_in = pd.DataFrame({
            "provinsi": ["Aceh"],
            "kab_kota": ["Test"],
            "penduduk": [200_000],
            "non_ri": [8],
            "ri": [5],
        })
        result = compute_derived_for_dataframe(df_in)
        assert isinstance(result, pd.DataFrame)
        assert "total_puskesmas" in result.columns
        assert "rasio" in result.columns
        assert "proporsi_ri" in result.columns
        assert "log_penduduk" in result.columns

    def test_multiple_rows(self):
        df_in = pd.DataFrame({
            "provinsi": ["A", "B"],
            "kab_kota": ["Kab1", "Kab2"],
            "penduduk": [100_000, 500_000],
            "non_ri": [5, 20],
            "ri": [3, 10],
        })
        result = compute_derived_for_dataframe(df_in)
        assert len(result) == 2
        assert result.loc[0, "total_puskesmas"] == 8
        assert result.loc[1, "total_puskesmas"] == 30


class TestScaler:
    def test_fit_returns_standard_scaler(self, df):
        sc = fit_and_save_scaler(df)
        assert isinstance(sc, StandardScaler)

    def test_scaler_saved_to_disk(self):
        from pathlib import Path
        scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
        assert scaler_path.exists(), "File scaler.pkl tidak ditemukan"

    def test_load_scaler_matches(self, scaler):
        loaded = load_scaler()
        assert loaded is not None, "Scaler tidak berhasil dimuat"
        # Mean harus sama
        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_, decimal=5)

    def test_transform_shape(self, df, scaler):
        X_scaled = transform_reference_data(df, scaler)
        assert X_scaled.shape == (154, 3)

    def test_transform_mean_near_zero(self, df, scaler):
        """Data terstandarisasi harus memiliki mean mendekati 0."""
        X_scaled = transform_reference_data(df, scaler)
        means = X_scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-6), f"Mean tidak mendekati 0: {means}"

    def test_transform_std_near_one(self, df, scaler):
        """Data terstandarisasi harus memiliki std mendekati 1."""
        X_scaled = transform_reference_data(df, scaler)
        stds = X_scaled.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-4), f"Std tidak mendekati 1: {stds}"


class TestTransformSingleInput:
    def test_shape(self, scaler):
        features = {
            "log_penduduk": np.log1p(200_000),
            "rasio": 6.5,
            "proporsi_ri": 40.0,
        }
        result = transform_single_input(features, scaler)
        assert result.shape == (1, 3)

    def test_no_fit_ulang(self, df, scaler):
        """Memastikan scaler tidak berubah setelah transform single input."""
        mean_before = scaler.mean_.copy()
        features = {"log_penduduk": 12.0, "rasio": 5.0, "proporsi_ri": 40.0}
        transform_single_input(features, scaler)
        np.testing.assert_array_equal(scaler.mean_, mean_before)


class TestTransformUploadDataframe:
    def test_correct_shape(self, scaler):
        df_up = pd.DataFrame({
            "log_penduduk": [12.0, 11.5],
            "rasio": [5.0, 8.0],
            "proporsi_ri": [40.0, 60.0],
        })
        result = transform_upload_dataframe(df_up, scaler)
        assert result.shape == (2, 3)

    def test_no_nan_in_result(self, scaler):
        df_up = pd.DataFrame({
            "log_penduduk": [12.0],
            "rasio": [5.0],
            "proporsi_ri": [40.0],
        })
        result = transform_upload_dataframe(df_up, scaler)
        assert not np.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
