"""
tests/test_models.py — Pengujian modul models.py

Menguji:
1. K-Means dilatih pada dataset referensi.
2. Prediksi cluster menggunakan model terlatih (bukan IF/ELSE).
3. Linear Regression — metrik evaluasi.
4. Elbow Method dan Silhouette Score.
5. Cluster profile dari nilai aktual.
6. Persistensi model (save/load).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from src.data_loader import load_main_data, COL
from src.preprocessing import (
    fit_and_save_scaler, transform_reference_data, transform_single_input,
    compute_derived_features,
)
from src.models import (
    train_kmeans, save_kmeans, load_kmeans, predict_cluster,
    elbow_method_data, silhouette_scores_data,
    get_cluster_labels, get_cluster_profile, get_cluster_interpretation,
    train_linear_regression, predict_puskesmas, load_lr_model,
)


@pytest.fixture(scope="module")
def df():
    return load_main_data()


@pytest.fixture(scope="module")
def scaler(df):
    return fit_and_save_scaler(df)


@pytest.fixture(scope="module")
def X_scaled(df, scaler):
    return transform_reference_data(df, scaler)


@pytest.fixture(scope="module")
def kmeans_model(X_scaled):
    return train_kmeans(X_scaled, n_clusters=3)


@pytest.fixture(scope="module")
def lr_result(df):
    return train_linear_regression(df)


class TestKMeans:
    def test_returns_kmeans_object(self, kmeans_model):
        assert isinstance(kmeans_model, KMeans)

    def test_n_clusters(self, kmeans_model):
        assert kmeans_model.n_clusters == 3

    def test_cluster_labels_count(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        assert "Cluster" in df_cl.columns
        assert len(df_cl) == 154

    def test_cluster_ids_valid(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        unique_clusters = sorted(df_cl["Cluster"].unique())
        assert len(unique_clusters) == 3
        assert all(c in [0, 1, 2] for c in unique_clusters)

    def test_all_rows_assigned_cluster(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        assert df_cl["Cluster"].isnull().sum() == 0

    def test_save_and_load(self, kmeans_model):
        save_kmeans(kmeans_model)
        loaded = load_kmeans()
        assert loaded is not None
        assert loaded.n_clusters == kmeans_model.n_clusters
        np.testing.assert_array_almost_equal(
            kmeans_model.cluster_centers_, loaded.cluster_centers_, decimal=5
        )

    def test_predict_cluster_uses_model(self, scaler, kmeans_model):
        """Prediksi cluster HARUS menggunakan model, bukan logika IF/ELSE."""
        features = {"log_penduduk": 12.0, "rasio": 6.0, "proporsi_ri": 40.0}
        X_new = transform_single_input(features, scaler)
        cluster = predict_cluster(X_new, kmeans_model)
        # Verifikasi: cluster yang sama dari predict() langsung
        expected = int(kmeans_model.predict(X_new)[0])
        assert cluster == expected, "predict_cluster harus menggunakan model.predict()"

    def test_predict_cluster_returns_valid_id(self, scaler, kmeans_model):
        features = {"log_penduduk": 12.0, "rasio": 6.0, "proporsi_ri": 40.0}
        X_new = transform_single_input(features, scaler)
        cluster = predict_cluster(X_new, kmeans_model)
        assert cluster in [0, 1, 2]

    def test_different_inputs_can_yield_different_clusters(self, scaler, kmeans_model):
        """Input yang sangat berbeda harus masuk cluster berbeda."""
        # Wilayah kecil, rasio tinggi
        f1 = {"log_penduduk": 10.5, "rasio": 18.0, "proporsi_ri": 80.0}
        # Wilayah besar, rasio rendah
        f2 = {"log_penduduk": 14.5, "rasio": 1.5, "proporsi_ri": 5.0}
        c1 = predict_cluster(transform_single_input(f1, scaler), kmeans_model)
        c2 = predict_cluster(transform_single_input(f2, scaler), kmeans_model)
        assert c1 != c2, "Input sangat berbeda seharusnya masuk cluster berbeda"


class TestElbowAndSilhouette:
    def test_elbow_returns_correct_structure(self, X_scaled):
        data = elbow_method_data(X_scaled, k_range=range(2, 6))
        assert "k_values" in data
        assert "inertia" in data
        assert len(data["k_values"]) == 4
        assert len(data["inertia"]) == 4

    def test_inertia_decreasing(self, X_scaled):
        """Inertia harus menurun seiring naiknya K."""
        data = elbow_method_data(X_scaled, k_range=range(2, 7))
        inertia = data["inertia"]
        for i in range(1, len(inertia)):
            assert inertia[i] <= inertia[i-1], f"Inertia tidak monoton menurun di K={i+2}"

    def test_silhouette_returns_correct_structure(self, X_scaled):
        data = silhouette_scores_data(X_scaled, k_range=range(2, 5))
        assert "k_values" in data
        assert "silhouette" in data

    def test_silhouette_scores_in_range(self, X_scaled):
        data = silhouette_scores_data(X_scaled, k_range=range(2, 5))
        for score in data["silhouette"]:
            assert -1 <= score <= 1, f"Silhouette score di luar range: {score}"


class TestClusterProfile:
    def test_returns_dataframe(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        profile = get_cluster_profile(df_cl)
        assert isinstance(profile, pd.DataFrame)

    def test_profile_has_all_clusters(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        profile = get_cluster_profile(df_cl)
        cluster_ids = set(profile["Cluster"].unique())
        assert cluster_ids == {0, 1, 2}

    def test_total_wilayah_matches(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        profile = get_cluster_profile(df_cl)
        assert profile["n_wilayah"].sum() == 154

    def test_label_from_actual_values(self, df, kmeans_model, X_scaled):
        """Label cluster harus ditentukan dari rasio aktual, bukan nomor cluster."""
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        profile = get_cluster_profile(df_cl)
        # Profile harus diurutkan dari rasio tertinggi
        rasio_sorted = profile["avg_rasio"].tolist()
        assert rasio_sorted == sorted(rasio_sorted, reverse=True), \
            "Profile tidak diurutkan dari rasio tertinggi ke terendah"

    def test_get_cluster_interpretation(self, df, kmeans_model, X_scaled):
        df_cl = get_cluster_labels(df, X_scaled, kmeans_model)
        profile = get_cluster_profile(df_cl)
        # Test untuk setiap cluster
        for cid in [0, 1, 2]:
            interp = get_cluster_interpretation(cid, profile)
            assert "label" in interp
            assert "n_wilayah" in interp
            assert "avg_rasio" in interp


class TestLinearRegression:
    def test_returns_dict(self, lr_result):
        assert isinstance(lr_result, dict)

    def test_required_keys(self, lr_result):
        required = ["model", "r2", "mae", "rmse", "intercept", "coef", "n_train", "n_test"]
        for k in required:
            assert k in lr_result, f"Kunci '{k}' tidak ada"

    def test_model_is_linear_regression(self, lr_result):
        assert isinstance(lr_result["model"], LinearRegression)

    def test_split_ratio(self, lr_result):
        total = lr_result["n_train"] + lr_result["n_test"]
        assert total == 154
        # ~80% training
        assert lr_result["n_train"] == 123
        assert lr_result["n_test"] == 31

    def test_r2_is_finite(self, lr_result):
        assert np.isfinite(lr_result["r2"])

    def test_mae_rmse_positive(self, lr_result):
        assert lr_result["mae"] >= 0
        assert lr_result["rmse"] >= 0

    def test_rmse_geq_mae(self, lr_result):
        assert lr_result["rmse"] >= lr_result["mae"]

    def test_coef_positive(self, lr_result):
        """Penduduk lebih banyak → lebih banyak Puskesmas (korelasi positif)."""
        assert lr_result["coef"] > 0, "Koefisien regresi harus positif"

    def test_r2_reasonable(self, lr_result):
        """R² harus > 0 (model lebih baik dari rata-rata)."""
        assert lr_result["r2"] > 0, f"R² = {lr_result['r2']:.4f} — model tidak berguna"

    def test_model_saved_to_disk(self):
        from pathlib import Path
        lr_path = Path(__file__).parent.parent / "models" / "lr_model.pkl"
        assert lr_path.exists(), "File lr_model.pkl tidak tersimpan"

    def test_predict_puskesmas_positive(self, lr_result):
        model = lr_result["model"]
        pred = predict_puskesmas(500_000, model)
        assert pred > 0, "Prediksi harus positif untuk populasi 500k"

    def test_predict_larger_population_more_puskesmas(self, lr_result):
        model = lr_result["model"]
        pred_small = predict_puskesmas(100_000, model)
        pred_large = predict_puskesmas(1_000_000, model)
        assert pred_large > pred_small, "Populasi lebih besar → lebih banyak Puskesmas"

    def test_y_pred_matches_stored(self, lr_result):
        """Prediksi ulang pada X_test harus sama dengan y_pred tersimpan."""
        y_recomputed = lr_result["model"].predict(lr_result["X_test"])
        np.testing.assert_array_almost_equal(y_recomputed, lr_result["y_pred"], decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
