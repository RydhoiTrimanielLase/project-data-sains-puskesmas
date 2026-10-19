"""
app.py — Dashboard Analisis Kesenjangan Ketersediaan Fasilitas Kesehatan di Sumatra 2022.

Aplikasi Streamlit dengan 7 halaman:
A. Dashboard
B. Eksplorasi Data
C. Clustering K-Means
D. Input Wilayah Baru
E. Upload Data Excel/CSV
F. Linear Regression
G. Model & Validasi
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── KONFIGURASI HALAMAN ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Puskesmas Sumatra 2022",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS KUSTOM ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #3B82F6;
    --secondary: #8B5CF6;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --bg-dark: #0F172A;
    --bg-card: #1E293B;
    --bg-card2: #253047;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --border: rgba(148,163,184,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* CARDS KPI */
.kpi-card {
    background: linear-gradient(135deg, #1E293B, #253047);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.2);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 12px 12px 0 0;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-sublabel {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 2px;
}
.kpi-icon {
    font-size: 1.5rem;
    margin-bottom: 8px;
    display: block;
}

/* SECTION HEADER */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-subtitle {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 24px;
    line-height: 1.5;
}

/* BADGES */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-blue  { background: rgba(59,130,246,0.15);  color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.badge-red   { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-yellow{ background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
.badge-purple{ background: rgba(139,92,246,0.15); color: #A78BFA; border: 1px solid rgba(139,92,246,0.3); }

/* INFO BOX */
.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 4px solid var(--primary);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #CBD5E1;
    font-size: 0.875rem;
    line-height: 1.6;
}
.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 4px solid var(--warning);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #FDE68A;
    font-size: 0.875rem;
    line-height: 1.6;
}
.success-box {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 4px solid var(--success);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #6EE7B7;
    font-size: 0.875rem;
    line-height: 1.6;
}
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 4px solid var(--danger);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #FCA5A5;
    font-size: 0.875rem;
    line-height: 1.6;
}

/* CLUSTER CARD */
.cluster-card {
    background: linear-gradient(135deg, #1E293B, #253047);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 18px 22px;
    margin-bottom: 12px;
}

/* TABLE */
.stDataFrame {
    border-radius: 10px !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,41,59,0.6);
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* METRIC DELTA */
[data-testid="stMetricDelta"] { color: var(--success) !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── IMPORT MODUL INTERNAL ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_main_data, get_dataset_metadata, get_province_summary,
    validate_upload_dataframe, COL,
)
from src.preprocessing import (
    fit_and_save_scaler, load_scaler,
    transform_reference_data, transform_single_input,
    compute_derived_features, compute_derived_for_dataframe,
    transform_upload_dataframe, FEATURE_DISPLAY_NAMES,
)
from src.models import (
    train_kmeans, save_kmeans, load_kmeans, predict_cluster,
    elbow_method_data, silhouette_scores_data, get_cluster_labels,
    get_cluster_profile, get_cluster_interpretation,
    train_linear_regression, predict_puskesmas, load_lr_model,
)
import src.visualizations as viz


# ─── LOAD DATA & INISIALISASI MODEL ───────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_main_data()


@st.cache_resource(show_spinner=False)
def initialize_models(n_clusters: int = 3):
    """Inisialisasi semua model. Di-cache agar tidak dilatih ulang."""
    df = cached_load_data()
    # Fit scaler & K-Means
    scaler = fit_and_save_scaler(df)
    X_scaled = transform_reference_data(df, scaler)
    kmeans = train_kmeans(X_scaled, n_clusters=n_clusters)
    save_kmeans(kmeans)
    # Linear Regression
    lr_result = train_linear_regression(df)
    # Elbow & Silhouette (hitung sekali)
    elbow_data = elbow_method_data(X_scaled)
    sil_data = silhouette_scores_data(X_scaled)
    return scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled


# ─── SIDEBAR NAVIGASI ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <div style="font-size:2.5rem;">🏥</div>
        <div style="font-size:1rem; font-weight:700; color:#F1F5F9; margin-top:6px;">
            Dashboard Puskesmas
        </div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:2px;">
            Sumatra 2022 · Data Sains
        </div>
    </div>
    <hr style="border-color:rgba(148,163,184,0.15); margin: 12px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        options=[
            "🏠  Dashboard",
            "📊  Eksplorasi Data",
            "🔵  Clustering K-Means",
            "📝  Input Wilayah Baru",
            "📂  Upload Data",
            "📈  Linear Regression",
            "🔬  Model & Validasi",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border-color:rgba(148,163,184,0.15); margin: 12px 0;">
    <div style="font-size:0.72rem; color:#475569; text-align:center; padding-bottom:12px;">
        154 Kab/Kota · 10 Provinsi<br>
        K-Means + Linear Regression<br>
        <span style="color:#3B82F6;">Teknik Informatika S1</span>
    </div>
    """, unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def fmt_num(n, decimals=0):
    """Format angka dengan pemisah ribuan."""
    if decimals == 0:
        return f"{int(n):,}".replace(",", ".")
    return f"{n:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kpi_card(icon, value, label, sublabel=""):
    return f"""
    <div class="kpi-card">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {"<div class='kpi-sublabel'>" + sublabel + "</div>" if sublabel else ""}
    </div>
    """


def cluster_label_color(label: str) -> str:
    if "Tinggi" in label:
        return "badge-blue"
    elif "Rendah" in label:
        return "badge-red"
    return "badge-yellow"


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
try:
    df = cached_load_data()
    meta = get_dataset_metadata(df)
    province_summary = get_province_summary(df)
except FileNotFoundError as e:
    st.error(f"❌ File dataset tidak ditemukan:\n\n{e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal memuat dataset: {e}")
    st.stop()


# ════════════════════════════════════════════════════════════
# HALAMAN A — DASHBOARD
# ════════════════════════════════════════════════════════════

if page == "🏠  Dashboard":
    st.markdown('<div class="section-title">🏠 Dashboard Utama</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Ringkasan ketersediaan Puskesmas di 154 kabupaten/kota Pulau Sumatra — Tahun 2022</div>',
        unsafe_allow_html=True,
    )

    # Baris KPI 1
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("🗺️", fmt_num(meta["n_wilayah"]), "Total Wilayah", "Kabupaten/Kota"), unsafe_allow_html=True)
    c2.markdown(kpi_card("🏛️", fmt_num(meta["n_provinsi"]), "Provinsi", "Di Pulau Sumatra"), unsafe_allow_html=True)
    c3.markdown(kpi_card("👥", fmt_num(meta["total_penduduk"]), "Total Penduduk", "Sensus BPS 2022"), unsafe_allow_html=True)
    c4.markdown(kpi_card("🏥", fmt_num(meta["total_puskesmas"]), "Total Puskesmas", "Seluruh Sumatra"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Baris KPI 2
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(kpi_card("🏨", fmt_num(meta["total_ri"]), "Puskesmas Rawat Inap", ""), unsafe_allow_html=True)
    c6.markdown(kpi_card("🏪", fmt_num(meta["total_non_ri"]), "Puskesmas Non Rawat Inap", ""), unsafe_allow_html=True)
    pct_ri = meta["total_ri"] / meta["total_puskesmas"] * 100
    c7.markdown(kpi_card("📊", f"{pct_ri:.1f}%", "% Puskesmas Rawat Inap", ""), unsafe_allow_html=True)
    c8.markdown(kpi_card("⚖️", f"{meta['rasio_agregat_nasional']:.2f}", "Rasio per 100k Penduduk", "Tingkat Sumatra"), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Wilayah tertinggi & terendah
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="cluster-card">
            <div style="font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;">
                📈 Rasio Tertinggi
            </div>
            <div style="font-size:1.4rem; font-weight:700; color:#60A5FA;">
                {kab}
            </div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-top:2px;">
                {prov}
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:#34D399; margin-top:8px;">
                {rasio:.2f}
                <span style="font-size:0.78rem; color:#94A3B8; font-weight:400;"> per 100.000 penduduk</span>
            </div>
        </div>
        """.format(
            kab=meta["wilayah_rasio_tertinggi"],
            prov=meta["provinsi_rasio_tertinggi"],
            rasio=meta["rasio_tertinggi"],
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="cluster-card">
            <div style="font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;">
                📉 Rasio Terendah
            </div>
            <div style="font-size:1.4rem; font-weight:700; color:#F87171;">
                {kab}
            </div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-top:2px;">
                {prov}
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:#F87171; margin-top:8px;">
                {rasio:.2f}
                <span style="font-size:0.78rem; color:#94A3B8; font-weight:400;"> per 100.000 penduduk</span>
            </div>
        </div>
        """.format(
            kab=meta["wilayah_rasio_terendah"],
            prov=meta["provinsi_rasio_terendah"],
            rasio=meta["rasio_terendah"],
        ), unsafe_allow_html=True)

    # Peta distribusi provinsi
    st.markdown("---")
    st.markdown("### 📍 Distribusi Wilayah dan Puskesmas per Provinsi")
    st.markdown(
        '<div class="info-box">💡 <b>Catatan metodologi:</b> Rasio agregat provinsi dihitung sebagai '
        '(Total Puskesmas Provinsi ÷ Total Penduduk Provinsi) × 100.000, BUKAN rata-rata sederhana '
        'rasio tiap kabupaten/kota. Metode ini lebih mencerminkan kondisi aktual provinsi.</div>',
        unsafe_allow_html=True,
    )

    prov_disp = province_summary.copy()
    prov_disp.columns = [
        "Provinsi", "Jml Kab/Kota", "Total Penduduk", "Total Puskesmas",
        "Non Rawat Inap", "Rawat Inap", "Rasio Agregat (per 100k)"
    ]
    prov_disp["Total Penduduk"] = prov_disp["Total Penduduk"].apply(lambda x: f"{x:,.0f}")
    prov_disp["Total Puskesmas"] = prov_disp["Total Puskesmas"].apply(lambda x: f"{x:,}")
    prov_disp["Rasio Agregat (per 100k)"] = prov_disp["Rasio Agregat (per 100k)"].apply(lambda x: f"{x:.2f}")
    st.dataframe(prov_disp, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Keterbatasan Dataset:</b> Seluruh data merujuk pada tahun 2022. 
        Aplikasi ini tidak dapat digunakan untuk analisis tren multitahun. 
        Kode Kab/Kota (sumber KMK) berbeda dari Kode BPS — keduanya dipertahankan terpisah.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN B — EKSPLORASI DATA
# ════════════════════════════════════════════════════════════

elif page == "📊  Eksplorasi Data":
    st.markdown('<div class="section-title">📊 Eksplorasi Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Visualisasi interaktif distribusi dan pola ketersediaan Puskesmas di Sumatra 2022.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Rasio Tertinggi", "📉 Rasio Terendah", "🏛️ Per Provinsi",
        "📊 Distribusi", "🔵 Scatter", "🔥 Korelasi",
    ])

    with tab1:
        st.markdown("**10 Wilayah dengan Rasio Puskesmas per 100.000 Penduduk Tertinggi**")
        st.markdown('<div class="info-box">Wilayah-wilayah ini umumnya memiliki populasi kecil tetapi jumlah Puskesmas relatif banyak, menghasilkan rasio tinggi. Ini tidak selalu berarti lebih baik — perlu konteks luas wilayah dan aksesibilitas.</div>', unsafe_allow_html=True)
        fig = viz.chart_top10_rasio_tertinggi(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**10 Wilayah dengan Rasio Puskesmas per 100.000 Penduduk Terendah**")
        st.markdown('<div class="info-box">Wilayah-wilayah ini umumnya merupakan kota besar atau kabupaten padat penduduk. Rasio rendah dapat mengindikasikan potensi kesenjangan layanan, meskipun kualitas dan aksesibilitas juga perlu dipertimbangkan.</div>', unsafe_allow_html=True)
        fig = viz.chart_top10_rasio_terendah(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Perbandingan Rasio Agregat Antarprovinsi**")
        st.markdown('<div class="info-box">Rasio agregat provinsi = Total Puskesmas Provinsi ÷ Total Penduduk Provinsi × 100.000. Berbeda dari rata-rata sederhana rasio tiap kab/kota.</div>', unsafe_allow_html=True)
        fig = viz.chart_rasio_provinsi(province_summary)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("**Distribusi Rasio Puskesmas per 100.000 Penduduk**")
        st.markdown('<div class="info-box">Histogram menunjukkan sebaran nilai rasio di 154 wilayah. Garis putus-putus adalah median nasional.</div>', unsafe_allow_html=True)
        fig = viz.chart_distribusi_rasio(df)
        st.plotly_chart(fig, use_container_width=True)

        col_stats = st.columns(5)
        rasio_col = COL["rasio"]
        stats = [
            ("Min", df[rasio_col].min(), ""),
            ("Q1 (25%)", df[rasio_col].quantile(0.25), ""),
            ("Median", df[rasio_col].median(), ""),
            ("Q3 (75%)", df[rasio_col].quantile(0.75), ""),
            ("Max", df[rasio_col].max(), ""),
        ]
        for c, (label, val, _) in zip(col_stats, stats):
            c.metric(label, f"{val:.2f}")

    with tab5:
        st.markdown("**Hubungan Jumlah Penduduk dan Jumlah Puskesmas**")
        st.markdown('<div class="info-box">Scatter plot dengan garis tren linear. Warna berbeda per provinsi. Hubungan ini bersifat deskriptif — bukan bukti sebab-akibat.</div>', unsafe_allow_html=True)
        fig = viz.chart_scatter_penduduk_puskesmas(df)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.markdown("**Heatmap Korelasi Variabel Numerik**")
        st.markdown('<div class="info-box">Nilai mendekati +1 (biru gelap) menunjukkan korelasi positif kuat, mendekati -1 (merah gelap) menunjukkan korelasi negatif kuat.</div>', unsafe_allow_html=True)
        fig = viz.chart_heatmap_korelasi(df)
        st.plotly_chart(fig, use_container_width=True)

    # Tabel Data Lengkap
    st.markdown("---")
    st.markdown("### 📋 Tabel Data Lengkap")

    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        prov_options = ["Semua Provinsi"] + sorted(df[COL["provinsi"]].unique().tolist())
        selected_prov = st.selectbox("Filter Provinsi", prov_options, key="b_prov_filter")
    with col_search:
        search_kab = st.text_input("🔍 Cari Kabupaten/Kota", placeholder="Ketik nama wilayah...", key="b_search")

    df_view = df.copy()
    if selected_prov != "Semua Provinsi":
        df_view = df_view[df_view[COL["provinsi"]] == selected_prov]
    if search_kab:
        df_view = df_view[df_view[COL["kab_kota"]].str.contains(search_kab, case=False, na=False)]

    display_cols = [
        COL["provinsi"], COL["kab_kota"], COL["total_puskesmas"],
        COL["non_ri"], COL["ri"], COL["penduduk"],
        COL["rasio"], COL["proporsi_ri"],
    ]
    df_show = df_view[display_cols].copy()
    df_show[COL["rasio"]] = df_show[COL["rasio"]].round(3)
    df_show[COL["proporsi_ri"]] = df_show[COL["proporsi_ri"]].round(2)
    df_show[COL["penduduk"]] = df_show[COL["penduduk"]].apply(lambda x: f"{int(x):,}")

    st.markdown(f"*Menampilkan {len(df_show)} dari {len(df)} wilayah*")
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Ekspor CSV
    csv_data = df_view[display_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Unduh Data Terfilter (CSV)",
        data=csv_data,
        file_name=f"puskesmas_sumatra_{selected_prov.replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════
# HALAMAN C — CLUSTERING K-MEANS
# ════════════════════════════════════════════════════════════

elif page == "🔵  Clustering K-Means":
    st.markdown('<div class="section-title">🔵 Clustering Wilayah dengan K-Means</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Pengelompokan 154 wilayah berdasarkan tiga fitur: '
        'Log Penduduk, Rasio Puskesmas, dan Proporsi Rawat Inap. '
        'Model dilatih pada dataset referensi Sumatra 2022.</div>',
        unsafe_allow_html=True,
    )

    # Pilih jumlah cluster
    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=3, step=1, key="c_k")

    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>Fitur K-Means yang digunakan:</b><br>
        1️⃣ <code>ln(1 + Jumlah Penduduk)</code> — mereduksi skewness distribusi penduduk<br>
        2️⃣ <code>Rasio Puskesmas per 100.000 Penduduk</code> — indikator ketersediaan layanan<br>
        3️⃣ <code>Proporsi Puskesmas Rawat Inap (%)</code> — indikator kapasitas layanan tingkat lanjut<br>
        <br>
        Preprocessing: <b>StandardScaler</b> (fit pada 154 data referensi, digunakan konsisten untuk semua input baru).
        </div>
        """, unsafe_allow_html=True)

    # Latih atau muat model
    with st.spinner("⏳ Melatih model K-Means..."):
        scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled_ref = initialize_models(n_clusters)
        # Jika user mengubah K, latih ulang dengan K baru
        if kmeans.n_clusters != n_clusters:
            kmeans = train_kmeans(X_scaled_ref, n_clusters=n_clusters)
            save_kmeans(kmeans)

        df_clustered = get_cluster_labels(df, X_scaled_ref, kmeans)
        profile_df = get_cluster_profile(df_clustered)

    # Elbow & Silhouette
    tab_el, tab_sil, tab_scatter, tab_3d, tab_tabel = st.tabs([
        "📐 Elbow Method", "📏 Silhouette Score", "🔵 Grafik Cluster",
        "🌐 3D View", "📋 Tabel Cluster",
    ])

    with tab_el:
        st.markdown("**Elbow Method — Menentukan K Optimal**")
        fig = viz.chart_elbow(elbow_data)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        Pilih K pada <i>titik siku</i> (elbow) — dimana penurunan inertia mulai landai.
        Inertia (WCSS) mengukur total jarak kuadrat setiap titik ke centroid clusternya.
        </div>
        """, unsafe_allow_html=True)

    with tab_sil:
        st.markdown("**Silhouette Score — Kualitas Pemisahan Cluster**")
        fig = viz.chart_silhouette(sil_data)
        st.plotly_chart(fig, use_container_width=True)
        current_sil = sil_data["silhouette"][sil_data["k_values"].index(n_clusters)]
        st.markdown(f"""
        <div class="success-box">
        ✅ Silhouette Score untuk K={n_clusters}: <b>{current_sil:.4f}</b><br>
        Semakin mendekati 1.0 = cluster lebih jelas terpisah. Nilai > 0.5 umumnya dianggap baik.
        </div>
        """, unsafe_allow_html=True)

    with tab_scatter:
        st.markdown("**Visualisasi Cluster (2D)**")
        x_axis_opt = st.selectbox("Sumbu X", [
            "Log Jumlah Penduduk",
            "Rasio Puskesmas per 100.000 Penduduk",
            "Proporsi Puskesmas Rawat Inap (%)",
        ], key="c_xaxis")
        y_axis_opt = st.selectbox("Sumbu Y", [
            "Rasio Puskesmas per 100.000 Penduduk",
            "Log Jumlah Penduduk",
            "Proporsi Puskesmas Rawat Inap (%)",
        ], key="c_yaxis")
        fig = viz.chart_cluster_scatter(df_clustered, x_col=x_axis_opt, y_col=y_axis_opt)
        st.plotly_chart(fig, use_container_width=True)

    with tab_3d:
        st.markdown("**Visualisasi 3D Cluster (3 Fitur Sekaligus)**")
        fig = viz.chart_cluster_3d(df_clustered)
        st.plotly_chart(fig, use_container_width=True)

    with tab_tabel:
        # Tabel hasil cluster
        df_cluster_view = df_clustered[[
            COL["provinsi"], COL["kab_kota"], "Cluster",
            COL["penduduk"], COL["total_puskesmas"],
            COL["rasio"], COL["proporsi_ri"],
        ]].copy()
        df_cluster_view[COL["rasio"]] = df_cluster_view[COL["rasio"]].round(3)
        df_cluster_view[COL["proporsi_ri"]] = df_cluster_view[COL["proporsi_ri"]].round(2)
        df_cluster_view = df_cluster_view.rename(columns={
            COL["provinsi"]: "Provinsi",
            COL["kab_kota"]: "Kabupaten/Kota",
            COL["penduduk"]: "Penduduk",
            COL["total_puskesmas"]: "Jml Puskesmas",
            COL["rasio"]: "Rasio per 100k",
            COL["proporsi_ri"]: "% RI",
        })
        st.dataframe(df_cluster_view.sort_values("Cluster"), use_container_width=True, hide_index=True)

        csv_cluster = df_cluster_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Unduh Hasil Cluster (CSV)", csv_cluster, "hasil_cluster.csv", "text/csv")

    # Profil Cluster
    st.markdown("---")
    st.markdown("### 🏷️ Karakteristik Tiap Cluster")
    st.markdown("""
    <div class="warning-box">
    ⚠️ Label cluster ditetapkan berdasarkan nilai aktual rasio (rata-rata per cluster), 
    BUKAN dari nomor cluster. Cluster 0 tidak selalu berarti terbaik.
    </div>
    """, unsafe_allow_html=True)

    cluster_cols = st.columns(len(profile_df))
    for i, (_, row) in enumerate(profile_df.iterrows()):
        with cluster_cols[i]:
            badge_cls = cluster_label_color(row["Label"])
            st.markdown(f"""
            <div class="cluster-card">
                <span class="badge {badge_cls}" style="margin-bottom:10px;">{row["Label"]}</span>
                <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin:8px 0;">
                    Cluster {int(row["Cluster"])}
                </div>
                <div style="font-size:0.8rem; color:#94A3B8;">
                    <b>{int(row["n_wilayah"])}</b> wilayah<br><br>
                    Rata-rata Rasio:<br>
                    <span style="color:#60A5FA; font-weight:600; font-size:1.1rem;">
                        {row["avg_rasio"]:.2f}
                    </span> per 100k<br><br>
                    Rata-rata Penduduk:<br>
                    <span style="color:#A78BFA;">{row["avg_penduduk"]:,.0f}</span><br><br>
                    Rata-rata Puskesmas:<br>
                    <span style="color:#34D399;">{row["avg_puskesmas"]:.1f}</span><br><br>
                    Rata-rata % RI:<br>
                    <span style="color:#FCD34D;">{row["avg_proporsi_ri"]:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN D — INPUT WILAYAH BARU
# ════════════════════════════════════════════════════════════

elif page == "📝  Input Wilayah Baru":
    st.markdown('<div class="section-title">📝 Prediksi Cluster untuk Wilayah Baru</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Masukkan data wilayah baru untuk menentukan cluster terdekat '
        'berdasarkan model K-Means yang telah dilatih pada 154 wilayah Sumatra 2022.</div>',
        unsafe_allow_html=True,
    )

    scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled_ref = initialize_models(3)
    df_clustered = get_cluster_labels(df, X_scaled_ref, kmeans)
    profile_df = get_cluster_profile(df_clustered)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Penting:</b> Hasil prediksi menunjukkan <b>kemiripan pola</b> dengan cluster data Sumatra 2022, 
    bukan penilaian definitif terhadap kualitas pelayanan kesehatan. 
    Hasil untuk wilayah di luar Sumatra atau tahun berbeda harus ditafsirkan dengan hati-hati.
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_input_wilayah"):
        st.markdown("#### 📋 Informasi Wilayah")
        col1, col2 = st.columns(2)
        with col1:
            nama_wilayah = st.text_input("Nama Wilayah / Kabupaten/Kota *", placeholder="mis. Kab. Contoh Baru")
            nama_provinsi = st.text_input("Nama Provinsi / Keterangan", placeholder="mis. Sumatra Utara")
        with col2:
            penduduk_input = st.number_input(
                "Jumlah Penduduk *", min_value=1, max_value=10_000_000,
                value=200_000, step=1_000, help="Harus lebih besar dari 0"
            )
            non_ri_input = st.number_input(
                "Jumlah Puskesmas Non Rawat Inap *", min_value=0, max_value=200,
                value=8, step=1, help="Bilangan bulat ≥ 0"
            )
            ri_input = st.number_input(
                "Jumlah Puskesmas Rawat Inap *", min_value=0, max_value=100,
                value=5, step=1, help="Bilangan bulat ≥ 0"
            )

        submit = st.form_submit_button("🔍 Proses & Prediksi Cluster", use_container_width=True)

    if submit:
        # Validasi
        errors_d = []
        if not nama_wilayah.strip():
            errors_d.append("Nama wilayah tidak boleh kosong.")
        if penduduk_input <= 0:
            errors_d.append("Jumlah penduduk harus lebih besar dari 0.")
        if non_ri_input < 0 or ri_input < 0:
            errors_d.append("Jumlah Puskesmas tidak boleh negatif.")
        if non_ri_input + ri_input == 0:
            errors_d.append("Total Puskesmas harus minimal 1.")

        if errors_d:
            for err in errors_d:
                st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
        else:
            # Hitung variabel turunan
            derived = compute_derived_features(
                penduduk=float(penduduk_input),
                non_ri=int(non_ri_input),
                ri=int(ri_input),
            )

            # Transformasi menggunakan scaler referensi (TIDAK fit ulang)
            X_new = transform_single_input(derived, scaler)

            # Prediksi cluster via model terlatih (BUKAN IF/ELSE)
            cluster_id = predict_cluster(X_new, kmeans)
            interp = get_cluster_interpretation(cluster_id, profile_df)

            # ── Tampilkan Hasil ────────────────────────────────
            st.markdown("---")
            st.markdown(f"## 🎯 Hasil Prediksi: {nama_wilayah}")

            # Summary input
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("#### 📥 Ringkasan Input")
                st.markdown(f"""
                <div class="cluster-card">
                    <table style="width:100%; font-size:0.85rem; color:#CBD5E1; border-spacing:0 8px;">
                        <tr><td style="color:#64748B;">Wilayah</td><td style="color:#F1F5F9; font-weight:600;">{nama_wilayah}</td></tr>
                        <tr><td style="color:#64748B;">Provinsi</td><td>{nama_provinsi or '—'}</td></tr>
                        <tr><td style="color:#64748B;">Penduduk</td><td>{int(penduduk_input):,}</td></tr>
                        <tr><td style="color:#64748B;">Non Rawat Inap</td><td>{non_ri_input}</td></tr>
                        <tr><td style="color:#64748B;">Rawat Inap</td><td>{ri_input}</td></tr>
                        <tr><td style="color:#64748B;">Total Puskesmas</td><td style="color:#34D399; font-weight:600;">{derived['total_puskesmas']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_out:
                st.markdown("#### 📊 Nilai Fitur K-Means")
                st.markdown(f"""
                <div class="cluster-card">
                    <table style="width:100%; font-size:0.85rem; color:#CBD5E1; border-spacing:0 8px;">
                        <tr>
                            <td style="color:#64748B;">ln(1 + Penduduk)</td>
                            <td style="color:#60A5FA; font-weight:600;">{derived['log_penduduk']:.4f}</td>
                        </tr>
                        <tr>
                            <td style="color:#64748B;">Rasio per 100k</td>
                            <td style="color:#34D399; font-weight:600;">{derived['rasio']:.4f}</td>
                        </tr>
                        <tr>
                            <td style="color:#64748B;">% Rawat Inap</td>
                            <td style="color:#FCD34D; font-weight:600;">{derived['proporsi_ri']:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="color:#64748B;">Fitur (Scaled)</td>
                            <td style="color:#A78BFA; font-size:0.75rem;">[{X_new[0,0]:.3f}, {X_new[0,1]:.3f}, {X_new[0,2]:.3f}]</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            # Hasil cluster
            badge_cls = cluster_label_color(interp["label"])
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E293B, #253047);
                        border-radius: 16px; border: 1px solid rgba(59,130,246,0.3);
                        padding: 28px; margin: 16px 0; text-align:center;">
                <div style="font-size:0.8rem; color:#64748B; text-transform:uppercase; letter-spacing:.1em;">
                    Hasil Prediksi Cluster (K-Means)
                </div>
                <div style="font-size:3rem; font-weight:800; color:#F1F5F9; margin:12px 0;">
                    Cluster {cluster_id}
                </div>
                <span class="badge {badge_cls}" style="font-size:0.9rem; padding:6px 16px;">
                    {interp["label"]}
                </span>
                <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:16px;">
                    <div style="color:#94A3B8; font-size:0.78rem;">
                        Jumlah Wilayah di Cluster<br>
                        <span style="color:#F1F5F9; font-size:1.2rem; font-weight:700;">{interp["n_wilayah"]}</span>
                    </div>
                    <div style="color:#94A3B8; font-size:0.78rem;">
                        Rata-rata Rasio<br>
                        <span style="color:#60A5FA; font-size:1.2rem; font-weight:700;">{interp["avg_rasio"]:.2f}</span>
                    </div>
                    <div style="color:#94A3B8; font-size:0.78rem;">
                        Rata-rata Penduduk<br>
                        <span style="color:#A78BFA; font-size:1.2rem; font-weight:700;">{interp["avg_penduduk"]:,.0f}</span>
                    </div>
                    <div style="color:#94A3B8; font-size:0.78rem;">
                        Rata-rata % RI<br>
                        <span style="color:#34D399; font-size:1.2rem; font-weight:700;">{interp["avg_proporsi_ri"]:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Perbandingan dengan data referensi
            st.markdown("#### 📊 Perbandingan dengan Karakteristik Data Referensi")
            fig_compare = go.Figure()
            metrics_compare = [
                ("Rasio per 100k", derived["rasio"], interp["avg_rasio"]),
                ("% Rawat Inap", derived["proporsi_ri"], interp["avg_proporsi_ri"]),
                ("Log Penduduk", derived["log_penduduk"], float(profile_df[profile_df["Cluster"]==cluster_id]["avg_log_penduduk"])),
            ]
            labels = [m[0] for m in metrics_compare]
            vals_input = [m[1] for m in metrics_compare]
            vals_cluster = [m[2] for m in metrics_compare]
            fig_compare.add_trace(go.Bar(
                name=f"Input: {nama_wilayah}",
                x=labels, y=vals_input,
                marker_color="rgba(59,130,246,0.8)",
            ))
            fig_compare.add_trace(go.Bar(
                name=f"Rata-rata Cluster {cluster_id}",
                x=labels, y=vals_cluster,
                marker_color="rgba(139,92,246,0.6)",
            ))
            fig_compare.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                legend=dict(bgcolor="rgba(30,41,59,0.8)"),
                height=340,
                margin=dict(l=50, r=30, t=40, b=50),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Scatter dengan highlight input baru
            st.markdown("#### 🗺️ Posisi Wilayah Baru dalam Peta Cluster")
            fig_scatter = viz.chart_cluster_scatter(
                df_clustered,
                highlight_point={
                    "x": derived["log_penduduk"],
                    "y": derived["rasio"],
                    "name": nama_wilayah,
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            💡 <b>Interpretasi:</b> Prediksi cluster diperoleh dari model K-Means yang dilatih pada 154 wilayah Sumatra 2022.
            Hasil menunjukkan kemiripan pola statistik dengan wilayah-wilayah di cluster tersebut, 
            bukan penilaian definitif atas kualitas atau kecukupan layanan kesehatan.
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN E — UPLOAD DATA
# ════════════════════════════════════════════════════════════

elif page == "📂  Upload Data":
    st.markdown('<div class="section-title">📂 Upload Data Excel / CSV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Kelompokkan banyak wilayah sekaligus menggunakan model K-Means yang sudah dilatih. '
        'Model referensi tetap menggunakan dataset Sumatra 2022 — tidak dilatih ulang dari file upload.</div>',
        unsafe_allow_html=True,
    )

    scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled_ref = initialize_models(3)
    df_clustered_ref = get_cluster_labels(df, X_scaled_ref, kmeans)
    profile_df = get_cluster_profile(df_clustered_ref)

    # Template download
    template_df = pd.DataFrame({
        "provinsi": ["Contoh Provinsi"],
        "kab_kota": ["Contoh Kabupaten"],
        "penduduk": [200000],
        "non_ri": [8],
        "ri": [5],
    })
    template_csv = template_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "📄 Unduh Template CSV",
        data=template_csv,
        file_name="template_upload_puskesmas.csv",
        mime="text/csv",
    )

    st.markdown("""
    <div class="info-box">
    <b>Kolom wajib dalam file upload:</b><br>
    • <code>provinsi</code> — Nama provinsi<br>
    • <code>kab_kota</code> — Nama kabupaten/kota<br>
    • <code>penduduk</code> — Jumlah penduduk (integer > 0)<br>
    • <code>non_ri</code> — Jumlah Puskesmas Non Rawat Inap (integer ≥ 0)<br>
    • <code>ri</code> — Jumlah Puskesmas Rawat Inap (integer ≥ 0)<br>
    <br>
    Jika nama kolom berbeda, gunakan fitur pemetaan kolom di bawah.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel (.xlsx)",
        type=["csv", "xlsx"],
        key="e_upload",
    )

    if uploaded_file:
        # Baca file
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload_raw = pd.read_csv(uploaded_file)
            else:
                df_upload_raw = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Gagal membaca file: {e}</div>', unsafe_allow_html=True)
            st.stop()

        st.markdown("#### 👀 Preview File (5 baris pertama)")
        st.dataframe(df_upload_raw.head(), use_container_width=True, hide_index=True)
        st.markdown(f"*File memiliki {len(df_upload_raw)} baris dan {len(df_upload_raw.columns)} kolom.*")

        # Pemetaan kolom
        required_cols = ["provinsi", "kab_kota", "penduduk", "non_ri", "ri"]
        file_cols = list(df_upload_raw.columns)
        auto_match = all(c in file_cols for c in required_cols)

        if not auto_match:
            st.markdown("#### 🔗 Pemetaan Kolom")
            st.markdown('<div class="warning-box">⚠️ Nama kolom tidak sesuai. Pilih kolom yang sesuai dari file Anda:</div>', unsafe_allow_html=True)
            col_map = {}
            for req_col in required_cols:
                default_idx = 0
                for i, fc in enumerate(file_cols):
                    if req_col.lower() in fc.lower():
                        default_idx = i
                        break
                col_map[req_col] = st.selectbox(
                    f"Kolom untuk '{req_col}'", options=file_cols, index=default_idx, key=f"map_{req_col}"
                )
            df_mapped = df_upload_raw[list(col_map.values())].copy()
            df_mapped.columns = required_cols
        else:
            df_mapped = df_upload_raw[required_cols].copy()

        # Validasi
        df_valid, val_errors, val_warnings = validate_upload_dataframe(df_mapped)

        if val_errors:
            st.markdown("#### ❌ Kesalahan Data")
            st.markdown('<div class="error-box">' +
                        "<br>".join([f"• {e}" for e in val_errors]) +
                        "</div>", unsafe_allow_html=True)
        if val_warnings:
            for w in val_warnings:
                st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

        if len(df_valid) == 0:
            st.error("Tidak ada baris yang lolos validasi. Silakan periksa file Anda.")
        else:
            if val_errors:
                n_invalid = len(df_mapped) - len(df_valid)
                st.markdown(f'<div class="warning-box">⚠️ {n_invalid} baris dihapus karena tidak valid. {len(df_valid)} baris akan diproses.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ Semua {len(df_valid)} baris lolos validasi.</div>', unsafe_allow_html=True)

            # Hitung variabel turunan
            df_proc = compute_derived_for_dataframe(df_valid)

            # Transformasi & prediksi cluster
            X_upload = transform_upload_dataframe(df_proc, scaler)
            df_proc["Cluster"] = kmeans.predict(X_upload)

            # Tampilkan hasil
            st.markdown("---")
            st.markdown("#### 🔵 Hasil Clustering")

            st.markdown("""
            <div class="warning-box">
            ⚠️ Model tidak dilatih ulang dari file upload. Prediksi menggunakan model K-Means yang 
            dilatih pada dataset Sumatra 2022. Hasil untuk wilayah di luar Sumatra atau periode berbeda 
            perlu ditafsirkan dengan hati-hati.
            </div>
            """, unsafe_allow_html=True)

            result_df = df_proc[[
                "provinsi", "kab_kota", "penduduk", "non_ri", "ri",
                "total_puskesmas", "rasio", "proporsi_ri", "log_penduduk", "Cluster",
            ]].copy()
            result_df.columns = [
                "Provinsi", "Kab/Kota", "Penduduk", "Non RI", "RI",
                "Total Pkm", "Rasio per 100k", "% RI", "Log Penduduk", "Cluster",
            ]
            result_df["Rasio per 100k"] = result_df["Rasio per 100k"].round(3)
            result_df["% RI"] = result_df["% RI"].round(2)
            result_df["Log Penduduk"] = result_df["Log Penduduk"].round(4)

            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # Unduh hasil
            csv_result = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ Unduh Hasil Cluster (CSV)",
                data=csv_result,
                file_name="hasil_cluster_upload.csv",
                mime="text/csv",
            )

            # Distribusi cluster
            st.markdown("#### 📊 Distribusi Cluster dari Data Upload")
            cluster_dist = df_proc["Cluster"].value_counts().reset_index()
            cluster_dist.columns = ["Cluster", "Jumlah"]
            fig_dist = go.Figure(go.Pie(
                labels=[f"Cluster {c}" for c in cluster_dist["Cluster"]],
                values=cluster_dist["Jumlah"],
                hole=0.45,
                textinfo="label+percent+value",
                textfont=dict(color="white"),
                marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
            ))
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                height=340,
                legend=dict(font=dict(color="#CBD5E1"), bgcolor="rgba(30,41,59,0.8)"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ════════════════════════════════════════════════════════════
# HALAMAN F — LINEAR REGRESSION
# ════════════════════════════════════════════════════════════

elif page == "📈  Linear Regression":
    st.markdown('<div class="section-title">📈 Regresi Linear — Estimasi Jumlah Puskesmas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Estimasi jumlah Puskesmas berdasarkan jumlah penduduk '
        'menggunakan Linear Regression. Split data: 80% training, 20% testing (random_state=42).</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Penting:</b> Model ini mempelajari hubungan statistik antarwilayah pada dataset 2022 —
    <b>bukan ramalan</b> jumlah Puskesmas di masa depan. Hubungan ini bersifat asosiatif, 
    bukan bukti sebab-akibat. Bedakan antara estimasi regresi dan hasil pengelompokan K-Means.
    </div>
    """, unsafe_allow_html=True)

    scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled_ref = initialize_models(3)

    # Metrik evaluasi
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R² Score", f"{lr_result['r2']:.4f}", help="Proporsi varians yang dijelaskan model (0–1, lebih tinggi lebih baik)")
    col_m2.metric("MAE", f"{lr_result['mae']:.2f}", help="Mean Absolute Error — rata-rata selisih absolut prediksi")
    col_m3.metric("RMSE", f"{lr_result['rmse']:.2f}", help="Root Mean Squared Error — lebih sensitif terhadap outlier")
    col_m4.metric("Data Training", f"{lr_result['n_train']} / {lr_result['n_test']}", help="Jumlah data training vs testing")

    # Persamaan regresi
    st.markdown(f"""
    <div class="success-box">
    📐 <b>Persamaan Regresi:</b><br>
    <code style="font-size:1.1rem; color:#34D399;">
    Jumlah Puskesmas = {lr_result['intercept']:.4f} + {lr_result['coef']:.8f} × Jumlah Penduduk
    </code>
    </div>
    """, unsafe_allow_html=True)

    tab_f1, tab_f2, tab_f3, tab_f4 = st.tabs([
        "📈 Garis Regresi", "📉 Aktual vs Prediksi", "📊 Residual", "🔢 Estimasi Manual"
    ])

    with tab_f1:
        fig = viz.chart_regression_line(df, lr_result)
        st.plotly_chart(fig, use_container_width=True)

    with tab_f2:
        fig = viz.chart_actual_vs_predicted(lr_result["y_test"], lr_result["y_pred"])
        st.plotly_chart(fig, use_container_width=True)

        # Tabel prediksi vs aktual
        test_indices = range(len(lr_result["y_test"]))
        df_pred_view = pd.DataFrame({
            "Aktual": lr_result["y_test"],
            "Prediksi": lr_result["y_pred"].round(1),
            "Selisih": (lr_result["y_test"] - lr_result["y_pred"]).round(1),
            "Selisih (%)": ((lr_result["y_test"] - lr_result["y_pred"]) / lr_result["y_test"] * 100).round(1),
        })
        st.dataframe(df_pred_view, use_container_width=True, hide_index=True)

    with tab_f3:
        fig = viz.chart_residual(lr_result["y_test"], lr_result["y_pred"])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        Residual plot yang baik menunjukkan titik tersebar acak di sekitar garis nol 
        tanpa pola sistematis. Pola sistematis mengindikasikan model mungkin perlu variabel tambahan.
        </div>
        """, unsafe_allow_html=True)

    with tab_f4:
        st.markdown("#### 🔢 Estimasi Jumlah Puskesmas dari Penduduk")
        penduduk_est = st.number_input(
            "Masukkan Jumlah Penduduk", min_value=1, max_value=10_000_000,
            value=300_000, step=10_000, key="f_penduduk_est"
        )
        lr_model = load_lr_model() or lr_result["model"]
        est_pkm = predict_puskesmas(float(penduduk_est), lr_model)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E293B, #253047);
                    border-radius: 12px; border: 1px solid rgba(59,130,246,0.3);
                    padding: 24px; margin: 12px 0; text-align: center;">
            <div style="font-size:0.8rem; color:#64748B; text-transform:uppercase; margin-bottom:8px;">
                Estimasi Jumlah Puskesmas
            </div>
            <div style="font-size:3rem; font-weight:800; color:#60A5FA;">
                {max(0, est_pkm):.1f}
            </div>
            <div style="font-size:0.82rem; color:#94A3B8; margin-top:4px;">
                untuk {int(penduduk_est):,} penduduk
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        ⚠️ Nilai ini adalah estimasi berdasarkan hubungan statistik antarwilayah di dataset 2022, 
        bukan standar normatif atau target kebijakan. Jumlah Puskesmas yang ideal bergantung pada 
        banyak faktor lain seperti luas wilayah, aksesibilitas, dan standar Kemenkes.
        </div>
        """, unsafe_allow_html=True)

    # Penjelasan keterbatasan
    st.markdown("---")
    st.markdown("### ⚠️ Keterbatasan Model Regresi")
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1:
        st.markdown("""
        <div class="cluster-card">
        <b>🔴 Bukan Ramalan Masa Depan</b><br><br>
        Model mempelajari pola hubungan antarwilayah pada 2022, bukan tren temporal.
        </div>
        """, unsafe_allow_html=True)
    with col_lim2:
        st.markdown("""
        <div class="cluster-card">
        <b>🔴 Satu Variabel Bebas</b><br><br>
        Model hanya menggunakan jumlah penduduk. Faktor lain (luas wilayah, anggaran, dll.) tidak dimasukkan.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN G — MODEL & VALIDASI
# ════════════════════════════════════════════════════════════

elif page == "🔬  Model & Validasi":
    st.markdown('<div class="section-title">🔬 Ringkasan Model & Validasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Dokumentasi teknis algoritma, preprocessing, parameter, '
        'metrik evaluasi, keterbatasan, dan status pemeriksaan kualitas data.</div>',
        unsafe_allow_html=True,
    )

    scaler, kmeans, lr_result, elbow_data, sil_data, X_scaled_ref = initialize_models(3)
    df_cl = get_cluster_labels(df, X_scaled_ref, kmeans)
    prof = get_cluster_profile(df_cl)
    current_sil = sil_data["silhouette"][sil_data["k_values"].index(3)]

    tab_g1, tab_g2, tab_g3 = st.tabs(["🤖 K-Means", "📈 Linear Regression", "🔍 Kualitas Data"])

    with tab_g1:
        st.markdown("### 🤖 K-Means Clustering")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("""
            <div class="cluster-card">
            <b>Algoritma</b><br>
            <span style="color:#94A3B8;">K-Means (sklearn.cluster.KMeans)</span><br><br>
            <b>Tipe</b><br>
            <span style="color:#94A3B8;">Unsupervised Learning — Clustering</span><br><br>
            <b>Variabel Input</b><br>
            <span style="color:#60A5FA;">
            1. ln(1 + Jumlah Penduduk)<br>
            2. Rasio Puskesmas per 100.000<br>
            3. Proporsi Puskesmas Rawat Inap (%)
            </span><br><br>
            <b>Preprocessing</b><br>
            <span style="color:#94A3B8;">StandardScaler (fit pada 154 data referensi)</span>
            </div>
            """, unsafe_allow_html=True)
        with col_a2:
            st.markdown(f"""
            <div class="cluster-card">
            <b>Parameter Model</b><br>
            <span style="color:#94A3B8;">
            n_clusters = 3 (default, dapat diubah user)<br>
            n_init = 20<br>
            max_iter = 500<br>
            random_state = 42
            </span><br><br>
            <b>Metrik Evaluasi</b><br>
            <span style="color:#34D399;">
            Silhouette Score (K=3): {current_sil:.4f}<br>
            Inertia (K=3): {elbow_data["inertia"][1]:.2f}
            </span><br><br>
            <b>Data Training</b><br>
            <span style="color:#94A3B8;">154 wilayah Sumatra 2022</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        💡 <b>Catatan Implementasi Penting:</b><br>
        • Scaler di-fit SEKALI pada 154 data referensi, disimpan, dan digunakan untuk semua input baru.<br>
        • Scaler TIDAK di-fit ulang terhadap satu baris input pengguna.<br>
        • Label cluster ditetapkan dari nilai aktual (bukan asumsi nomor cluster).<br>
        • Model K-Means disimpan ke <code>models/kmeans_model.pkl</code>.
        </div>
        """, unsafe_allow_html=True)

    with tab_g2:
        st.markdown("### 📈 Linear Regression")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown(f"""
            <div class="cluster-card">
            <b>Algoritma</b><br>
            <span style="color:#94A3B8;">LinearRegression (sklearn.linear_model)</span><br><br>
            <b>Tipe</b><br>
            <span style="color:#94A3B8;">Supervised Learning — Regresi</span><br><br>
            <b>Variabel Bebas (X)</b><br>
            <span style="color:#60A5FA;">Jumlah Penduduk</span><br><br>
            <b>Variabel Target (Y)</b><br>
            <span style="color:#60A5FA;">Jumlah Puskesmas</span><br><br>
            <b>Split Data</b><br>
            <span style="color:#94A3B8;">80% Training ({lr_result["n_train"]} data) / 20% Testing ({lr_result["n_test"]} data)<br>
            random_state = 42</span>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown(f"""
            <div class="cluster-card">
            <b>Persamaan Regresi</b><br>
            <span style="color:#34D399; font-size:0.82rem;">
            Y = {lr_result["intercept"]:.4f} + {lr_result["coef"]:.8f} × X
            </span><br><br>
            <b>Metrik Evaluasi</b><br>
            <span style="color:#FCD34D;">
            R² Score: {lr_result["r2"]:.4f}<br>
            MAE: {lr_result["mae"]:.2f}<br>
            RMSE: {lr_result["rmse"]:.2f}
            </span><br><br>
            <b>Interpretasi R²</b><br>
            <span style="color:#94A3B8;">
            {"Baik (>0.7)" if lr_result["r2"] > 0.7 else "Sedang (0.5–0.7)" if lr_result["r2"] > 0.5 else "Lemah (<0.5)"}
            — {lr_result["r2"]*100:.1f}% varians Jumlah Puskesmas dijelaskan oleh Penduduk.
            </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Asumsi dan Keterbatasan:</b><br>
        • Model mengasumsikan hubungan linear antara penduduk dan jumlah Puskesmas.<br>
        • Hanya menggunakan satu variabel bebas — faktor lain tidak dimodelkan.<br>
        • BUKAN ramalan masa depan — hanya memodelkan hubungan statistik 2022.<br>
        • Tidak membuktikan hubungan sebab-akibat (korelasi ≠ kausalitas).
        </div>
        """, unsafe_allow_html=True)

    with tab_g3:
        st.markdown("### 🔍 Status Pemeriksaan Kualitas Data")

        checks = [
            ("✅", "Jumlah observasi", "154 wilayah", "Sesuai target 10 provinsi Sumatra"),
            ("✅", "Nilai null", "0 sel kosong", "Semua kolom inti lengkap"),
            ("✅", "Duplikat baris", "0 duplikat", "Data unik per wilayah"),
            ("✅", "Kode Kab/Kota duplikat", "0 kode duplikat", "Setiap kode unik"),
            ("✅", "Kode BPS duplikat", "0 kode duplikat", "Setiap kode BPS unik"),
            ("✅", "Pencocokan Puskesmas–Penduduk", "154/154 cocok", "Seluruh wilayah berhasil dicocokkan"),
            ("⚠️", "Kode Kab/Kota ≠ Kode BPS", "Berbeda untuk semua wilayah", "Keduanya disimpan terpisah"),
            ("⚠️", "8 Outlier teridentifikasi", "Semua diverifikasi", "Sesuai sumber KMK resmi — dipertahankan"),
        ]

        for icon, item, result, notes in checks:
            color = "#34D399" if icon == "✅" else "#FCD34D"
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:12px; 
                        padding:12px 16px; margin-bottom:6px;
                        background:rgba(30,41,59,0.5); border-radius:8px;
                        border-left:3px solid {color};">
                <span style="font-size:1.1rem;">{icon}</span>
                <div style="flex:1;">
                    <div style="font-weight:600; color:#E2E8F0; font-size:0.88rem;">{item}</div>
                    <div style="color:{color}; font-size:0.82rem; margin-top:2px;">{result}</div>
                    <div style="color:#64748B; font-size:0.76rem; margin-top:2px;">{notes}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
        📁 <b>Sumber data aktif:</b> 
        <code>Data/hasil_tahap4_normalisasi_puskesmas_2022_REVISI_TAHAP3_EDA.xlsx</code> — Sheet: Data_Gabungan<br>
        Detail lengkap: lihat file <code>DATA_QUALITY_NOTES.md</code> di root proyek.
        </div>
        """, unsafe_allow_html=True)
