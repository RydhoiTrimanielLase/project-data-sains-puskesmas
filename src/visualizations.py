"""
visualizations.py — Fungsi visualisasi Plotly interaktif untuk semua halaman aplikasi.

Semua fungsi mengembalikan objek plotly.graph_objects.Figure.
Tidak ada side-effect Streamlit di modul ini (separation of concerns).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palet warna proyek
COLORS = {
    "primary": "#2563EB",      # Biru utama
    "secondary": "#7C3AED",    # Ungu
    "success": "#059669",      # Hijau
    "warning": "#D97706",      # Oranye
    "danger": "#DC2626",       # Merah
    "neutral": "#6B7280",      # Abu-abu
    "bg": "#0F172A",           # Background gelap
    "card": "#1E293B",         # Card background
}

CLUSTER_COLORS = px.colors.qualitative.Set2
FONT_FAMILY = "Inter, Arial, sans-serif"

PLOTLY_LAYOUT_BASE = dict(
    font=dict(family=FONT_FAMILY, size=12, color="#E2E8F0"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=30, t=60, b=60),
)


def _apply_base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Terapkan layout dasar ke semua chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F1F5F9"), x=0.02),
        **PLOTLY_LAYOUT_BASE,
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            linecolor="rgba(148,163,184,0.3)",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            linecolor="rgba(148,163,184,0.3)",
            tickfont=dict(color="#94A3B8"),
        ),
    )
    return fig


# ──────────────────────────────────────────────
# HALAMAN B — EKSPLORASI DATA
# ──────────────────────────────────────────────

def chart_top10_rasio_tertinggi(df: pd.DataFrame) -> go.Figure:
    """Bar chart: 10 wilayah dengan rasio Puskesmas tertinggi."""
    col_rasio = "Rasio Puskesmas per 100.000 Penduduk"
    col_kab = "Kabupaten/Kota"
    col_prov = "Provinsi"

    top10 = df.nlargest(10, col_rasio).sort_values(col_rasio)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10[col_rasio],
        y=top10[col_kab] + " (" + top10[col_prov] + ")",
        orientation="h",
        marker=dict(
            color=top10[col_rasio],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Rasio"),
        ),
        text=top10[col_rasio].round(2).astype(str) + " per 100k",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Rasio: <b>%{x:.2f}</b> per 100.000 penduduk<br>"
            "<extra></extra>"
        ),
    ))
    fig = _apply_base_layout(fig, "📈 10 Wilayah dengan Rasio Puskesmas Tertinggi")
    fig.update_layout(
        xaxis_title="Rasio Puskesmas per 100.000 Penduduk",
        yaxis_title="",
        height=420,
        showlegend=False,
    )
    return fig


def chart_top10_rasio_terendah(df: pd.DataFrame) -> go.Figure:
    """Bar chart: 10 wilayah dengan rasio Puskesmas terendah."""
    col_rasio = "Rasio Puskesmas per 100.000 Penduduk"
    col_kab = "Kabupaten/Kota"
    col_prov = "Provinsi"

    bot10 = df.nsmallest(10, col_rasio).sort_values(col_rasio, ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bot10[col_rasio],
        y=bot10[col_kab] + " (" + bot10[col_prov] + ")",
        orientation="h",
        marker=dict(
            color=bot10[col_rasio],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Rasio"),
        ),
        text=bot10[col_rasio].round(2).astype(str) + " per 100k",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Rasio: <b>%{x:.2f}</b> per 100.000 penduduk<br>"
            "<extra></extra>"
        ),
    ))
    fig = _apply_base_layout(fig, "📉 10 Wilayah dengan Rasio Puskesmas Terendah")
    fig.update_layout(
        xaxis_title="Rasio Puskesmas per 100.000 Penduduk",
        yaxis_title="",
        height=420,
        showlegend=False,
    )
    return fig


def chart_rasio_provinsi(summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart: rasio agregat per provinsi (total pkm / total penduduk × 100k)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_df["rasio_agregat"],
        y=summary_df["Provinsi"],
        orientation="h",
        marker=dict(
            color=summary_df["rasio_agregat"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Rasio"),
        ),
        text=summary_df["rasio_agregat"].round(2).astype(str) + " per 100k",
        textposition="outside",
        customdata=summary_df[["total_puskesmas", "total_penduduk", "jumlah_wilayah"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Rasio agregat: <b>%{x:.2f}</b> per 100.000 penduduk<br>"
            "Total Puskesmas: %{customdata[0]:,.0f}<br>"
            "Total Penduduk: %{customdata[1]:,.0f}<br>"
            "Jumlah Kab/Kota: %{customdata[2]}<br>"
            "<extra></extra>"
        ),
    ))
    fig = _apply_base_layout(fig, "🏥 Rasio Agregat Puskesmas per Provinsi")
    fig.update_layout(
        xaxis_title="Rasio Puskesmas per 100.000 Penduduk",
        yaxis_title="",
        height=420,
    )
    return fig


def chart_distribusi_rasio(df: pd.DataFrame) -> go.Figure:
    """Histogram distribusi rasio Puskesmas per 100k penduduk."""
    col = "Rasio Puskesmas per 100.000 Penduduk"
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col],
        nbinsx=25,
        marker=dict(
            color=COLORS["primary"],
            opacity=0.8,
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        hovertemplate="Rentang: %{x:.1f}<br>Jumlah wilayah: %{y}<extra></extra>",
    ))
    # Garis median
    median_val = df[col].median()
    fig.add_vline(
        x=median_val, line_dash="dash", line_color=COLORS["warning"],
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["warning"]),
    )
    fig = _apply_base_layout(fig, "📊 Distribusi Rasio Puskesmas per 100.000 Penduduk")
    fig.update_layout(
        xaxis_title="Rasio Puskesmas per 100.000 Penduduk",
        yaxis_title="Jumlah Wilayah",
        height=380,
    )
    return fig


def chart_scatter_penduduk_puskesmas(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Jumlah Penduduk vs Jumlah Puskesmas, warna per provinsi."""
    col_pend = "Jumlah Penduduk"
    col_pkm = "Jumlah Puskesmas"
    col_prov = "Provinsi"
    col_kab = "Kabupaten/Kota"

    fig = px.scatter(
        df,
        x=col_pend,
        y=col_pkm,
        color=col_prov,
        hover_name=col_kab,
        hover_data={
            col_pend: ":,.0f",
            col_pkm: True,
            "Rasio Puskesmas per 100.000 Penduduk": ":.2f",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    # Garis tren
    x_vals = df[col_pend].values
    y_vals = df[col_pkm].values
    coef = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = np.polyval(coef, x_line)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        name="Tren Linear",
        line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=2),
    ))

    _apply_base_layout(fig, "🔵 Hubungan Jumlah Penduduk dan Jumlah Puskesmas")
    fig.update_layout(
        xaxis_title="Jumlah Penduduk",
        yaxis_title="Jumlah Puskesmas",
        height=480,
        legend=dict(
            font=dict(color="#CBD5E1", size=10),
            bgcolor="rgba(30,41,59,0.8)",
        ),
    )
    fig.update_xaxes(tickformat=",.0f")
    return fig


def chart_heatmap_korelasi(df: pd.DataFrame) -> go.Figure:
    """Heatmap korelasi variabel numerik utama."""
    cols = [
        "Jumlah Penduduk",
        "Jumlah Puskesmas",
        "Puskesmas Non Rawat Inap",
        "Puskesmas Rawat Inap",
        "Rasio Puskesmas per 100.000 Penduduk",
        "Proporsi Puskesmas Rawat Inap (%)",
        "Log Jumlah Penduduk",
    ]
    corr = df[cols].corr()
    short_labels = [
        "Penduduk", "Jml Pkm", "Non RI", "RI",
        "Rasio", "% RI", "Log Pddk",
    ]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=short_labels,
        y=short_labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#E2E8F0"),
        hovertemplate="<b>%{y} × %{x}</b><br>Korelasi: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color="#CBD5E1")),
    ))
    fig = _apply_base_layout(fig, "🔥 Heatmap Korelasi Variabel Utama")
    fig.update_layout(height=480)
    return fig


# ──────────────────────────────────────────────
# HALAMAN C — CLUSTERING
# ──────────────────────────────────────────────

def chart_elbow(elbow_data: dict) -> go.Figure:
    """Line chart Elbow Method."""
    k_vals = elbow_data["k_values"]
    inertia = elbow_data["inertia"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals, y=inertia,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=9, color=COLORS["primary"], line=dict(color="white", width=1.5)),
        hovertemplate="K=%{x}<br>Inertia: %{y:,.1f}<extra></extra>",
        name="Inertia (WCSS)",
    ))
    fig = _apply_base_layout(fig, "📐 Elbow Method — Menentukan Jumlah Cluster Optimal")
    fig.update_layout(
        xaxis=dict(title="Jumlah Cluster (K)", dtick=1),
        yaxis_title="Inertia (WCSS)",
        height=360,
    )
    return fig


def chart_silhouette(sil_data: dict) -> go.Figure:
    """Bar chart Silhouette Score per K."""
    k_vals = sil_data["k_values"]
    scores = sil_data["silhouette"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_vals, y=scores,
        marker=dict(
            color=scores,
            colorscale="Greens",
            showscale=False,
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        hovertemplate="K=%{x}<br>Silhouette Score: %{y:.4f}<extra></extra>",
    ))
    fig = _apply_base_layout(fig, "📏 Silhouette Score per Jumlah Cluster")
    fig.update_layout(
        xaxis=dict(title="Jumlah Cluster (K)", dtick=1),
        yaxis_title="Silhouette Score",
        height=360,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def chart_cluster_scatter(
    df_clustered: pd.DataFrame,
    x_col: str = "Log Jumlah Penduduk",
    y_col: str = "Rasio Puskesmas per 100.000 Penduduk",
    highlight_point: dict | None = None,
) -> go.Figure:
    """
    Scatter plot cluster.

    Parameters
    ----------
    df_clustered : pd.DataFrame
        Dataset dengan kolom 'Cluster'.
    highlight_point : dict | None
        Dict {'x': ..., 'y': ..., 'name': ...} untuk highlight input baru.
    """
    fig = px.scatter(
        df_clustered,
        x=x_col,
        y=y_col,
        color="Cluster",
        hover_name="Kabupaten/Kota",
        hover_data={
            "Provinsi": True,
            "Jumlah Penduduk": ":,.0f",
            y_col: ":.2f",
            "Proporsi Puskesmas Rawat Inap (%)": ":.1f",
        },
        color_discrete_sequence=CLUSTER_COLORS,
        symbol="Cluster",
        title="",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))

    if highlight_point:
        fig.add_trace(go.Scatter(
            x=[highlight_point["x"]],
            y=[highlight_point["y"]],
            mode="markers+text",
            marker=dict(
                size=18, color=COLORS["danger"],
                symbol="star", line=dict(color="white", width=2),
            ),
            text=[highlight_point.get("name", "Input Baru")],
            textposition="top right",
            textfont=dict(color=COLORS["danger"], size=12),
            name="Input Baru",
            hovertemplate=f"<b>{highlight_point.get('name','Input Baru')}</b><br>"
                          f"{x_col}: {highlight_point['x']:.3f}<br>"
                          f"{y_col}: {highlight_point['y']:.3f}<extra></extra>",
        ))

    _apply_base_layout(fig, f"🔵 Visualisasi Cluster: {y_col} vs {x_col}")
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        legend=dict(
            title="Cluster",
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(30,41,59,0.8)",
        ),
    )
    return fig


def chart_cluster_3d(df_clustered: pd.DataFrame) -> go.Figure:
    """3D scatter cluster menggunakan 3 fitur K-Means."""
    fig = px.scatter_3d(
        df_clustered,
        x="Log Jumlah Penduduk",
        y="Rasio Puskesmas per 100.000 Penduduk",
        z="Proporsi Puskesmas Rawat Inap (%)",
        color="Cluster",
        hover_name="Kabupaten/Kota",
        hover_data={"Provinsi": True},
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title=dict(text="🌐 Visualisasi 3D Cluster", font=dict(color="#F1F5F9")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family=FONT_FAMILY),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.2)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.2)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.2)"),
        ),
        height=550,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ──────────────────────────────────────────────
# HALAMAN F — LINEAR REGRESSION
# ──────────────────────────────────────────────

def chart_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, df_test_meta: pd.DataFrame | None = None) -> go.Figure:
    """Scatter plot aktual vs prediksi pada data test."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(
            color=COLORS["primary"],
            size=10,
            opacity=0.8,
            line=dict(color="white", width=1),
        ),
        hovertemplate="Aktual: %{x}<br>Prediksi: %{y:.1f}<extra></extra>",
        name="Data Test",
    ))
    # Garis ideal y=x
    lim_min = min(y_test.min(), y_pred.min()) - 1
    lim_max = max(y_test.max(), y_pred.max()) + 1
    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max], y=[lim_min, lim_max],
        mode="lines",
        line=dict(color=COLORS["success"], dash="dash", width=2),
        name="Prediksi Sempurna",
    ))
    fig = _apply_base_layout(fig, "📉 Aktual vs Prediksi Jumlah Puskesmas (Data Test)")
    fig.update_layout(
        xaxis_title="Jumlah Puskesmas Aktual",
        yaxis_title="Jumlah Puskesmas Prediksi",
        height=450,
        legend=dict(font=dict(color="#CBD5E1"), bgcolor="rgba(30,41,59,0.8)"),
    )
    return fig


def chart_regression_line(df: pd.DataFrame, lr_result: dict) -> go.Figure:
    """Scatter plot seluruh data dengan garis regresi."""
    col_x = "Jumlah Penduduk"
    col_y = "Jumlah Puskesmas"

    x_all = df[col_x].values
    y_all = df[col_y].values

    x_line = np.linspace(x_all.min(), x_all.max(), 200)
    y_line = lr_result["intercept"] + lr_result["coef"] * x_line

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_all, y=y_all,
        mode="markers",
        marker=dict(
            color=COLORS["secondary"],
            size=8, opacity=0.7,
            line=dict(color="white", width=0.5),
        ),
        hovertext=df["Kabupaten/Kota"],
        hovertemplate="<b>%{hovertext}</b><br>Penduduk: %{x:,.0f}<br>Puskesmas: %{y}<extra></extra>",
        name="Data Aktual",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=COLORS["warning"], width=2.5),
        name=f"Garis Regresi (R²={lr_result['r2']:.3f})",
        hoverinfo="skip",
    ))
    fig = _apply_base_layout(fig, "📈 Garis Regresi Linear: Penduduk → Jumlah Puskesmas")
    fig.update_layout(
        xaxis_title="Jumlah Penduduk",
        yaxis_title="Jumlah Puskesmas",
        height=450,
        legend=dict(font=dict(color="#CBD5E1"), bgcolor="rgba(30,41,59,0.8)"),
    )
    fig.update_xaxes(tickformat=",.0f")
    return fig


def chart_residual(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residual plot: Prediksi vs Residual."""
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(
            color=residuals,
            colorscale="RdYlGn",
            cmid=0,
            size=9,
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Residual"),
        ),
        hovertemplate="Prediksi: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig = _apply_base_layout(fig, "📊 Residual Plot")
    fig.update_layout(
        xaxis_title="Nilai Prediksi",
        yaxis_title="Residual (Aktual − Prediksi)",
        height=360,
    )
    return fig
