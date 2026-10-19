# Dashboard Analisis Kesenjangan Ketersediaan Fasilitas Kesehatan di Sumatra 2022

Aplikasi Data Sains interaktif untuk menganalisis kesenjangan ketersediaan Puskesmas di 154 kabupaten/kota Sumatra tahun 2022, menggunakan K-Means Clustering dan Linear Regression.

## Instalasi

```powershell
# Pastikan Python 3.10+ sudah terinstal
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```powershell
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## Struktur Halaman

| Halaman | Keterangan |
|---|---|
| **A — Dashboard** | KPI cards: total wilayah, penduduk, Puskesmas, rasio |
| **B — Eksplorasi Data** | 6 visualisasi interaktif + tabel filter + ekspor CSV |
| **C — Clustering K-Means** | K-Means dengan Elbow Method, Silhouette, grafik cluster |
| **D — Input Wilayah Baru** | Form input manual → prediksi cluster via model terlatih |
| **E — Upload Data** | Upload Excel/CSV → batch clustering + ekspor hasil |
| **F — Linear Regression** | Estimasi Jumlah Puskesmas dari penduduk + metrik evaluasi |
| **G — Model & Validasi** | Ringkasan algoritma, parameter, metrik, keterbatasan |

## Dataset

- **Sumber utama:** `Data/hasil_tahap4_normalisasi_puskesmas_2022_REVISI_TAHAP3_EDA.xlsx`
- **Sheet:** `Data_Gabungan` (154 baris × 16 kolom)
- **Cakupan:** 10 provinsi, 154 kabupaten/kota Sumatra
- **Tahun:** 2022

## Fitur K-Means

1. ln(1 + Jumlah Penduduk)
2. Rasio Puskesmas per 100.000 Penduduk
3. Proporsi Puskesmas Rawat Inap (%)

## Teknologi

- Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Openpyxl

## Catatan Kualitas Data

Lihat [DATA_QUALITY_NOTES.md](DATA_QUALITY_NOTES.md) untuk detail temuan kualitas data.
