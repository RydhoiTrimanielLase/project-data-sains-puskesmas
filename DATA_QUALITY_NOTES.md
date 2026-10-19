# DATA QUALITY NOTES
# Catatan Kualitas Data — Proyek Puskesmas Sumatra 2022

## Sumber Data Aktif Aplikasi

**File:** `Data/hasil_tahap4_normalisasi_puskesmas_2022_REVISI_TAHAP3_EDA.xlsx`  
**Sheet:** `Data_Gabungan`  
**Alasan dipilih:** File ini adalah tahap akhir pengolahan data (Tahap 4 — Normalisasi + Revisi EDA), berisi 154 baris data lengkap, 0 nilai null, 0 duplikat, dan sudah mencakup semua variabel turunan yang diperlukan (rasio, log penduduk, proporsi rawat inap).

---

## Ringkasan Kualitas Data

| Pemeriksaan | Hasil | Keterangan |
|---|---|---|
| Jumlah observasi | 154 wilayah | Sesuai target (154 kab/kota Sumatra) |
| Jumlah provinsi | 10 provinsi | Semua provinsi Sumatra terwakili |
| Nilai null | 0 | Data lengkap |
| Duplikat baris | 0 | Tidak ada duplikat |
| Kode Kab/Kota duplikat | 0 | Unik |
| Kode BPS duplikat | 0 | Unik |

---

## Perbedaan Kode Wilayah

**TEMUAN PENTING:** `Kode Kab/Kota` (sumber Puskesmas/KMK) **TIDAK SAMA** dengan `Kode BPS` (sumber penduduk BPS).

Contoh:
- ACEH SELATAN: Kode Kab/Kota = 1101, Kode BPS = 1103
- ACEH TENGGARA: Kode Kab/Kota = 1102, Kode BPS = 1104

**Keputusan:** Kedua kode disimpan terpisah di kolom berbeda. Pencocokan dilakukan berdasarkan nama wilayah dan nama provinsi, bukan kode numerik.

---

## Outlier Teridentifikasi

8 wilayah teridentifikasi sebagai kandidat outlier berdasarkan metode IQR:

| Wilayah | Variabel Pemicu | Status Verifikasi |
|---|---|---|
| ACEH BESAR | Puskesmas Rawat Inap (24) | ✅ Sesuai sumber KMK resmi |
| DELI SERDANG (Sumut) | Puskesmas Rawat Inap (27) | ✅ Sesuai sumber KMK resmi |
| SIMALUNGUN (Sumut) | Non RI + Jumlah Puskesmas | ✅ Sesuai sumber KMK resmi |
| (5 wilayah lainnya) | Bervariasi | ✅ Sesuai sumber KMK resmi |

**Keputusan:** Seluruh outlier dipertahankan karena terverifikasi sesuai sumber resmi. Data tidak dimodifikasi.

---

## Batas IQR Variabel Utama

| Variabel | Q1 | Q3 | IQR | Batas Bawah | Batas Atas |
|---|---|---|---|---|---|
| Puskesmas Non Rawat Inap | 6.00 | 14 | 8.00 | -6.000 | 26.000 |
| Puskesmas Rawat Inap | 4.00 | 10 | 6.00 | -5.000 | 19.000 |
| Jumlah Puskesmas | 11.25 | 22 | 10.75 | -4.875 | 38.125 |

---

## Distribusi Wilayah per Provinsi

| Provinsi | Jumlah Kab/Kota |
|---|---|
| Sumatera Utara | 33 |
| Aceh | 23 |
| Sumatera Barat | 19 |
| Lampung | 15 |
| Sumatera Selatan | 17 |
| Jambi | 11 |
| Riau | 12 |
| Bengkulu | 10 |
| Kepulauan Bangka Belitung | 7 |
| Kepulauan Riau | 7 |
| **Total** | **154** |

---

## Catatan Metode Rasio Provinsi

Rasio agregat per provinsi dihitung sebagai:

> **Rasio Provinsi = (Total Puskesmas Provinsi / Total Penduduk Provinsi) × 100.000**

Ini **berbeda** dari rata-rata sederhana rasio tiap kabupaten/kota. Metode agregat ini dipilih karena lebih mencerminkan kondisi aktual provinsi secara keseluruhan dan tidak terpengaruh oleh perbedaan ukuran wilayah.

---

## Fitur K-Means yang Digunakan

Sesuai instruksi tugas:
1. `log_penduduk` = ln(1 + Jumlah Penduduk) = `Log Jumlah Penduduk` dari dataset
2. `rasio_puskesmas` = Rasio Puskesmas per 100.000 Penduduk
3. `proporsi_ri` = Proporsi Puskesmas Rawat Inap (%)

Scaler: `StandardScaler`, di-fit pada 154 data referensi, disimpan dan digunakan konsisten untuk input baru.

---

## Masalah Kualitas Data yang Belum Terselesaikan

1. **URL Sumber:** Beberapa URL mungkin sudah tidak aktif (tidak diverifikasi ulang saat aplikasi dibuat).
2. **Tahun data:** Seluruh data merujuk pada tahun 2022. Aplikasi tidak dapat digunakan untuk analisis tren multitahun.
3. **Kepulauan Riau:** Hanya 7 kab/kota — secara statistik terbatas untuk analisis di level provinsi.

---

*Catatan ini dibuat otomatis saat inisialisasi aplikasi. Terakhir diperbarui: 2022-2026.*
