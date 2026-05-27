---
title: E-Commerce Sales Predictions
emoji: 📈
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 📈 E-Commerce Sales Predictions Dashboard

Aplikasi analisis dan prediksi penjualan produk *E-Commerce* berbasis web yang interaktif. Dashboard ini dirancang menggunakan **Streamlit** dan memanfaatkan berbagai algoritma *Machine Learning* serta statistik untuk melakukan *forecasting* data deret waktu (*time-series*) baik secara bulanan maupun mingguan.

---

## 🚀 Fitur Utama

- **Multi-Algorithm Forecasting**: Anda dapat membandingkan hasil prediksi dari berbagai algoritma populer:
  - Linear Regression
  - Moving Average
  - XGBoost (Machine Learning)
  - Exponential Smoothing (Statistik)
  - Prophet (Facebook/Meta Open Source)
- **Flexible Time-Series Resampling**: Analisis data penjualan secara **Bulanan (Monthly)** atau **Mingguan (Weekly)**.
- **Granular Filter**: Analisis spesifik per produk atau melihat agregasi penjualan global ("All Products").
- **Robust Model Evaluation**: Transparansi performa model menggunakan 3 metrik evaluasi sekaligus:
  - **WMAPE**: Skor Akurasi berbasis persentase ($100 - \text{WMAPE}$).
  - **MAE**: Rata-rata nilai melesetnya prediksi dalam satuan item ($\pm \text{X items}$).
  - **RMSE**: Pengukur eror yang sensitif terhadap lonjakan data ekstrem ($\pm \text{X items}$).
- **Interactive Line Chart**: Visualisasi interaktif menggunakan Vega-Lite dengan titik koordinat (*point markers*) yang besar sehingga memudahkan pengguna melakukan *hovering* untuk melihat detail data.

---

## 🛠️ Struktur File Proyek

Di dalam Space ini terdapat beberapa file utama:
```text
├── app.py                # File kode utama aplikasi Streamlit
├── dataset.csv           # File data mentah transaksi penjualan e-commerce
├── requirements.txt      # Daftar library Python yang dibutuhkan oleh server
└── README.md             # Dokumentasi aplikasi (File ini)