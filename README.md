# E-Commerce Sales Forecasting Dashboard

Aplikasi berbasis Web Dashboard menggunakan **Streamlit** untuk memprediksi volume penjualan produk atau kategori *e-commerce* di berbagai wilayah (*Region*). Aplikasi ini mengintegrasikan 5 algoritma *Machine Learning* dan statistik untuk memberikan proyeksi jangka pendek secara akurat.

<br>

## 🛠️ Struktur & Fitur Utama Aplikasi

### 1. Control Panel (Sidebar)
* **Filter Wilayah (*Region Filter*):** Membatasi cakupan analisis data untuk wilayah tertentu (*Granular Forecasting*).
* **Basis Prediksi:** Memilih mode peramalan berdasarkan produk spesifik (*By Product*) atau kategori besar (*By Category*).
* **Periode Peramalan:** Mengubah granularitas waktu (Mingguan, Bulanan, Kuartalan).
* **Pemilihan Algoritma:** Menyediakan 5 pilihan model: *Linear Regression*, *Moving Average*, *XGBoost*, *Exponential Smoothing*, dan *Prophet*.
* **Rentang Tampilan Data:** Memotong visualisasi sumbu X (dari 20% hingga 100% data) untuk fokus pada tren terbaru.

### 2. Main Dashboard (Konten Utama)
* **KPI Metrics Block:** Menampilkan angka hasil prediksi periode berikutnya beserta metrik evaluasi model (WMAPE, MAE, RMSE).
* **Dynamic Interactive Chart:** Visualisasi interaktif menggunakan mesin *Vega-Lite* yang membedakan garis data aktual, evaluasi uji, dan hasil proyeksi masa depan.
* **Raw Dataset Preview (*Modal Popup*):** Jendela ringkasan statistik deskriptif dan pratinjau 50 baris pertama data mentah.

<br>

## 🧬 Alur Kerja Data Science (*Data Science Pipeline*)

Aplikasi ini menerapkan standardisasi *pipeline* data yang ketat di dalam `app.py`, mulai dari pemuatan data hingga evaluasi performa model:
```bash
[Dataset Mentah] ➔ [Pembersihan Data] ➔ [Filter Region] ➔ [Penyusunan Deret Waktu] ➔ [Rekayasa Fitur] ➔ [Split Data 80:20] ➔ [Pelatihan & Pengujian Model] ➔ [Evaluasi Metrik Akurasi]
```
### 1. Ingesti & Pembersihan Data (*Data Cleaning*)
* **Proteksi Data Kosong:** Menghapus baris yang kehilangan nilai pada kolom kritikal (`Product Name`, `Quantity`, `Category`, `Region`).
* **Standardisasi Waktu:** Mengonversi kolom `Order Date` menjadi format objek *Datetime* Pandas. Baris dengan format tanggal cacat otomatis dibuang.
* **Aplikasi *Caching*:** Proses ini dibungkus dekorator `@st.cache_data` agar data mentah tidak perlu dibaca ulang dari cakram keras setiap kali filter diubah.

### 2. Penyaringan Hulu (*Upstream Filtering*)
* Sebelum data disusun menjadi deret waktu, data dipotong berdasarkan *Region* yang dipilih. Hal ini penting agar model mempelajari karakteristik unik dari wilayah tersebut tanpa tercampur bias dari wilayah lain.

### 3. Penyusunan Deret Waktu & Penanganan Data Kosong (*Resampling & Re-indexing*)
* **Agregasi Waktu:** Transaksi harian yang tidak beraturan digabungkan (SUM) menjadi interval berkala (Mingguan `W`, Bulanan `ME`, atau Kuartalan `QE`).
* **Mengisi Kekosongan (*Handling Missing Timesteps*):** Jika ada periode waktu di mana barang tidak terjual sama sekali, sistem akan menyisipkan tanggal yang hilang tersebut dan mengisinya dengan angka `0`. Ini menjaga struktur waktu tetap kontinu (tidak bolong).

### 4. Rekayasa Fitur (*Feature Engineering*)
Untuk mendukung algoritma *Supervised Learning*, komponen tanggal dipecah menjadi fitur numerik baru:
* `Timestep`: Angka urut murni (0, 1, 2, dst.) untuk menangkap tren laju waktu.
* `Month` & `Year`: Untuk menangkap pola musiman (*seasonality*) tahunan.
* `Lag_1`: Nilai penjualan dari 1 periode sebelumnya, digunakan sebagai prediktor perilaku belanja terdekat.

### 5. Pembagian Data (*Data Splitting*)
* Data dipecah secara kronologis (bukan acak, karena ini data deret waktu) dengan proporsi **80% untuk Data Latih (*Training Data*)** dan **20% untuk Data Uji (*Testing Data*)**.

### 6. Proses Inti Pemodelan (*Model Training & Forecasting*)
Jika pengguna memilih satu item spesifik, sistem akan menjalankan salah satu dari 5 mesin matematika berikut:
* **Linear Regression:** Mencari tren garis lurus terbaik yang menghubungkan indeks waktu dengan kuantitas penjualan.
* **Moving Average:** Mengalkulasi nilai rata-rata dari 3 periode terakhir untuk memproyeksikan periode berikutnya.
* **XGBoost:** Algoritma berbasis *Decision Tree* yang mempelajari pola kompleks dari seluruh fitur (`Timestep`, `Month`, `Year`, `Lag_1`) secara bertahap.
* **Exponential Smoothing (Holt-Winters):** Model statistik yang memberikan bobot eksponensial lebih berat pada data-data penjualan terbaru.
* **Prophet:** Mengurai deret waktu menjadi komponen tren makro dan musiman mikro secara aditif.

> ⚠️ **Catatan Mode Multi-Item:** Jika pengguna memilih "All Products" atau "All Categories", aplikasi akan mematikan fungsi pelatihan model (*Fallback Mode*) untuk menghindari lonjakan beban komputasi server, lalu beralih menampilkan grafik komparasi performa antar-item.

### 7. Pengujian, Evaluasi, & Perhitungan Akurasi
Model diuji kemampuannya menggunakan 20% data uji yang sebelumnya dirahasiakan, kemudian akurasinya dihitung menggunakan 3 indikator:
* **WMAPE (*Weighted Mean Absolute Percentage Error*):** Mengukur persentase rata-rata kesalahan tebakan model.
* **Skor Akurasi (*Accuracy Score*):** Dihitung dari formula 100% - WMAPE. Angka inilah yang tampil pada boks informasi dashboard.
* **MAE (*Mean Absolute Error*):** Menampilkan rata-rata penyimpangan riil dalam satuan unit barang (misal: kesalahan tebakan rata-rata ± 5 pcs).
* **RMSE (*Root Mean Squared Error*):** Kembaran MAE, namun memberikan penalti bobot yang jauh lebih besar jika model membuat kesalahan tebakan yang sangat fatal.

<br>

## 🚀 Cara Menjalankan Aplikasi

1. Pastikan berkas `dataset.csv` berada dalam direktori yang sama dengan `app.py`.
2. Pasang pustaka yang dibutuhkan:
   ```
   pip install streamlit pandas numpy scikit-learn xgboost statsmodels prophet
   ```
3. Jalankan aplikasi melalui terminal:
   ```
   streamlit run app.py
   ```

<br>

---
---