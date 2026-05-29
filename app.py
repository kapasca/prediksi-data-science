import streamlit as st
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Menonaktifkan log internal dari library Prophet dan Stan agar konsol terminal tetap bersih
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==============================================================================
# 01. KONFIGURASI APLIKASI & METADATA HALAMAN WEB
# ==============================================================================
# Mengatur tata letak halaman menjadi mode lebar (wide) dan menentukan judul aplikasi pada tab peramban
st.set_page_config(layout="wide", page_title="E-Commerce Sales Predictions", initial_sidebar_state="expanded")

# ==============================================================================
# 02. ARSITEKTUR VISUAL & INJEKSI CUSTOM CSS
# ==============================================================================
# Menyuntikkan kode CSS kustom untuk memodifikasi elemen bawaan Streamlit demi estetika dan responsivitas antarmuka
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Pengaturan kontainer utama: responsif tanpa mematikan scrollbar sistem */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            overflow-y: auto !important;
        }
        
        h1 {
            margin-bottom: 0.1rem !important;
            font-size: 1.8rem !important;
            color: #edae3e !important;
            text-align: center !important;
            width: 100%;
        }
        
        hr {
            margin-top: 0.4rem !important;
            margin-bottom: 1rem !important;
        }
        
        .stVegaLiteChart {
            height: 52vh !important;
            margin-top: 0.5rem !important;
        }
        
        .stAppDeployButton {
            display: none !important;
        }
        
        div[data-testid="stHorizontalBlock"] {
            margin-top: 0.2rem !important;
        }
        
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
        
        /* Mengurangi jarak antar blok komponen di sidebar */
        [data-testid="stSidebarUserContent"] .stElementContainer {
            margin-bottom: -10px !important;
        }
        
        /* Mengurangi padding vertikal bawaan widget selectbox */
        [data-testid="stSidebarUserContent"] div[data-baseweb="select"] {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        /* Styling khusus teks label di dalam sidebar */
        .sidebar-label {
            font-size: 0.8rem !important;
            color: #B0B0B0 !important;
            margin-bottom: 1rem;
            font-weight: 500;
        }
        
        .custom-legend-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 25px;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
            width: 100%;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 0.75rem;
            color: #E0E0E0;
            font-weight: 400;
        }
        
        .legend-color-box {
            width: 14px;
            height: 4px;
            border-radius: 2px;
            margin-right: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 03. PIPELINE DATA - FASE 1: INGESTI DATA & PEMBERSIHAN DATA (DATA CLEANING)
# ==============================================================================
@st.cache_data
def load_raw_data():
    """
    Fungsi untuk membaca data mentah dari berkas CSV.
    Menggunakan dekorator @st.cache_data agar data dimuat ke dalam memori lokal (cache),
    sehingga aplikasi tidak perlu membaca ulang berkas fisik setiap kali pengguna berinteraksi.
    """
    data = pd.read_csv("dataset.csv")
    
    # Menentukan kolom-kolom kritikal yang wajib ada dan tidak boleh kosong demi kelancaran proses pemodelan
    required_cols = ['Product Name', 'Order Date', 'Quantity']
    if 'Category' in data.columns:
        required_cols.append('Category')
    if 'Region' in data.columns:
        required_cols.append('Region')
        
    # Menghapus baris data yang tidak memiliki nilai (NaN) pada kolom-kolom kritikal tersebut
    data = data.dropna(subset=required_cols)
    
    # Mengonversi format kolom tanggal menjadi tipe objek Datetime standar Pandas untuk analisis deret waktu
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    
    # Menghapus kembali jika ada kegagalan konversi tanggal yang menghasilkan nilai NaT (Not a Time)
    data = data.dropna(subset=['Order Date'])
    return data

# Mengeksekusi fungsi pemuatan data dengan penanganan galat (Error Handling)
try:
    dataset_raw = load_raw_data()
except Exception as error:
    st.error(f"Failed to read dataset file: {error}")
    st.stop()

# Mengekstraksi daftar unik dari dataset untuk mengisi opsi pada komponen dropdown antarmuka pengguna
product_list = sorted(list(dataset_raw['Product Name'].unique()))
category_list = sorted(list(dataset_raw['Category'].unique())) if 'Category' in dataset_raw.columns else []
region_list = sorted(list(dataset_raw['Region'].unique())) if 'Region' in dataset_raw.columns else []

# ==============================================================================
# 04. SUB-RUTIN: MODAL PREVIEW DATASET MENTAH
# ==============================================================================
@st.dialog("Data Preview", width="large")
def show_dataset_preview_modal():
    """
    Fungsi sub-rutin untuk menampilkan jendela pop-up (modal) yang menyajikan
    ringkasan statistik serta cuplikan data mentah kepada pengguna.
    """
    
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 900; color: #edae3e; margin-bottom: 0; text-align: center;'>E-Commerce Sales Dataset</h2>", unsafe_allow_html=True)
    st.write("---")
    
    # Menampilkan metrik utama volume data menggunakan tata letak tiga kolom
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", len(dataset_raw))
    with col2:
        st.metric("Total Columns", len(dataset_raw.columns))
    with col3:
        st.metric("Unique Products", len(product_list))
    with col4:
        st.metric("Unique Categories", len(category_list))
    with col5:
        st.metric("Unique Regions", len(region_list))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        oldest_date = dataset_raw['Order Date'].min().strftime('%Y-%m-%d')
        newest_date = dataset_raw['Order Date'].max().strftime('%Y-%m-%d')
        avg_quantity_per_transaction = dataset_raw['Quantity'].mean()
        avg_sales_per_transaction = dataset_raw['Sales'].mean() if 'Sales' in dataset_raw.columns else "N/A"
        avg_profit_per_transaction = dataset_raw['Profit'].mean() if 'Profit' in dataset_raw.columns else "N/A"
        st.markdown(f"<div style='font-size: 0.75rem; color: #edae3e; margin-bottom: 0.5rem;'>Oldest Date<div style='font-size: 0.875rem; font-weight: bold; color: #fff;'>{oldest_date}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.75rem; color: #edae3e; margin-bottom: 0.5rem;'>Newest Date<div style='font-size: 0.875rem; font-weight: bold; color: #fff;'>{newest_date}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.75rem; color: #edae3e; margin-bottom: 0.5rem;'>Average Quantity<div style='font-size: 0.875rem; font-weight: bold; color: #fff;'>{avg_quantity_per_transaction:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.75rem; color: #edae3e; margin-bottom: 0.5rem;'>Average Sales<div style='font-size: 0.875rem; font-weight: bold; color: #fff;'>{avg_sales_per_transaction:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.75rem; color: #edae3e; margin-bottom: 0.5rem;'>Average Profit<div style='font-size: 0.875rem; font-weight: bold; color: #fff;'>{avg_profit_per_transaction:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='font-size: 0.75rem; color: #edae3e;'>List of Attributes</div>", unsafe_allow_html=True)
        column_names_html = "<div style='display: flex; flex-direction: column; margin-top: 0.5rem; gap: 5px;'>"
        for col in dataset_raw.columns:
            column_names_html += f"<div style='background-color: #2A2A2A; padding: 5px 10px; border-radius: 5px; font-size: 0.75rem;'>{col}</div>"
        column_names_html += "</div>"
        st.markdown(column_names_html, unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='font-size: 0.75rem; color: #edae3e;'>List of Products</div>", unsafe_allow_html=True)
        products_html = "<div style='display: flex; flex-direction: column; margin-top: 0.5rem; gap: 5px;'>"
        for product in product_list:
            products_html += f"<div style='background-color: #2A2A2A; padding: 5px 10px; border-radius: 5px; font-size: 0.75rem;'>{product}</div>"
        products_html += "</div>"
        st.markdown(products_html, unsafe_allow_html=True)
    with col4:
        st.markdown("<div style='font-size: 0.75rem; color: #edae3e;'>List of Categories</div>", unsafe_allow_html=True)
        categories_html = "<div style='display: flex; flex-direction: column; margin-top: 0.5rem; gap: 5px;'>"
        for category in category_list:
            categories_html += f"<div style='background-color: #2A2A2A; padding: 5px 10px; border-radius: 5px; font-size: 0.75rem;'>{category}</div>"
        categories_html += "</div>"
        st.markdown(categories_html, unsafe_allow_html=True)
    with col5:
        st.markdown("<div style='font-size: 0.75rem; color: #edae3e;'>List of Regions</div>", unsafe_allow_html=True)
        regions_html = "<div style='display: flex; flex-direction: column; margin-top: 0.5rem; gap: 5px;'>"
        for region in region_list:
            regions_html += f"<div style='background-color: #2A2A2A; padding: 5px 10px; border-radius: 5px; font-size: 0.75rem;'>{region}</div>"
        regions_html += "</div>"
        st.markdown(regions_html, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("<div style='color: #cecece; font-weight: 300; font-size: 1.2rem; margin: 1rem 0;'><span style='color: #edae3e; font-weight: bold;'>Review All Data</span> (RAW Dataset)</div>", unsafe_allow_html=True)
    st.dataframe(dataset_raw, use_container_width=True, height=350)    
    
    st.markdown("---")
    if st.button("Close Preview", key="btn_close_dataset", use_container_width=True):
        st.rerun()

# ==============================================================================
# 05. KONTROL ANTARMUKA PENGGUNA (SIDEBAR PANEL)
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.3rem; font-weight: 900; color: #edae3e; margin-top: -1.6rem; margin-bottom: 2.5rem; background-color: #0e1117; text-align: center; border-radius: 10px;'>Control Panel</h2>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 1: Basis Prediksi (Berdasarkan Produk Spesifik atau Kategori Global)
    st.markdown("<div class='sidebar-label'>Prediction Basis</div>", unsafe_allow_html=True)
    prediction_basis = st.selectbox("", ["By Product", "By Category"], index=0, label_visibility="collapsed", key="ctl_basis")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 2: Filter Nama Item secara Adaptif mengikuti Basis Prediksi yang dipilih
    st.markdown("<div class='sidebar-label'>Item Filter</div>", unsafe_allow_html=True)
    if prediction_basis == "By Product":
        item_options = ["All Products"] + product_list
        target_column = 'Product Name'
        fallback_all_label = "All Products"
    else:
        item_options = ["All Categories"] + category_list
        target_column = 'Category'
        fallback_all_label = "All Categories"
        
    selected_item = st.selectbox("", item_options, index=0, label_visibility="collapsed", key="ctl_item")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 3: Wilayah Kerja (Region)
    st.markdown("<div class='sidebar-label'>Region Filter</div>", unsafe_allow_html=True)
    region_options = ["All Regions"] + region_list
    selected_region = st.selectbox("", region_options, index=0, label_visibility="collapsed", key="ctl_region")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 4: Granularitas Waktu atau Interval Agregasi Data Deret Waktu
    st.markdown("<div class='sidebar-label'>Forecasting Period</div>", unsafe_allow_html=True)
    selected_period = st.selectbox("", ["Monthly", "Quarterly", "Weekly"], index=0, label_visibility="collapsed", key="ctl_period")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 5: Penentuan Algoritma Matematika / Machine Learning yang akan digunakan
    st.markdown("<div class='sidebar-label'>Machine Learning Algorithm</div>", unsafe_allow_html=True)
    algorithm_options = ["Linear Regression", "Moving Average", "XGBoost", "Exponential Smoothing", "Prophet", "ARIMA"]
    selected_method = st.selectbox("", algorithm_options, index=0, label_visibility="collapsed", key="ctl_method")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # PILIHAN FILTER 6: Batasan Rentang Data Historis yang Ditampilkan pada Sumbu Grafik X
    st.markdown("<div class='sidebar-label'>Displayed Data Range</div>", unsafe_allow_html=True)
    range_options = ["All Data", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%"]
    selected_range = st.selectbox("", range_options, index=0, label_visibility="collapsed", key="ctl_range")
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    for _ in range(2):
        st.write()
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    if st.button("View Dataset Summary", use_container_width=True, key="btn_view_dataset"):
        show_dataset_preview_modal()

# Mengonversi string pilihan frekuensi menjadi kode parameter penanggalan standar Pandas
if selected_period == "Monthly":
    freq_code = 'ME' # Akhir Bulan (Month End)
elif selected_period == "Quarterly":
    freq_code = 'QE' # Akhir Kuartal (Quarter End)
else:
    freq_code = 'W' # Mingguan (Weekly)

st.title("E-Commerce Sales Predictions")
st.write("---")

# Membuat kontainer dinamis (placeholder) kosong yang nantinya akan diisi oleh komponen visualisasi grafik
placeholder_chart = st.empty()

# ==============================================================================
# FASE UPSTREAM FILTERING: KETERLIBATAN WILAYAH (REGION) DALAM PIPELINE DATA SCIENCE
# ==============================================================================
# PENJELASAN DATA SCIENCE: Agar model cerdas dan peka terhadap dinamika lokal (Granular Forecasting),
# data dipotong berdasarkan Region di hulu SEBELUM proses penyusunan ulang struktur data (resampling).
# Hal ini memastikan fase pelatihan (training) model benar-benar murni mempelajari pola transaksi dari wilayah tersebut.
if selected_region != "All Regions":
    dataset_working = dataset_raw[dataset_raw['Region'] == selected_region]
else:
    dataset_working = dataset_raw.copy()

# ==============================================================================
# 06. PIPELINE DATA - FASE 2: PEMROSESAN & PERAMALAN MODEL (MODE ITEM TUNGGAL)
# ==============================================================================
if selected_item != fallback_all_label:
    
    # --------------------------------------------------------------------------
    # SUB-FASE A: PENYUSUNAN ULANG STRUKTUR DERET WAKTU & RE-INDEXING
    # --------------------------------------------------------------------------
    # Menyaring data berdasarkan item spesifik yang dipilih oleh pengguna
    dataset_filtered = dataset_working[dataset_working[target_column] == selected_item]
    
    # Antisipasi jika pada kombinasi wilayah dan item tersebut tidak ditemukan jejak transaksi historis sama sekali
    if len(dataset_filtered) == 0:
        st.warning(f"No historical data found for {selected_item} in Region: {selected_region}")
        st.stop()
        
    # PENJELASAN DATA SCIENCE (Handling Missing Timesteps): 
    # Transaksi e-commerce bersifat intermiten (tidak terjadi setiap hari). Kita harus membuat linimasa waktu yang utuh 
    # tanpa ada hari/bulan yang bolong dari tanggal awal hingga akhir, kemudian menjumlahkan kuantitas penjualan (SUM).
    start_date = dataset_filtered['Order Date'].min()
    end_date = dataset_filtered['Order Date'].max()
    
    df_interim = dataset_filtered.resample(freq_code, on='Order Date')['Quantity'].sum()
    actual_end = max(end_date, df_interim.index.max())
    full_timeline = pd.date_range(start=start_date, end=actual_end, freq=freq_code)
    
    # Mengisi kekosongan periode dengan nilai 0 (asumsi tidak ada penjualan) agar pola temporal tidak rusak saat dipelajari model
    dataset_resampled = df_interim.reindex(full_timeline, fill_value=0).to_frame()
    dataset_resampled = dataset_resampled.sort_index()
    
    original_datetime_index = dataset_resampled.index.copy()
    
    # Membuat label string kustom untuk sumbu X grafik agar nyaman dibaca manusia
    if selected_period == "Weekly":
        chart_string_labels = dataset_resampled.index.map(lambda dt: f"{dt.strftime('%Y-%m')} (W-{dt.isocalendar()[1]})").tolist()
    elif selected_period == "Quarterly":
        chart_string_labels = dataset_resampled.index.map(lambda dt: f"{dt.strftime('%Y')}-Q{(dt.month-1)//3 + 1}").tolist()
    else:
        chart_string_labels = dataset_resampled.index.strftime('%Y-%m').tolist()

    # PENJELASAN DATA SCIENCE (Feature Engineering):
    # Mengonversi informasi tanggal menjadi fitur-fitur numerik baru agar dapat dicerna oleh algoritma Machine Learning supervised.
    dataset_resampled['Timestep'] = np.arange(len(dataset_resampled)) # Angka urut sebagai representasi berjalannya waktu
    dataset_resampled['Month'] = dataset_resampled.index.month # Menangkap pola musiman bulanan (Seasonality)
    dataset_resampled['Year'] = dataset_resampled.index.year # Menangkap tren pertumbuhan tahunan
    dataset_resampled['Lag_1'] = dataset_resampled['Quantity'].shift(1).bfill() # Penjualan periode sebelumnya sebagai prediktor
    
    # Menentukan fitur pembelajar berdasarkan kebutuhan kompleksitas algoritma
    if selected_method == "XGBoost":
        feature_cols = ['Timestep', 'Month', 'Year', 'Lag_1'] # XGBoost butuh banyak konteks fitur
    else:
        feature_cols = ['Timestep'] # Model statistik dasar hanya berbasis laju indeks waktu
        
    X_features = dataset_resampled[feature_cols].values # Matriks Fitur (Variabel Independen)
    y_target = dataset_resampled['Quantity'].values # Target Luaran (Variabel Dependen)
    
    # PENJELASAN DATA SCIENCE (Data Splitting):
    # Membagi data menjadi 80% Data Latih (Training) untuk melatih kecerdasan model,
    # dan 20% Data Uji (Testing) yang dirahasiakan dari model untuk mengukur kemampuan akurasi riilnya nanti.
    split_index = int(len(dataset_resampled) * 0.8)
    if split_index == 0: split_index = 1
    
    X_train, X_test = X_features[:split_index], X_features[split_index:]
    y_train, y_test = y_target[:split_index], y_target[split_index:]
    
    # Inisialisasi variabel penampung nilai metrik performa
    predicted_value = 0
    mape_error = 0
    mae_val = 0
    rmse_val = 0
    
    y_eval_preds = np.full(len(dataset_resampled), np.nan, dtype=float)
    y_pred_test = np.zeros(len(y_test))
    
    # --------------------------------------------------------------------------
    # SUB-FASE B: INTI MESIN PROSES PEMODELAN ML & FORECASTING STATISTIK
    # --------------------------------------------------------------------------
    if len(dataset_resampled) >= 2:
        
        # --- ALGORITMA 1: LINEAR REGRESSION (Regresi Linear) ---
        # Pola pikir: Mencari garis lurus terbaik yang menghubungkan pergerakan waktu dengan volume penjualan.
        if selected_method == "Linear Regression":
            # Tahap Evaluasi: Melatih model pada 80% data untuk memprediksi sisa 20% data uji
            model_evaluator = LinearRegression()
            model_evaluator.fit(X_train, y_train)
            y_pred_test = model_evaluator.predict(X_test)
            y_pred_test = np.maximum(0, y_pred_test.astype(int)) # Nilai kuantitas barang tidak boleh bernilai negatif
            
            # Tahap Produksi: Menggunakan 100% data penuh untuk memproyeksikan penjualan di masa depan
            model_final = LinearRegression()
            model_final.fit(X_features, y_target)
            next_timestep = np.array([[len(dataset_resampled)]])
            predicted_value = max(0, int(model_final.predict(next_timestep)[0]))
            
        # --- ALGORITMA 2: MOVING AVERAGE (Rata-rata Bergerak) ---
        # Pola pikir: Memprediksi masa depan murni berdasarkan rata-rata penjualan dari beberapa periode terakhir.
        elif selected_method == "Moving Average":
            rolling_window = min(3, len(y_train)) # Mengambil jendela 3 periode ke belakang
            y_pred_list = []
            training_history = list(y_train)
            
            # Simulasi prediksi langkah demi langkah melintasi garis data uji (Testing Data)
            for idx in range(len(y_test)):
                window_prediction = np.mean(training_history[-rolling_window:]) if len(training_history) >= rolling_window else 0
                y_pred_list.append(window_prediction)
                training_history.append(y_test[idx]) # Memasukkan nilai aktual sebenarnya ke dalam histori untuk langkah berikutnya
            
            y_pred_test = np.array(y_pred_list).astype(int)
            predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= rolling_window else 0

        # --- ALGORITMA 3: XGBOOST (Extreme Gradient Boosting) ---
        # Pola pikir: Algoritma tingkat lanjut berbasis pohon keputusan (Decision Trees) yang belajar secara bertahap dari kesalahan sebelumnya.
        elif selected_method == "XGBoost":
            if len(dataset_resampled) < 5:
                # Fallback aman jika data terlalu sedikit
                rolling_window = min(3, len(y_train))
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= 2 else 0
            else:
                # 1. Tahap Evaluasi yang Valid (Recursive Forecasting)
                model_evaluator = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_evaluator.fit(X_train, y_train)
                
                # Kita prediksi data uji secara rekursif satu per satu
                y_pred_test_list = []
                # Ambil nilai lag awal dari ujung data training
                current_lag = y_train[-1] 
                
                for idx in range(len(y_test)):
                    # Ambil baris fitur untuk data uji saat ini
                    current_features = X_test[idx].copy()
                    # Paksa fitur Lag_1 diisi oleh hasil prediksi kita sebelumnya (bukan nilai riil dataset)
                    current_features[3] = current_lag 
                    
                    # Lakukan prediksi satu langkah
                    pred_step = model_evaluator.predict(np.array([current_features]))[0]
                    pred_step = max(0, int(pred_step))
                    y_pred_test_list.append(pred_step)
                    
                    # Update nilai lag untuk iterasi berikutnya menggunakan hasil prediksi barusan
                    current_lag = pred_step
                
                y_pred_test = np.array(y_pred_test_list)
                
                # 2. Tahap Produksi (Gunakan Data Penuh)
                model_final = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_final.fit(X_features, y_target)
                
                last_date_parsed = original_datetime_index[-1]
                if selected_period == "Monthly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=1)
                elif selected_period == "Quarterly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=3)
                else:
                    future_date_obj = last_date_parsed + pd.Timedelta(weeks=1)
                
                # Fitur lag untuk masa depan murni diambil dari titik terakhir data aktual penuh
                X_future_step = np.array([[len(dataset_resampled), future_date_obj.month, future_date_obj.year, y_target[-1]]])
                predicted_value = max(0, int(model_final.predict(X_future_step)[0]))

        # --- ALGORITMA 4: EXPONENTIAL SMOOTHING (Pemulusan Eksponensial / Holt-Winters) ---
        # Pola pikir: Model statistik penanggalan waktu yang memberikan bobot lebih berat pada data terbaru dibandingkan data masa lampau.
        elif selected_method == "Exponential Smoothing":
            try:
                # Gunakan strategi rolling yang sama agar grafik bergelombang dinamis
                history = list(y_train)
                es_preds = []
                
                for idx in range(len(y_test)):
                    model_temp = ExponentialSmoothing(history, initialization_method="estimated").fit()
                    pred_one_step = model_temp.forecast(1)
                    es_preds.append(pred_one_step[0])
                    history.append(y_test[idx])
                    
                y_pred_test = np.maximum(0, np.array(es_preds).astype(int))
                
                model_final = ExponentialSmoothing(y_target, initialization_method="estimated").fit()
                predicted_value = max(0, int(model_final.forecast(1)[0]))
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0

        # --- ALGORITMA 5: PROPHET (Dikembangkan oleh Meta/Facebook) ---
        # Pola pikir: Model aditif canggih yang memecah deret waktu menjadi komponen Tren, Musiman (Seasonality), dan Efek Hari Libur.
        elif selected_method == "Prophet":
            # Kebijakan Khusus: Prophet mewajibkan struktur kolom yang kaku, yaitu kolom waktu dinamai 'ds' dan target dinamai 'y'
            df_prophet = pd.DataFrame({'ds': original_datetime_index, 'y': y_target})
            df_train = df_prophet.iloc[:split_index]
            
            try:
                m_eval = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m_eval.fit(df_train)
                
                future_eval = m_eval.make_future_dataframe(periods=len(y_test), freq=freq_code)
                forecast_eval = m_eval.predict(future_eval)
                y_pred_test = forecast_eval['yhat'].iloc[split_index:].values
                y_pred_test = np.maximum(0, y_pred_test.astype(int))
                
                m_final = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m_final.fit(df_prophet)
                future_final = m_final.make_future_dataframe(periods=1, freq=freq_code)
                forecast_final = m_final.predict(future_final)
                predicted_value = max(0, int(forecast_final['yhat'].iloc[-1]))
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0
                
        # --- ALGORITMA 6: ARIMA (Autoregressive Integrated Moving Average) ---
        # Pola pikir: Model statistik klasik yang menggabungkan komponen Autoregressive (AR), Differencing (I), dan Moving Average (MA) untuk menangkap pola temporal.
        elif selected_method == "ARIMA":
            series_train = pd.Series(y_train, index=original_datetime_index[:split_index])
            series_target = pd.Series(y_target, index=original_datetime_index)
            
            series_train.index.freq = freq_code
            series_target.index.freq = freq_code
            
            try:
                # REKAYASA ROLLING EVALUATION AGAR GRAFIK TESTING TIDAK LURUS FLAT
                history = list(y_train)
                arima_preds = []
                
                # Prediksi satu per satu melintasi data uji
                for idx in range(len(y_test)):
                    current_series = pd.Series(history, index=original_datetime_index[:split_index+idx])
                    current_series.index.freq = freq_code
                    
                    # Latih model dengan histori yang terus bertambah
                    model_temp = ARIMA(current_series, order=(1, 1, 1))
                    model_temp_fitted = model_temp.fit()
                    
                    # Ramal 1 langkah ke depan
                    pred_one_step = model_temp_fitted.forecast(steps=1)
                    arima_preds.append(pred_one_step.iloc[0])
                    
                    # Masukkan data aktual uji ke histori untuk modal prediksi bulan berikutnya
                    history.append(y_test[idx])
                
                y_pred_test = np.maximum(0, np.array(arima_preds).astype(int))
                
                # 2. Tahap Produksi (Tetap Ramal Masa Depan)
                model_final = ARIMA(series_target, order=(1, 1, 1))
                model_final_fitted = model_final.fit()
                future_forecast = model_final_fitted.forecast(steps=1)
                predicted_value = max(0, int(future_forecast.iloc[0]))
                
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0

        # Menyisipkan hasil prediksi uji ke dalam susunan data evaluasi untuk visualisasi grafik (Garis Kuning)
        y_eval_preds[split_index:] = y_pred_test

        # PENJELASAN DATA SCIENCE (Metrik Evaluasi Performa Model):
        # 1. WMAPE (Weighted Mean Absolute Percentage Error): Mengukur persentase rata-rata deviasi kesalahan prediksi.
        sum_actual = np.sum(y_test)
        if sum_actual > 0:
            mape_error = (np.sum(np.abs(y_test - y_pred_test)) / sum_actual) * 100
        else:
            mape_error = 100.0 if np.sum(y_pred_test) > 0 else 0.0
            
        # 2. MAE (Mean Absolute Error): Rata-rata absolut jarak selisih antara jumlah barang asli dengan hasil tebakan model.
        mae_val = mean_absolute_error(y_test, y_pred_test)
        
        # 3. RMSE (Root Mean Squared Error): Kembaran MAE namun memberikan hukuman (penalti) kuadrat jauh lebih berat pada kesalahan tebakan yang fatal.
        rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # --------------------------------------------------------------------------
    # SUB-FASE C: FORMALISASI KPI METRIK & PENGEPAKAN DATAFRAME UNTUK GRAFIK
    # --------------------------------------------------------------------------
    # Menghitung skor akurasi teoritis berbasis 100 dikurangi persentase galat WMAPE
    accuracy_score = max(0.0, 100.0 - mape_error)
    
    if selected_period == "Monthly":
        periode_label = "Next Month"
    elif selected_period == "Quarterly":
        periode_label = "Next Quarter"
    else:
        periode_label = "Next Week"
    
    # Membuat blok informasi HTML dinamis untuk diletakkan pada header visual bagian kiri
    html_title_box = f"""
    <div style='text-align: left; line-height: 1.2;'>
        <div style='font-size: 0.9rem; color: #E0E0E0; font-weight: 500;'>Sales Statistics ({prediction_basis} | Scope: {selected_region})</div>
        <div style='font-size: 1.6rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;'>{selected_item}</div>
        <div style='font-size: 0.75rem; color: #B0B0B0; margin-top: 0.3rem;'>{selected_period} using <span style='color: #00FFA6; font-weight: 600;'>{selected_method}</span></div>
    </div>
    """
    
    # Membuat blok informasi HTML dinamis untuk diletakkan pada header visual bagian kanan (Berisi Nilai Prediksi & Skor Akurasi)
    html_info_box = f"""
    <div style='text-align: right; line-height: 1.3;'>
        <div style='font-size: 0.85rem; color: #E0E0E0; font-weight: 500;'>Forecast for {periode_label}</div>
        <div style='font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin: 1px 0;'>{predicted_value:,} Qty</div>
        <div style='font-size: 0.72rem; color: #B0B0B0; margin-top: 0.3rem;'>
            <span style='color: #FFFFFF; font-weight: 600;'>[Accuracy Score]</span> 
            WMAPE: <span style='color: #00FFA6; font-weight: 600;'>{accuracy_score:.1f}%</span><br/>
            MAE: <span style='color: #FFB300; font-weight: 600;'>± {mae_val:.0f} qty</span> | 
            RMSE: <span style='color: #FF6B6B; font-weight: 600;'>± {rmse_val:.0f} qty</span>
        </div>
    </div>
    """
    
    # Mengalkulasi label string penunjuk masa depan di sumbu X grafik
    last_date_parsed = original_datetime_index[-1]
    if selected_period == "Monthly":
        future_date_string = (last_date_parsed + pd.DateOffset(months=1)).strftime('%Y-%m')
    elif selected_period == "Quarterly":
        future_date = last_date_parsed + pd.DateOffset(months=3)
        future_date_string = f"{future_date.strftime('%Y')}-Q{(future_date.month-1)//3 + 1}"
    else:
        future_date = last_date_parsed + pd.Timedelta(weeks=1)
        future_date_string = f"{future_date.strftime('%Y-%m')} (W-{future_date.isocalendar()[1]})"

    # Menyusun tabel visualisasi data tunggal (Single Render Dataframe)
    df_single_render = pd.DataFrame(index=chart_string_labels, columns=['Historical Sales', 'Model Evaluation [20%]', 'Forecast'])
    df_single_render['Historical Sales'] = y_target
    df_single_render['Model Evaluation [20%]'] = y_eval_preds
    
    # Menyambungkan ujung garis historis dengan pangkal awal garis ramalan masa depan agar tidak terputus secara visual
    last_historical_label = chart_string_labels[-1]
    df_single_render.loc[last_historical_label, 'Forecast'] = y_target[-1]
    df_single_render.loc[future_date_string] = [np.nan, np.nan, predicted_value]
    
    # Memotong baris visualisasi jika pengguna meminta pembatasan persentase data (Show Data Filter)
    if selected_range != "All Data":
        pct = int(selected_range.replace("%", "")) / 100.0
        keep_points = max(2, int(len(df_single_render) * pct))
        df_single_render = df_single_render.iloc[-keep_points:]
        
    df_single_render = df_single_render.replace([np.inf, -np.inf], np.nan)
    
    # Melelehkan struktur dataframe (Melt Technique) dari format mendatar (Wide) menjadi memanjang ke bawah (Long)
    # Ini merupakan standar format masukan mutlak agar library grafik Vega-Lite/Altair bisa membedakan warna garis.
    df_melted = df_single_render.reset_index().rename(columns={'index': 'Date'}).melt('Date', var_name='Category', value_name='Amount')
    
    # Konfigurasi skema warna pembeda untuk masing-masing tipe kategori garis pada grafik tunggal
    color_schema = {
        "field": "Category", 
        "type": "nominal", 
        "scale": {"domain": ['Historical Sales', 'Model Evaluation [20%]', 'Forecast'], "range": ["#4A90E2", "#F5A623", "#FF4B4B"]},
        "legend": None
    }
    
    # Injeksi legenda penunjuk warna kustom berbasis elemen HTML div agar serasi dengan desain tema gelap aplikasi
    html_custom_legend = """
    <div class='custom-legend-container'>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #4A90E2;'></div>Historical Sales</div>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #F5A623;'></div>Model Evaluation [20%]</div>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #FF4B4B;'></div>Forecast</div>
    </div>
    """

else:
    # ==============================================================================
    # 07. PIPELINE DATA - MODE MULTI ITEM / SEMUA ITEM (FALLBACK MULTI LINE CHART)
    # ==============================================================================
    # PENJELASAN LOGIKA APLIKASI: Jika pengguna memilih opsi "All Products" atau "All Categories", 
    # aplikasi tidak menjalankan proses latih model karena beban komputasi akan meledak secara berlebihan. 
    # Sebagai gantinya, aplikasi menampilkan grafik komparasi pertumbuhan volume antar item secara langsung.
    html_title_box = f"""
    <div style='text-align: left; line-height: 1.2;'>
        <div style='font-size: 0.9rem; color: #E0E0E0; font-weight: 500;'>Sales Statistics (Scope: {selected_region})</div>
        <div style='font-size: 1.6rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;'>{selected_item}</div>
        <div style='font-size: 0.75rem; color: #B0B0B0; margin-top: 0.3rem;'>Data Source: <span style='color: #00FFA6; font-weight: 600;'>dataset.csv</span></div>
    </div>
    """
    html_info_box = "" # Mengosongkan boks info metrik akurasi karena tidak ada pemodelan yang dieksekusi
    
    if len(dataset_working) == 0:
        st.warning(f"No historical transaction data available for this region selection: {selected_region}")
        st.stop()
        
    # Mengompilasi linimasa global komparatif
    global_timeline = dataset_working.resample(freq_code, on='Order Date')['Quantity'].sum().index
    if selected_period == "Weekly":
        fallback_labels = global_timeline.map(lambda dt: f"{dt.strftime('%Y-%m')} (W-{dt.isocalendar()[1]})").tolist()
    elif selected_period == "Quarterly":
        fallback_labels = global_timeline.map(lambda dt: f"{dt.strftime('%Y')}-Q{(dt.month-1)//3 + 1}").tolist()
    else:
        fallback_labels = global_timeline.strftime('%Y-%m').tolist()
        
    df_all_render = pd.DataFrame(index=fallback_labels)
    
    # Menentukan target perulangan item berdasarkan basis prediksi yang aktif
    loop_items = product_list if prediction_basis == "By Product" else category_list
    
    # Memetakan performa kuantitas penjualan masing-masing item ke dalam kolom-kolom tabel mandiri
    for item in loop_items:
        dataset_item_filter = dataset_working[dataset_working[target_column] == item]
        if len(dataset_item_filter) > 0:
            dataset_item_res = dataset_item_filter.resample(freq_code, on='Order Date')['Quantity'].sum().reindex(global_timeline, fill_value=0)
            df_all_render[item] = dataset_item_res.values
        else:
            df_all_render[item] = 0 # Diisi nol jika item tersebut tidak pernah terjual sama sekali di wilayah bersangkutan
        
    if selected_range != "All Data":
        pct = int(selected_range.replace("%", "")) / 100.0
        keep_points = max(2, int(len(df_all_render) * pct))
        df_all_render = df_all_render.iloc[-keep_points:]
        
    # Transformasi struktur data meleleh (Melt) untuk persiapan plotting multi-line chart
    df_melted = df_all_render.reset_index().rename(columns={'index': 'Date'}).melt('Date', var_name='Category', value_name='Amount')
    
    # Biarkan sistem grafik mengalokasikan variasi warna garis secara otomatis (Nominal Color Mapping)
    color_schema = {
        "field": "Category", 
        "type": "nominal",
        "legend": None
    }
    
    # Penyusunan komponen legenda melingkar kustom untuk mewakili identitas puluhan item sekaligus
    html_custom_legend = "<div class='custom-legend-container' style='flex-wrap: wrap; max-width: 80%; margin: 0 auto; gap: 10px;'>"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, item in enumerate(loop_items):
        c = colors[i % len(colors)]
        html_custom_legend += f"<div class='legend-item'><div class='legend-color-box' style='background-color: {c}; height: 10px; width: 10px; border-radius: 50%;'></div>{item}</div>"
    html_custom_legend += "</div>"

# ==============================================================================
# 08. INJEKSI KONTEN DINAMIS & EKSEKUSI RENDERING GRAFIK MESIN VEGA-LITE
# ==============================================================================
with placeholder_chart.container():
    # Membuat struktur baris atas berisi judul informasi di kiri dan skor akurasi di kanan
    col_title, col_info = st.columns([1, 1])
    
    with col_title:
        st.markdown(html_title_box, unsafe_allow_html=True)
        
    with col_info:
        if html_info_box:
            st.markdown(html_info_box, unsafe_allow_html=True)
            
    # Menyuntikkan legenda visual kustom tepat di atas badan grafik
    st.markdown(html_custom_legend, unsafe_allow_html=True)
            
    # Deklarasi JSON spesifikasi teknis pembuatan grafik berbasis pustaka mesin deklaratif Vega-Lite
    vega_lite_spec = {
        "width": "container",
        "height": "container",
        "mark": {
            "type": "line", 
            "tooltip": True, # Mengaktifkan boks informasi nilai (tooltip) saat kursor mouse mendekati titik data
            "interpolate": "linear", # Metode penarikan garis lurus antar koordinat titik waktu
            "point": {"size": 60, "filled": True, "cursor": "pointer"} # Mempertegas visualisasi simpul koordinat berupa lingkaran
        },
        "encoding": {
            "x": {
                "field": "Date",
                "type": "nominal",
                "sort": None,
                "axis": {
                    "labelAngle": -90,
                    "title": None
                }
            },
            "y": {
                "field": "Amount",
                "type": "quantitative",
                "axis": {
                    "title": "Transaction Volume (Qty)"
                }
            },
            "color": color_schema
        },
        "config": {
            "background": "transparent", # Mengintegrasikan latar belakang grafik agar menyatu jernih dengan tema gelap Streamlit
            "view": {"stroke": "transparent"}
        }
    }
    
    # Melempar olahan data final beserta spesifikasi teknisnya ke dalam komponen visualisasi bawaan Streamlit
    st.vega_lite_chart(df_melted, vega_lite_spec, use_container_width=True)