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

# Nonaktifkan log internal dari pustaka Prophet dan Stan untuk menjaga kebersihan konsol terminal
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==============================================================================
# 01. KONFIGURASI APLIKASI & METADATA HALAMAN
# ==============================================================================
st.set_page_config(layout="wide", page_title="E-Commerce Sales Predictions", initial_sidebar_state="expanded")

# ==============================================================================
# 02. ARSITEKTUR VISUAL & INJEKSI CUSTOM CSS
# ==============================================================================
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        
        /* Gaya Kontainer Utama Dashboard */
        .dashboard-stage {
            background-color: #11151C;
            border: 1px solid #222E3D;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }
        
        /* Desain Metrik Blok KPI */
        .metric-box {
            background: linear-gradient(135deg, #161F2B 0%, #0F1622 100%);
            border-left: 5px solid #00E5FF;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-bottom: 12px;
        }
        .metric-label {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8A99AD;
            font-weight: 500;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #FFFFFF;
            margin-top: 4px;
        }
        
        /* Desain Kotak Informasi Metrik Evaluasi Model */
        .html-info-box {
            background-color: #171E28;
            border: 1px solid #253346;
            border-radius: 8px;
            padding: 15px;
            height: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 03. LOGIKA BACKEND: PEMUATAN DATA & CACHING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset():
    """
    Memuat dataset dari berkas CSV lokal dan melakukan konversi tipe data esensial.
    Returns:
        pd.DataFrame: Dataset yang siap diproses dengan kolom tanggal terstruktur.
    """
    try:
        df = pd.read_csv("dataset.csv")
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df = df.dropna(subset=['Order Date'])
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {str(e)}")
        return pd.DataFrame()

# Inisialisasi pemuatan dataset
raw_dataset = load_and_preprocess_dataset()

if raw_dataset.empty:
    st.warning("Aplikasi tidak dapat dilanjutkan karena dataset kosong atau tidak ditemukan berkas 'dataset.csv'.")
    st.stop()

# ==============================================================================
# 04. PANEL KONTROL (SIDEBAR INTERFACE)
# ==============================================================================
with st.sidebar:
    st.markdown("<h2 style='margin-top:0; color:#00E5FF;'>🎛️ Control Panel</h2>", unsafe_allow_html=True)
    st.write("Atur parameter analisis dan pemodelan prediksi di bawah ini:")
    st.markdown("---")
    
    # Kriteria 1: Filter Segmentasi Wilayah (Region)
    available_regions = ["All Regions"] + sorted(raw_dataset['Region'].dropna().unique().tolist())
    selected_region = st.selectbox("🎯 Target Region", available_regions, index=0)
    
    # Kriteria 2: Basis Metode Prediksi
    forecasting_basis = st.radio("📈 Basis Prediksi", ["By Product Name", "By Category"], index=0)
    
    # Filter dinamis untuk daftar item berdasarkan basis prediksi yang dipilih
    if forecasting_basis == "By Product Name":
        item_label = "Nama Produk"
        target_column = 'Product Name'
        available_items = ["All Products"] + sorted(raw_dataset['Product Name'].dropna().unique().tolist())
    else:
        item_label = "Kategori"
        target_column = 'Category'
        available_items = ["All Categories"] + sorted(raw_dataset['Category'].dropna().unique().tolist())
        
    selected_item = st.selectbox(f"📦 Pilih {item_label}", available_items, index=0)
    
    # Kriteria 3: Granularitas Waktu (Siklus Periode)
    period_mapping = {
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE"
    }
    selected_period = st.selectbox("📅 Periode Peramalan", list(period_mapping.keys()), index=1)
    freq_code = period_mapping[selected_period]
    
    # Kriteria 4: Pemilihan Algoritma Forecasting
    supported_methods = ["Linear Regression", "Moving Average", "XGBoost", "Exponential Smoothing", "Prophet", "ARIMA"]
    selected_method = st.selectbox("🤖 Algoritma Forecasting", supported_methods, index=2)
    
    # Kriteria 5: Filter Rentang Visualisasi Grafik (Sumbu X)
    st.markdown("---")
    st.write("👁️ Pengaturan Tampilan Grafik:")
    display_range_options = ["All Data", "Last 80% Data", "Last 50% Data", "Last 20% Data"]
    selected_range = st.radio("Rentang Tampilan Data", display_range_options, index=0)

# ==============================================================================
# 05. ENGINERING HUB: FILTERING & RESAMPLING DATA TIME SERIES
# ==============================================================================
# Tahap 1: Penyaringan data berdasarkan Region
if selected_region != "All Regions":
    dataset_filtered = raw_dataset[raw_dataset['Region'] == selected_region]
else:
    dataset_filtered = raw_dataset.copy()

# Cek kondisi aktivasi Mode Fallback (Multi-Item)
is_fallback_mode = False
if (forecasting_basis == "By Product Name" and selected_item == "All Products") or \
   (forecasting_basis == "By Category" and selected_item == "All Categories"):
    is_fallback_mode = True

# Proses berlanjut jika pengguna memilih item tunggal secara spesifik
if not is_fallback_mode:
    dataset_filtered = dataset_filtered[dataset_filtered[target_column] == selected_item]
    
    if dataset_filtered.empty:
        st.info("Tidak ada transaksi untuk kombinasi filter yang Anda pilih.")
        st.stop()
        
    # Urutkan kronologi berdasarkan tanggal transaksi
    dataset_filtered = dataset_filtered.sort_values(by='Order Date')
    
    # Penentuan batas linimasa yang aman agar ujung akhir periode tidak terpotong frekuensi Pandas
    start_date = dataset_filtered['Order Date'].min()
    end_date = dataset_filtered['Order Date'].max()
    
    # Agregasi volume penjualan (Quantity) berdasarkan kode frekuensi waktu
    df_interim = dataset_filtered.resample(freq_code, on='Order Date')['Quantity'].sum()
    actual_end = max(end_date, df_interim.index.max())
    
    # Rekonstruksi linimasa waktu yang utuh tanpa celah periode kosong
    full_timeline = pd.date_range(start=start_date, end=actual_end, freq=freq_code)
    dataset_resampled = df_interim.reindex(full_timeline, fill_value=0).to_frame()
    dataset_resampled = dataset_resampled.sort_index()
    
    # Penyusunan matriks fitur untuk model Supervised Learning (Fase Rekayasa Fitur)
    original_datetime_index = dataset_resampled.index
    dataset_resampled['Timestep'] = np.arange(len(dataset_resampled))
    dataset_resampled['Month'] = original_datetime_index.month
    dataset_resampled['Year'] = original_datetime_index.year
    dataset_resampled['Lag_1'] = dataset_resampled['Quantity'].shift(1).fillna(0)
    
    # Definisikan target array dan matriks fitur numerik
    y_target = dataset_resampled['Quantity'].values
    feature_cols = ['Timestep', 'Month', 'Year', 'Lag_1']
    X_features = dataset_resampled[feature_cols].values
    
    # Pembagian proporsi dataset: 80% Data Pelatihan dan 20% Data Pengujian
    split_index = int(len(dataset_resampled) * 0.8)
    if split_index >= len(dataset_resampled):
        split_index = max(1, len(dataset_resampled) - 1)
        
    X_train, X_test = X_features[:split_index], X_features[split_index:]
    y_train, y_test = y_target[:split_index], y_target[split_index:]

# ==============================================================================
# 06. CORE ENGINE: MATEMATIKA MODEL & EXECUTION FORECAST
# ==============================================================================
predicted_value = 0
y_pred_test = np.array([])

if not is_fallback_mode:
    # Validasi batas minimum baris data untuk melakukan pemodelan statistik
    if len(dataset_resampled) >= 2:
        
        # --- ALGORITMA 1: LINEAR REGRESSION ---
        if selected_method == "Linear Regression":
            X_train_lr = X_train[:, :3]
            X_test_lr = X_test[:, :3]
            
            model_evaluator = LinearRegression()
            model_evaluator.fit(X_train_lr, y_train)
            y_pred_test = model_evaluator.predict(X_test_lr)
            y_pred_test = np.maximum(0, y_pred_test.astype(int))
            
            model_final = LinearRegression()
            X_features_lr = X_features[:, :3]
            model_final.fit(X_features_lr, y_target)
            
            last_date_parsed = original_datetime_index[-1]
            if selected_period == "Monthly":
                future_date_obj = last_date_parsed + pd.DateOffset(months=1)
            elif selected_period == "Quarterly":
                future_date_obj = last_date_parsed + pd.DateOffset(months=3)
            else:
                future_date_obj = last_date_parsed + pd.Timedelta(weeks=1)
                
            X_future_step = np.array([[len(dataset_resampled), future_date_obj.month, future_date_obj.year]])
            predicted_value = max(0, int(model_final.predict(X_future_step)[0]))
            
        # --- ALGORITMA 2: MOVING AVERAGE ---
        elif selected_method == "Moving Average":
            rolling_window = min(3, len(y_train))
            y_pred_list = []
            history = list(y_train)
            
            for idx in range(len(y_test)):
                window_prediction = np.mean(history[-rolling_window:]) if len(history) >= rolling_window else 0
                y_pred_list.append(window_prediction)
                history.append(y_test[idx]) 
            
            y_pred_test = np.array(y_pred_list).astype(int)
            predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= rolling_window else 0

        # --- ALGORITMA 3: XGBOOST ---
        elif selected_method == "XGBoost":
            if len(dataset_resampled) < 5:
                rolling_window = min(3, len(y_train))
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= 2 else 0
            else:
                model_evaluator = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_evaluator.fit(X_train, y_train)
                
                y_pred_test_list = []
                for idx in range(len(y_test)):
                    current_features = X_test[idx].copy()
                    pred_step = model_evaluator.predict(np.array([current_features]))[0]
                    y_pred_test_list.append(max(0, int(pred_step)))
                
                y_pred_test = np.array(y_pred_test_list)
                
                model_final = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_final.fit(X_features, y_target)
                
                last_date_parsed = original_datetime_index[-1]
                if selected_period == "Monthly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=1)
                elif selected_period == "Quarterly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=3)
                else:
                    future_date_obj = last_date_parsed + pd.Timedelta(weeks=1)
                
                X_future_step = np.array([[len(dataset_resampled), future_date_obj.month, future_date_obj.year, y_target[-1]]])
                predicted_value = max(0, int(model_final.predict(X_future_step)[0]))

        # --- ALGORITMA 4: EXPONENTIAL SMOOTHING ---
        elif selected_method == "Exponential Smoothing":
            try:
                es_preds = []
                history = list(y_train)
                
                for idx in range(len(y_test)):
                    model_temp = ExponentialSmoothing(history, initialization_method="estimated").fit()
                    pred_one_step = model_temp.forecast(1)[0]
                    es_preds.append(pred_one_step)
                    history.append(y_test[idx])
                    
                y_pred_test = np.maximum(0, np.array(es_preds).astype(int))
                
                model_final = ExponentialSmoothing(y_target, initialization_method="estimated").fit()
                predicted_value = max(0, int(model_final.forecast(1)[0]))
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0

        # --- ALGORITMA 5: PROPHET ---
        elif selected_method == "Prophet":
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
                
        # --- ALGORITMA 6: ARIMA ---
        elif selected_method == "ARIMA":
            try:
                arima_preds = []
                history = list(y_train)
                
                for idx in range(len(y_test)):
                    current_index = original_datetime_index[:split_index + idx]
                    series_temp = pd.Series(history, index=current_index)
                    series_temp.index.freq = freq_code
                    
                    model_temp = ARIMA(series_temp, order=(1, 1, 1))
                    model_temp_fitted = model_temp.fit()
                    
                    pred_one_step = model_temp_fitted.forecast(steps=1)[0]
                    arima_preds.append(pred_one_step)
                    history.append(y_test[idx])
                
                y_pred_test = np.maximum(0, np.array(arima_preds).astype(int))
                
                series_target = pd.Series(y_target, index=original_datetime_index)
                series_target.index.freq = freq_code
                model_final = ARIMA(series_target, order=(1, 1, 1))
                model_final_fitted = model_final.fit()
                predicted_value = max(0, int(model_final_fitted.forecast(steps=1).iloc[0]))
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0

    else:
        # Penanganan darurat jika jumlah baris data historis terlalu sedikit
        y_pred_test = np.zeros(len(y_test)).astype(int)
        predicted_value = 0

# ==============================================================================
# 07. METRICS CALCULATOR & EVALUATION BLOK
# ==============================================================================
if not is_fallback_mode:
    # Inisialisasi variabel metrik performa model
    wmape_score, accuracy_display, mae_val, rmse_val = 0.0, 100.0, 0.0, 0.0
    
    if len(y_test) > 0 and len(y_pred_test) == len(y_test):
        sum_actual = np.sum(y_test)
        # Menghitung Weighted Mean Absolute Percentage Error (WMAPE)
        if sum_actual > 0:
            wmape_score = np.sum(np.abs(y_test - y_pred_test)) / sum_actual
        else:
            wmape_score = 0.0
            
        accuracy_display = max(0.0, (1.0 - wmape_score) * 100.0)
        mae_val = mean_absolute_error(y_test, y_pred_test)
        rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))

# ==============================================================================
# 08. LAYOUT DASHBOARD & VISUALISASI UTAMA
# ==============================================================================
# Judul Aplikasi pada Header Dashboard Utama
st.markdown("<h1 style='margin-top: 0; margin-bottom: 5px; font-weight: 800; color: #FFFFFF;'>📊 E-Commerce Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #8A99AD; font-size:15px; margin-bottom:25px;'>Sistem proyeksi analitik prediktif penjualan. Filter Aktif: <span style='color:#00E5FF; font-weight:600;'>{selected_region}</span></p>", unsafe_allow_html=True)

# KONDISI A: TAMPILAN DASHBOARD UTAMA (MODE ITEM TUNGGAL)
if not is_fallback_mode:
    # Mengonversi format indeks penanggalan untuk keperluan visualisasi grafik sumbu X
    if selected_period == "Weekly":
        chart_string_labels = original_datetime_index.strftime('%Y-W%V').tolist()
    elif selected_period == "Quarterly":
        chart_string_labels = original_datetime_index.strftime('%Y-Q').tolist()
        # Modifikasi teks manual agar mencerminkan nomor kuartal yang akurat
        for q_idx in range(len(chart_string_labels)):
            current_month = original_datetime_index[q_idx].month
            quarter_num = (current_month - 1) // 3 + 1
            chart_string_labels[q_idx] = f"{original_datetime_index[q_idx].year}-Q{quarter_num}"
    else:
        chart_string_labels = original_datetime_index.strftime('%Y-%m').tolist()

    # Formatter label penanggalan untuk titik masa depan fiktif (Forecast)
    last_date_parsed = original_datetime_index[-1]
    if selected_period == "Monthly":
        future_date_obj = last_date_parsed + pd.DateOffset(months=1)
        future_date_string = future_date_obj.strftime('%Y-%m')
    elif selected_period == "Quarterly":
        future_date_obj = last_date_parsed + pd.DateOffset(months=3)
        future_quarter_num = (future_date_obj.month - 1) // 3 + 1
        future_date_string = f"{future_date_obj.year}-Q{future_quarter_num}"
    else:
        future_date_obj = last_date_parsed + pd.Timedelta(weeks=1)
        future_date_string = future_date_obj.strftime('%Y-W%V')

    # Alokasi susunan metrik nilai uji (Model Evaluation) pada baris DataFrame visualisasi
    y_eval_preds = np.full(len(y_target), np.nan)
    if len(y_pred_test) == (len(y_target) - split_index):
        y_eval_preds[split_index:] = y_pred_test

    # Pembentukan struktur tabel tunggal siap render (Single Render Dataframe)
    df_single_render = pd.DataFrame(index=chart_string_labels, columns=['Historical Sales', 'Model Evaluation [20%]', 'Forecast'])
    df_single_render['Historical Sales'] = y_target
    df_single_render['Model Evaluation [20%]'] = y_eval_preds
    
    # Sinkronisasi visual agar pangkal awal garis ramalan menyambung dengan titik akhir historis
    last_historical_label = chart_string_labels[-1]
    df_single_render.loc[last_historical_label, 'Forecast'] = y_target[-1]
    df_single_render.loc[future_date_string] = [np.nan, np.nan, predicted_value]
    
    # Penentuan teks deskriptif penunjuk waktu periode masa depan pada Metric Card
    if selected_period == "Weekly":
        future_period_info = f"Minggu Depan ({future_date_string})"
    elif selected_period == "Quarterly":
        future_period_info = f"Kuartal Depan ({future_date_string})"
    else:
        future_period_info = f"Bulan Depan ({future_date_string})"

    # Pembagian kolom tata letak: Komponen Nilai KPI Utama (Atas)
    col_kpi_1, col_kpi_2, col_kpi_3 = st.columns([1.2, 1.2, 2.1])
    
    with col_kpi_1:
        st.markdown(f"""
            <div class='metric-box' style='border-left-color: #00E5FF;'>
                <div class='metric-label'>Prediksi Penjualan {future_period_info}</div>
                <div class='metric-value'>{predicted_value:,} <span style='font-size:16px; color:#8A99AD; font-weight:400;'>units</span></div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_kpi_2:
        st.markdown(f"""
            <div class='metric-box' style='border-left-color: #00E676;'>
                <div class='metric-label'>Skor Akurasi Model ({selected_method})</div>
                <div class='metric-value'>{accuracy_display:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_kpi_3:
        st.markdown(f"""
            <div class='html-info-box'>
                <table style='width:100%; border-collapse:collapse; color:#FFFFFF; font-size:13px;'>
                    <tr>
                        <td style='padding: 3px 0; color:#8A99AD;'>Weighted MAPE (WMAPE)</td>
                        <td style='text-align:right; font-weight:600; color:#00E676;'>{wmape_score*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td style='padding: 3px 0; color:#8A99AD;'>Mean Absolute Error (MAE)</td>
                        <td style='text-align:right; font-weight:600; color:#FFB300;'>± {mae_val:.0f} qty</td>
                    </tr>
                    <tr>
                        <td style='padding: 3px 0; color:#8A99AD;'>Root Mean Squared Error (RMSE)</td>
                        <td style='text-align:right; font-weight:600; color:#FF5252;'>{rmse_val:.1f}</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # Implementasi penyaringan rentang visualisasi sumbu X berdasarkan input pilihan pengguna
    if selected_range == "Last 80% Data":
        slice_limit = int(len(df_single_render) * 0.2)
        df_single_render = df_single_render.iloc[slice_limit:]
    elif selected_range == "Last 50% Data":
        slice_limit = int(len(df_single_render) * 0.5)
        df_single_render = df_single_render.iloc[slice_limit:]
    elif selected_range == "Last 20% Data":
        slice_limit = int(len(df_single_render) * 0.8)
        df_single_render = df_single_render.iloc[slice_limit:]

    # Transformasi struktur bentuk DataFrame (Melting) agar sesuai dengan format skema Vega-Lite
    df_single_render = df_single_render.reset_index().rename(columns={'index': 'Date'})
    df_melted = df_single_render.melt(id_vars=['Date'], value_vars=['Historical Sales', 'Model Evaluation [20%]', 'Forecast'],
                                      var_name='Status', value_name='Amount').dropna(subset=['Amount'])

    # Pembagian warna garis grafik koordinat
    color_schema = {
        "field": "Status",
        "type": "nominal",
        "scale": {
            "domain": ["Historical Sales", "Model Evaluation [20%]", "Forecast"],
            "range": ["#00E5FF", "#FFB300", "#FF5252"]
        },
        "legend": {
            "orient": "top-left",
            "title": None,
            "labelColor": "#FFFFFF",
            "offset": 10
        }
    }

    # Wadah Visualisasi Grafik Penjualan Utama
    st.markdown("<div class='dashboard-stage'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0; margin-bottom:15px; font-size:16px; color:#FFFFFF;'>📈 Grafik Deret Waktu Penjualan: {selected_item}</h3>", unsafe_allow_html=True)
    
    vega_lite_spec = {
        "width": "container",
        "height": 380,
        "mark": {
            "type": "line", 
            "tooltip": True,
            "interpolate": "linear",
            "point": {"size": 60, "filled": True, "cursor": "pointer"}
        },
        "encoding": {
            "x": {
                "field": "Date",
                "type": "nominal",
                "sort": None,
                "axis": {
                    "labelAngle": -45,
                    "title": None,
                    "labelColor": "#8A99AD",
                    "grid": False
                }
            },
            "y": {
                "field": "Amount",
                "type": "quantitative",
                "axis": {
                    "title": f"Volume Transaksi ({selected_period})",
                    "titleColor": "#8A99AD",
                    "labelColor": "#8A99AD",
                    "grid": True,
                    "gridColor": "#1F2937"
                }
            },
            "color": color_schema
        },
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"}
        }
    }
    
    st.vega_lite_chart(df_melted, vega_lite_spec, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# KONDISI B: TAMPILAN DASHBOARD UTAMA (MODE COMBINED MULTI-ITEM FALLBACK)
else:
    st.markdown("<div class='dashboard-stage'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0; color:#FFB300;'>⚠️ Mode Fallback Komparasi Multi-Item Aktif</h3>", unsafe_allow_html=True)
    st.write(f"Fungsi latih mesin peramalan otomatis dinonaktifkan saat memilih opsi gabungan kelompok guna menghemat beban komputasi server. Grafik di bawah membandingkan total tren volume transaksi antar-item berdasarkan filter wilayah yang Anda tentukan.")
    st.markdown("---")
    
    # Agregasi matriks data kumulatif untuk perbandingan antar kelompok item tunggal
    df_pivot_fallback = dataset_filtered.groupby(['Order Date', target_column])['Quantity'].sum().unstack().fillna(0)
    df_pivot_fallback = df_pivot_fallback.resample(freq_code).sum()
    
    # Konversi label waktu koordinat sumbu X
    if selected_period == "Weekly":
        fallback_labels = df_pivot_fallback.index.strftime('%Y-W%V').tolist()
    elif selected_period == "Quarterly":
        fallback_labels = df_pivot_fallback.index.strftime('%Y-Q').tolist()
        for q_idx in range(len(fallback_labels)):
            current_month = df_pivot_fallback.index[q_idx].month
            quarter_num = (current_month - 1) // 3 + 1
            fallback_labels[q_idx] = f"{df_pivot_fallback.index[q_idx].year}-Q{quarter_num}"
    else:
        fallback_labels = df_pivot_fallback.index.strftime('%Y-%m').tolist()
        
    df_pivot_fallback.index = fallback_labels
    df_pivot_fallback = df_pivot_fallback.reset_index().rename(columns={'index': 'Date'})
    
    # Mengubah bentuk matriks tabel komparasi agar siap dirender oleh Vega-Lite
    df_melted_fallback = df_pivot_fallback.melt(id_vars=['Date'], var_name='Item Group', value_name='Volume')
    
    fallback_chart_spec = {
        "width": "container",
        "height": 400,
        "mark": {"type": "line", "tooltip": True, "point": True},
        "encoding": {
            "x": {
                "field": "Date",
                "type": "nominal",
                "sort": None,
                "axis": {"labelAngle": -45, "labelColor": "#8A99AD"}
            },
            "y": {
                "field": "Volume",
                "type": "quantitative",
                "axis": {"title": "Total Quantity Terjual", "labelColor": "#8A99AD"}
            },
            "color": {
                "field": "Item Group",
                "type": "nominal",
                "legend": {"orient": "top", "title": None, "labelColor": "#FFFFFF"}
            }
        },
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"}
        }
    }
    
    st.vega_lite_chart(df_melted_fallback, fallback_chart_spec, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 09. FOOTER INTERFACE: AUDIT PREVIEW DATA MENTAH (POPUP EXPANDER)
# ==============================================================================
st.markdown("---")
with st.expander("📂 Pratinjau Kumpulan Data Mentah (Raw Dataset Preview)"):
    st.write("Metrik ringkasan deskriptif statistik dari dataset yang tersaring:")
    st.dataframe(dataset_filtered.describe(), use_container_width=True)
    st.write("Menampilkan 50 baris catatan transaksi riil pertama:")
    st.dataframe(dataset_filtered.head(50), use_container_width=True)