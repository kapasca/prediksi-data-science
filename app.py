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

# Suppress logs internal milik Prophet & Stan agar terminal bersih
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==============================================================================
# 01. APPLICATION CONFIGURATION & WEB METADATA
# ==============================================================================
# initial_sidebar_state="expanded" dipasang agar mencegah sidebar hilang total saat reload
st.set_page_config(layout="wide", page_title="E-Commerce Sales Predictions", initial_sidebar_state="expanded")

# ==============================================================================
# 02. VISUAL UI ARCHITECTURE & CUSTOM CSS INJECTION
# ==============================================================================
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
        
        /* Styling khusus teks label di dalam sidebar */
        .sidebar-label {
            font-size: 0.8rem !important;
            color: #B0B0B0 !important;
            margin-bottom: 0.2rem;
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
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 03. DATA PIPELINE - PHASE 1: INGESTION & DATA CLEANING
# ==============================================================================
@st.cache_data
def load_raw_data():
    data = pd.read_csv("dataset.csv")
    # Membersihkan baris yang kehilangan data kritikal untuk pemodelan
    data = data.dropna(subset=['Product Name', 'Order Date', 'Quantity'])
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data = data.dropna(subset=['Order Date'])
    return data

try:
    dataset_raw = load_raw_data()
except Exception as error:
    st.error(f"Failed to read dataset file: {error}")
    st.stop()

product_list = sorted(list(dataset_raw['Product Name'].unique()))

# Inisialisasi session state untuk kontrol penampilan modal dataset
if "show_dataset_modal" not in st.session_state:
    st.session_state.show_dataset_modal = False

# ==============================================================================
# 04. SUB-ROUTINE: DATASET MODAL VIEWER FUNCTION
# ==============================================================================
@st.dialog("RAW Dataset Preview", width="large")
def show_dataset_preview_modal():
    st.markdown("<h3 style='color: #edae3e; text-align: center; margin-top: -1.5rem;'>RAW Dataset Preview</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(dataset_raw))
    with col2:
        st.metric("Total Columns", len(dataset_raw.columns))
    with col3:
        st.metric("Products", len(product_list))
    
    st.markdown("")
    st.markdown("<div style='color: #B0B0B0; font-size: 0.9rem; margin: 1rem 0;'><b>Column Names:</b></div>", unsafe_allow_html=True)
    st.write(dataset_raw.columns.tolist())
    
    st.markdown("")
    st.markdown("<div style='color: #B0B0B0; font-size: 0.85rem; margin: 1rem 0;'><b>Summary Statistics:</b></div>", unsafe_allow_html=True)
    st.dataframe(dataset_raw.describe(), use_container_width=True)
    
    st.markdown("<div style='color: #B0B0B0; font-size: 0.9rem; margin: 1rem 0;'><b>Data Preview (First 50 rows):</b></div>", unsafe_allow_html=True)
    st.dataframe(dataset_raw.head(50), use_container_width=True, height=350)
    
    st.markdown("---")
    # Tombol close di bawah sekarang cukup memicu st.rerun() untuk menutup dialog secara bersih
    if st.button("Close Preview", key="btn_close_dataset", use_container_width=True):
        st.rerun()

# ==============================================================================
# 05. USER INTERFACE CONTROLS (SIDEBAR PANEL)
# ==============================================================================
with st.sidebar:
    # TOP SECTION: Control Panel Title
    st.markdown("<h2 style='font-size: 1.3rem; font-weight: 900; color: #edae3e; margin-top: -1.6rem; margin-bottom: 1.7rem; background-color: #0e1117; text-align: center; border-radius: 10px;'>Control Panel</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-label'>Product Name Filter</div>", unsafe_allow_html=True)
    product_options = ["All Products"] + product_list
    selected_product = st.selectbox("", product_options, index=0, label_visibility="collapsed", key="ctl_product")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-label'>Forecasting Period</div>", unsafe_allow_html=True)
    selected_period = st.selectbox("", ["Monthly", "Quarterly", "Weekly"], index=0, label_visibility="collapsed", key="ctl_period")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-label'>Machine Learning Algorithm</div>", unsafe_allow_html=True)
    algorithm_options = ["Linear Regression", "Moving Average", "XGBoost", "Exponential Smoothing", "Prophet"]
    selected_method = st.selectbox("", algorithm_options, index=0, label_visibility="collapsed", key="ctl_method")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-label'>Show Data</div>", unsafe_allow_html=True)
    range_options = ["All Data", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%"]
    selected_range = st.selectbox("", range_options, index=0, label_visibility="collapsed", key="ctl_range")
    
    # SPACER: Memberikan jarak vertikal yang dinamis ke bawah
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    for _ in range(5):
        st.write()
    
    # BOTTOM SECTION: Dataset Viewer Button (Teks statis, lurus tanpa ribet state)
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    if st.button("View Dataset (Summary)", use_container_width=True, key="btn_view_dataset"):
        show_dataset_preview_modal()

# Konversi String Frekuensi ke Format Datetime Pandas
if selected_period == "Monthly":
    freq_code = 'ME'
elif selected_period == "Quarterly":
    freq_code = 'QE'
else:  # Weekly
    freq_code = 'W'

# Picu penampilan fungsi dialog modal jika status True
if st.session_state.show_dataset_modal:
    show_dataset_preview_modal()

# Judul utama dipasang di area konten utama kanan
st.title("E-Commerce Sales Predictions")
st.write("---")

# Placeholder untuk rendering grafik
placeholder_chart = st.empty()

# ==============================================================================
# 06. DATA PIPELINE - PHASE 2: PROCESSING & MODEL FORECASTING (SINGLE PRODUCT)
# ==============================================================================
if selected_product != "All Products":
    # --------------------------------------------------------------------------
    # SUB-PHASE A: TIME-SERIES RESAMPLING & RE-INDEXING
    # --------------------------------------------------------------------------
    dataset_product = dataset_raw[dataset_raw['Product Name'] == selected_product]
    
    # Membuat timeline penuh untuk menangani 'missing period' menggunakan reindex
    full_timeline = pd.date_range(start=dataset_product['Order Date'].min(), end=dataset_product['Order Date'].max(), freq=freq_code)
    df_interim = dataset_product.resample(freq_code, on='Order Date')['Quantity'].sum()
    dataset_resampled = df_interim.reindex(full_timeline, fill_value=0).to_frame()
    dataset_resampled = dataset_resampled.sort_index()
    
    original_datetime_index = dataset_resampled.index.copy()
    
    # Pemetaan string label sumbu X berdasarkan periode pilihan user
    if selected_period == "Weekly":
        chart_string_labels = dataset_resampled.index.map(lambda dt: f"{dt.strftime('%Y-%m')} (W-{dt.isocalendar()[1]})").tolist()
    elif selected_period == "Quarterly":
        chart_string_labels = dataset_resampled.index.map(lambda dt: f"{dt.strftime('%Y')}-Q{(dt.month-1)//3 + 1}").tolist()
    else:  # Monthly
        chart_string_labels = dataset_resampled.index.strftime('%Y-%m').tolist()

    # Feature Engineering (Untuk Linear Regression & XGBoost supervised learning)
    dataset_resampled['Timestep'] = np.arange(len(dataset_resampled))
    dataset_resampled['Month'] = dataset_resampled.index.month
    dataset_resampled['Year'] = dataset_resampled.index.year
    dataset_resampled['Lag_1'] = dataset_resampled['Quantity'].shift(1).bfill()
    
    if selected_method == "XGBoost":
        feature_cols = ['Timestep', 'Month', 'Year', 'Lag_1']
    else:
        feature_cols = ['Timestep']
        
    X_features = dataset_resampled[feature_cols].values
    y_target = dataset_resampled['Quantity'].values
    
    # Pembagian Data: 80% Training Data dan 20% Testing Data untuk Model Evaluation
    split_index = int(len(dataset_resampled) * 0.8)
    if split_index == 0: split_index = 1
    
    X_train, X_test = X_features[:split_index], X_features[split_index:]
    y_train, y_test = y_target[:split_index], y_target[split_index:]
    
    predicted_value = 0
    mape_error = 0
    mae_val = 0
    rmse_val = 0
    
    y_eval_preds = np.full(len(dataset_resampled), np.nan, dtype=float)
    y_pred_test = np.zeros(len(y_test))
    
    # --------------------------------------------------------------------------
    # SUB-PHASE B: MACHINE LEARNING & STATISTICAL FORECASTING CORE ENGINE
    # --------------------------------------------------------------------------
    if len(dataset_resampled) >= 2:
        
        # --- ALGORITMA 1: LINEAR REGRESSION ---
        if selected_method == "Linear Regression":
            model_evaluator = LinearRegression()
            model_evaluator.fit(X_train, y_train)
            y_pred_test = model_evaluator.predict(X_test)
            y_pred_test = np.maximum(0, y_pred_test.astype(int))
            
            model_final = LinearRegression()
            model_final.fit(X_features, y_target)
            next_timestep = np.array([[len(dataset_resampled)]])
            predicted_value = max(0, int(model_final.predict(next_timestep)[0]))
            
        # --- ALGORITMA 2: MOVING AVERAGE ---
        elif selected_method == "Moving Average":
            rolling_window = min(3, len(y_train))
            y_pred_list = []
            training_history = list(y_train)
            
            for idx in range(len(y_test)):
                window_prediction = np.mean(training_history[-rolling_window:]) if len(training_history) >= rolling_window else 0
                y_pred_list.append(window_prediction)
                training_history.append(y_test[idx])
            
            y_pred_test = np.array(y_pred_list).astype(int)
            predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= rolling_window else 0

        # --- ALGORITMA 3: XGBOOST ---
        elif selected_method == "XGBoost":
            if len(dataset_resampled) < 5:
                rolling_window = min(3, len(y_train))
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-rolling_window:])) if len(y_target) >= rolling_window else 0
            else:
                model_evaluator = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_evaluator.fit(X_train, y_train)
                y_pred_test = model_evaluator.predict(X_test)
                y_pred_test = np.maximum(0, y_pred_test.astype(int))
                
                model_final = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.1)
                model_final.fit(X_features, y_target)
                
                last_date_parsed = original_datetime_index[-1]
                if selected_period == "Monthly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=1)
                elif selected_period == "Quarterly":
                    future_date_obj = last_date_parsed + pd.DateOffset(months=3)
                else:  # Weekly
                    future_date_obj = last_date_parsed + pd.Timedelta(weeks=1)
                
                X_future_step = np.array([[len(dataset_resampled), future_date_obj.month, future_date_obj.year, y_target[-1]]])
                predicted_value = max(0, int(model_final.predict(X_future_step)[0]))

        # --- ALGORITMA 4: EXPONENTIAL SMOOTHING ---
        elif selected_method == "Exponential Smoothing":
            try:
                model_evaluator = ExponentialSmoothing(y_train, initialization_method="estimated").fit()
                y_pred_test = model_evaluator.forecast(len(y_test))
                y_pred_test = np.maximum(0, y_pred_test.astype(int))
                
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
                m_eval = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                m_eval.fit(df_train)
                
                future_eval = m_eval.make_future_dataframe(periods=len(y_test), freq=freq_code)
                forecast_eval = m_eval.predict(future_eval)
                y_pred_test = forecast_eval['yhat'].iloc[split_index:].values
                y_pred_test = np.maximum(0, y_pred_test.astype(int))
                
                m_final = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                m_final.fit(df_prophet)
                future_final = m_final.make_future_dataframe(periods=1, freq=freq_code)
                forecast_final = m_final.predict(future_final)
                predicted_value = max(0, int(forecast_final['yhat'].iloc[-1]))
            except:
                y_pred_test = np.full(len(y_test), np.mean(y_train)).astype(int)
                predicted_value = int(np.mean(y_target[-2:])) if len(y_target) >= 2 else 0

        # Taruh hasil prediksi test ke dalam array render visual
        y_eval_preds[split_index:] = y_pred_test

        # Perhitungan Metrik Evaluasi (WMAPE, MAE, RMSE)
        sum_actual = np.sum(y_test)
        if sum_actual > 0:
            mape_error = (np.sum(np.abs(y_test - y_pred_test)) / sum_actual) * 100
        else:
            mape_error = 100.0 if np.sum(y_pred_test) > 0 else 0.0
            
        mae_val = mean_absolute_error(y_test, y_pred_test)
        rmse_val = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # --------------------------------------------------------------------------
    # SUB-PHASE C: KPI METRICS & DATAFRAME RENDERING PACKING
    # --------------------------------------------------------------------------
    accuracy_score = max(0.0, 100.0 - mape_error)
    if selected_period == "Monthly":
        periode_label = "Next Month"
    elif selected_period == "Quarterly":
        periode_label = "Next Quarter"
    else:  # Weekly
        periode_label = "Next Week"
    
    html_title_box = f"""
    <div style='text-align: left; line-height: 1.2;'>
        <div style='font-size: 0.9rem; color: #E0E0E0; font-weight: 500;'>Sales Statistics</div>
        <div style='font-size: 1.6rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;'>{selected_product}</div>
        <div style='font-size: 0.75rem; color: #B0B0B0;'>{selected_period} with <span style='color: #00FFA6; font-weight: 600;'>{selected_method}</span></div>
    </div>
    """
    
    html_info_box = f"""
    <div style='text-align: right; line-height: 1.3;'>
        <div style='font-size: 0.85rem; color: #E0E0E0; font-weight: 500;'>{periode_label} Prediction</div>
        <div style='font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin: 1px 0;'>{predicted_value:,} Qty</div>
        <div style='font-size: 0.72rem; color: #B0B0B0;'>
            <span style='color: #FFFFFF; font-weight: 600;'>[Accuracy Score]</span> 
            WMAPE: <span style='color: #00FFA6; font-weight: 600;'>{accuracy_score:.1f}%</span><br/>
            MAE: <span style='color: #FFB300; font-weight: 600;'>&plusmn; {mae_val:.0f} qty</span> | 
            RMSE: <span style='color: #FF6B6B; font-weight: 600;'>&plusmn; {rmse_val:.0f} qty</span>
        </div>
    </div>
    """
    
    last_date_parsed = original_datetime_index[-1]
    if selected_period == "Monthly":
        future_date_string = (last_date_parsed + pd.DateOffset(months=1)).strftime('%Y-%m')
    elif selected_period == "Quarterly":
        future_date = last_date_parsed + pd.DateOffset(months=3)
        future_date_string = f"{future_date.strftime('%Y')}-Q{(future_date.month-1)//3 + 1}"
    else:  # Weekly
        future_date = last_date_parsed + pd.Timedelta(weeks=1)
        future_date_string = f"{future_date.strftime('%Y-%m')} (W-{future_date.isocalendar()[1]})"

    df_single_render = pd.DataFrame(index=chart_string_labels, columns=['Historical Sales', 'Model Evaluation [20%]', 'Forecast'])
    df_single_render['Historical Sales'] = y_target
    df_single_render['Model Evaluation [20%]'] = y_eval_preds
    
    last_historical_label = chart_string_labels[-1]
    df_single_render.loc[last_historical_label, 'Forecast'] = y_target[-1]
    df_single_render.loc[future_date_string] = [np.nan, np.nan, predicted_value]
    
    if selected_range != "All Data":
        pct = int(selected_range.replace("%", "")) / 100.0
        keep_points = max(2, int(len(df_single_render) * pct))
        df_single_render = df_single_render.iloc[-keep_points:]
        
    df_single_render = df_single_render.replace([np.inf, -np.inf], np.nan)
    df_melted = df_single_render.reset_index().rename(columns={'index': 'Date'}).melt('Date', var_name='Category', value_name='Amount')
    
    color_schema = {
        "field": "Category", 
        "type": "nominal", 
        "scale": {"domain": ['Historical Sales', 'Model Evaluation [20%]', 'Forecast'], "range": ["#4A90E2", "#F5A623", "#FF4B4B"]},
        "legend": None
    }
    
    html_custom_legend = """
    <div class='custom-legend-container'>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #4A90E2;'></div>Historical Sales</div>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #F5A623;'></div>Model Evaluation [20%]</div>
        <div class='legend-item'><div class='legend-color-box' style='background-color: #FF4B4B;'></div>Forecast</div>
    </div>
    """

else:
    # ==============================================================================
    # 07. DATA PIPELINE - ALL PRODUCTS MODE (FALLBACK MULTI LINE)
    # ==============================================================================
    html_title_box = """
    <div style='text-align: left; line-height: 1.2;'>
        <div style='font-size: 0.9rem; color: #E0E0E0; font-weight: 500;'>Sales Statistics</div>
        <div style='font-size: 1.6rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;'>All Products</div>
        <div style='font-size: 0.75rem; color: #B0B0B0;'>Source: <span style='color: #00FFA6; font-weight: 600;'>dataset.csv</span></div>
    </div>
    """
    html_info_box = ""
    
    global_timeline = dataset_raw.resample(freq_code, on='Order Date')['Quantity'].sum().index
    if selected_period == "Weekly":
        fallback_labels = global_timeline.map(lambda dt: f"{dt.strftime('%Y-%m')} (W-{dt.isocalendar()[1]})").tolist()
    elif selected_period == "Quarterly":
        fallback_labels = global_timeline.map(lambda dt: f"{dt.strftime('%Y')}-Q{(dt.month-1)//3 + 1}").tolist()
    else:  # Monthly
        fallback_labels = global_timeline.strftime('%Y-%m').tolist()
        
    df_all_render = pd.DataFrame(index=fallback_labels)
    
    for product_item in product_list:
        dataset_p_filter = dataset_raw[dataset_raw['Product Name'] == product_item]
        dataset_p_res = dataset_p_filter.resample(freq_code, on='Order Date')['Quantity'].sum().reindex(global_timeline, fill_value=0)
        df_all_render[product_item] = dataset_p_res.values
        
    if selected_range != "All Data":
        pct = int(selected_range.replace("%", "")) / 100.0
        keep_points = max(2, int(len(df_all_render) * pct))
        df_all_render = df_all_render.iloc[-keep_points:]
        
    df_melted = df_all_render.reset_index().rename(columns={'index': 'Date'}).melt('Date', var_name='Category', value_name='Amount')
    
    color_schema = {
        "field": "Category", 
        "type": "nominal",
        "legend": None
    }
    
    html_custom_legend = "<div class='custom-legend-container' style='flex-wrap: wrap; max-width: 80%; margin: 0 auto; gap: 15px;'>"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, product_item in enumerate(product_list):
        c = colors[i % len(colors)]
        html_custom_legend += f"<div class='legend-item'><div class='legend-color-box' style='background-color: {c}; height: 10px; width: 10px; border-radius: 50%;'></div>{product_item}</div>"
    html_custom_legend += "</div>"

# ==============================================================================
# 08. DYNAMIC PLACEHOLDER INJECTION & ENGINE GRAPHICS RENDERING
# ==============================================================================
with placeholder_chart.container():
    col_title, col_info = st.columns([1, 1])
    
    with col_title:
        st.markdown(html_title_box, unsafe_allow_html=True)
        
    with col_info:
        if html_info_box:
            st.markdown(html_info_box, unsafe_allow_html=True)
            
    st.markdown(html_custom_legend, unsafe_allow_html=True)
            
    vega_lite_spec = {
        "width": "container",
        "height": "container",
        "mark": {
            "type": "line", 
            "tooltip": True, 
            "interpolate": "linear",
            "point": {"size": 60, "filled": True, "cursor": "pointer"}
        },
        "encoding": {
            "x": {"field": "Date", "type": "nominal", "sort": None, "axis": {"labelAngle": -90, "title": None}},
            "y": {"field": "Amount", "type": "quantitative", "axis": {"title": "Transaction Volume"}},
            "color": color_schema
        },
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"}
        }
    }
    
    st.vega_lite_chart(df_melted, vega_lite_spec, use_container_width=True)