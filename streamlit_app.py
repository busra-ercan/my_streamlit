
# streamlit_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
@st.cache_resource
def load_model():
    return load("best_lightgbm_optuna.joblib")

model = load_model()

# ========= Sabitler (eğitimle birebir aynı olmalı) =========
CURRENT_YEAR = 2025
NUM_COLS = [
    "CAE","log_mileage","km_per_year","engineSize","mpg","tax",
    "mileage_age_interaction","high_mileage","large_engine","auto_large_engine"
]
CAT_COLS = ["transmission","fuelType"]
ALL_X_COLS = NUM_COLS + CAT_COLS

# Test setinden gelen tipik hata için MAE bandı (pratik aralık)
GLOBAL_MAE_GBP = 1740.0  # ihtiyaç olursa güncelle

# ========= Yardımcılar =========
def fe_transform_single(year: int,
                        mileage: float,
                        engine_size: float,
                        transmission: str,
                        fuel_type: str,
                        mpg: float | None = None,
                        tax: float | None = None,
                        model_name: str | None = None) -> pd.DataFrame:
    """Tek girdiyi eğitimdekiyle aynı FE'den geçirip X matrisi üretir."""
    row = {
        "year": year, "mileage": mileage, "engineSize": engine_size,
        "transmission": transmission, "fuelType": fuel_type,
        "mpg": np.nan if mpg is None else mpg,
        "tax": np.nan if tax is None else tax,
        "model": model_name or ""
    }
    df = pd.DataFrame([row])

    # Feature engineering (eğitimle aynı sıra)
    df["vehicle_age"] = CURRENT_YEAR - df["year"]
    df["CAE"] = np.log1p(df["vehicle_age"])
    df["log_mileage"] = np.log1p(df["mileage"])
    df["km_per_year"] = df["mileage"] / (df["vehicle_age"] + 1)
    df["mileage_age_interaction"] = df["log_mileage"] * df["CAE"]
    df["high_mileage"] = (df["mileage"] > 100_000).astype(int)
    df["large_engine"] = (df["engineSize"] >= 2.0).astype(int)
    df["auto_large_engine"] = ((df["transmission"] == "Automatic") & (df["engineSize"] >= 2.0)).astype(int)

    X = df[[c for c in ALL_X_COLS if c in df.columns]].copy()
    for c in ALL_X_COLS:
        if c not in X.columns:
            X[c] = np.nan  # pipeline imputer doldurur
    return X[ALL_X_COLS]

@st.cache_resource(show_spinner=False)
def load_pipe(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    return load(path)

def predict_mid(pipe_mid, X_one: pd.DataFrame) -> float:
    """Orta tahmin: log_price -> expm1 -> £"""
    y_log = pipe_mid.predict(X_one)
    return float(np.expm1(y_log)[0])

def predict_interval(pipe_lo, pipe_hi, X_one: pd.DataFrame):
    """Quantile aralığı: doğrudan £ hedefinden [lo, hi]."""
    if pipe_lo is None or pipe_hi is None:
        return None
    lo = float(pipe_lo.predict(X_one)[0])
    hi = float(pipe_hi.predict(X_one)[0])
    return (lo, hi) if lo <= hi else (hi, lo)

# ========= UI =========
st.set_page_config(page_title="Araç Fiyat Tahmini", page_icon="🚗", layout="centered")
st.title("🚗 Araç Fiyat Tahmini")

with st.sidebar:
    st.header("⚙️ Modeller")
    mid_path = st.text_input("Orta (log->£) boru hattı",value="best_lightgbm_optuna.joblib")

    q10_path = st.text_input("Alt Limit (opsiyonel)", value="pipe_q10.joblib")
    q90_path = st.text_input("Üst Limit (opsiyonel)", value="pipe_q90.joblib")

    if st.button("Modelleri Yükle"):
        try:
            st.session_state["pipe_mid"] = load_pipe(mid_path)
            st.session_state["pipe_q10"] = load_pipe(q10_path) if os.path.exists(q10_path) else None
            st.session_state["pipe_q90"] = load_pipe(q90_path) if os.path.exists(q90_path) else None
            st.success("Modeller yüklendi.")
        except Exception as e:
            st.error(str(e))

# İlk açılışta orta modeli otomatik dene
if "pipe_mid" not in st.session_state:
    try:
        st.session_state["pipe_mid"] = load_pipe("best_lightgbm_optuna.joblib")
    except Exception as e:
        st.warning(str(e))

pipe_mid = st.session_state.get("pipe_mid", None)
pipe_q10 = st.session_state.get("pipe_q10", None)
pipe_q90 = st.session_state.get("pipe_q90", None)

st.subheader("📝 Girdiler")
c1, c2 = st.columns(2)
with c1:
    model_name = st.text_input("Model (serbest metin)", "Focus")
    year = st.number_input("Yıl", min_value=1980, max_value=CURRENT_YEAR, value=2018, step=1)
    mileage = st.number_input("Kilometre (km)", min_value=0, max_value=1_000_000, value=60_000, step=1)
    engine_size = st.number_input("Motor Hacmi (L)", min_value=0.6, max_value=6.0, value=1.6, step=0.1)
with c2:
    transmission = st.selectbox("Vites", ["Manual", "Automatic", "Semi-Auto"])
    fuel_type = st.selectbox("Yakıt", ["Petrol", "Diesel", "Hybrid"])
    mpg_opt = st.number_input("MPG (opsiyonel)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    tax_opt = st.number_input("Tax (opsiyonel, £)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

mpg_val = None if mpg_opt == 0.0 else mpg_opt
tax_val = None if tax_opt == 0.0 else tax_opt

# ---- Tek tahmin + MAE bandı + kuantil aralığı ----
if st.button("💡 Tahmini Fiyat", use_container_width=True):
    if pipe_mid is None:
        st.error("Orta model yüklenmedi.")
    else:
        X_one = fe_transform_single(
            year=year, mileage=mileage, engine_size=engine_size,
            transmission=transmission, fuel_type=fuel_type,
            mpg=mpg_val, tax=tax_val, model_name=model_name
        )
        try:
            mid = predict_mid(pipe_mid, X_one)
            st.success(f"Tahmini Fiyat: **£{mid:,.0f}**")

            # MAE bandı (pratik aralık)
            k = st.slider("MAE Katsayısı (aralık genişliği)", 0.5, 2.0, 1.0, 0.1)
            mae_band = k * GLOBAL_MAE_GBP
            lower_mae = max(0.0, mid - mae_band)
            upper_mae = mid + mae_band
            st.info(f"Fiyat Aralıgı: **£{lower_mae:,.0f} – £{upper_mae:,.0f}** (±£{mae_band:,.0f})")

            # Kuantil aralığı (varsa)
            qi = predict_interval(pipe_q10, pipe_q90, X_one)
            if qi is not None:
                lo, hi = qi
                # orta tahmini kuantil bandına projekte et (garanti içeride kalsın)
                mid_adj = float(np.clip(mid, lo, hi))
                #st.warning(f"Aracların Yuzde kacı bu degerin altında (≈%80): **£{lo:,.0f} – £{hi:,.0f}**")
                if abs(mid_adj - mid) > 1e-6:
                    st.caption(f"Not: Beklenen sonuç, en olumsuz ve en iyimser senaryolar arasında kalan güven aralığına yerleştirildi.: £{mid:,.0f} → £{mid_adj:,.0f}")
                    mid = mid_adj

            # Mini özet tablo
            df_out = pd.DataFrame({
                "Metrik": ["Önerilen Fiyat Tahmini", "Olası En düşük Değer", "Olası En Yüksek Değer"],
                "£": [round(mid), round(lower_mae), round(upper_mae)]
            })
            st.dataframe(df_out, use_container_width=True)

        except Exception as e:
            st.error(f"Tahmin hatası: {e}")

st.divider()

# ========= Senaryo Analizi: KM / Yıl / Motor =========
st.subheader("📈 Senaryo Analizi")
tab_km, tab_year, tab_engine = st.tabs(["KM", "Yıl", "Motor Hacmi"])

with tab_km:
    km_min, km_max = st.slider("KM aralığı", 0, 300_000, (20_000, 120_000), step=5_000)
    n_points_km = st.selectbox("Nokta sayısı (KM)", [5,6,7,8,9,10], index=5)
    if st.button("KM Grafiği", use_container_width=True):
        if pipe_mid is None:
            st.error("Önce orta modeli yükleyin.")
        else:
            grid = np.linspace(km_min, km_max, n_points_km, dtype=int)
            preds = []
            for km in grid:
                Xg = fe_transform_single(year, km, engine_size, transmission, fuel_type, mpg_val, tax_val, model_name)
                preds.append(predict_mid(pipe_mid, Xg))
            df_plot = pd.DataFrame({"mileage": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("mileage"))
            st.caption("Diğer değişkenler sabit tutularak KM etkisi.")

with tab_year:
    y_min, y_max = st.slider("Yıl aralığı", 1980, CURRENT_YEAR, (2008, CURRENT_YEAR), step=1)
    n_points_year = st.selectbox("Nokta sayısı (Yıl)", [5,6,7,8,9,10], index=5)
    if st.button("Yıl Grafiği", use_container_width=True):
        if pipe_mid is None:
            st.error("Önce orta modeli yükleyin.")
        else:
            grid = np.linspace(y_min, y_max, n_points_year, dtype=int)
            preds = []
            for yv in grid:
                Xg = fe_transform_single(yv, mileage, engine_size, transmission, fuel_type, mpg_val, tax_val, model_name)
                preds.append(predict_mid(pipe_mid, Xg))
            df_plot = pd.DataFrame({"year": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("year"))
            st.caption("Diğer değişkenler sabit tutularak Yıl etkisi.")

with tab_engine:
    e_min, e_max = st.slider("Motor Hacmi (L) aralığı", 0.6, 4.0, (1.0, 2.5), step=0.1)
    n_points_eng = st.selectbox("Nokta sayısı (Motor)", [5,6,7,8,9,10], index=5)
    if st.button("Motor Grafiği", use_container_width=True):
        if pipe_mid is None:
            st.error("Önce orta modeli yükleyin.")
        else:
            grid = np.round(np.linspace(e_min, e_max, n_points_eng), 2)
            preds = []
            for ev in grid:
                Xg = fe_transform_single(year, mileage, float(ev), transmission, fuel_type, mpg_val, tax_val, model_name)
                preds.append(predict_mid(pipe_mid, Xg))
            df_plot = pd.DataFrame({"engineSize": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("engineSize"))
            st.caption("Diğer değişkenler sabit tutularak Motor hacmi etkisi.")

st.markdown(
    """
    **Açıklama**  
    -Beklenen fiyat, geçmiş verilerden öğrenilen ortalama davranışı temsil eder.
    -Alt ve üst değerler, fiyatın makul şekilde sapabileceği sınırları gösterir.
    -Tahminler, Birleşik Krallık’ta 1980–2023 yılları arasında trafiğe çıkan araçlardan elde edilen geçmiş verilere dayanmaktadır. 
    """
)
