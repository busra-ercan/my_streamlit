# file: streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from typing import Optional, Tuple

# ===================== Constants =====================
CURRENT_YEAR = 2025
NUM_COLS = [
    "CAE","log_mileage","km_per_year","engineSize","mpg","tax",
    "mileage_age_interaction","high_mileage","large_engine","auto_large_engine"
]
CAT_COLS = ["transmission","fuelType"]
ALL_X_COLS = NUM_COLS + CAT_COLS
GLOBAL_MAE_GBP = 1740.0  # Practical MAE band

REQUIRED_BASE = {"year", "mileage"}
BRAND_ALIASES = ["brand", "make", "manufacturer"]
MODEL_ALIASES = ["model", "model_name", "series"]

# ===================== Page =====================
st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Estimator")

# ===================== Utils =====================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def auto_map_brand_model(df: pd.DataFrame) -> pd.DataFrame:
    """Why: unify heterogeneous column names to 'brand' and 'model' for reliable UI."""
    df = normalize_cols(df)
    if "brand" not in df.columns:
        for c in ("make","manufacturer"):
            if c in df.columns:
                df.rename(columns={c: "brand"}, inplace=True)
                break
    if "model" not in df.columns:
        for c in ("model_name","series"):
            if c in df.columns:
                df.rename(columns={c: "model"}, inplace=True)
                break
    return df

def detect_brand_model_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = set(df.columns)
    brand_col = next((c for c in BRAND_ALIASES if c in cols), None)
    model_col = next((c for c in MODEL_ALIASES if c in cols), None)
    return brand_col, model_col

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None, path: Optional[str]=None) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file is not None:
            return auto_map_brand_model(pd.read_csv(uploaded_file))
        if path and os.path.exists(path):
            return auto_map_brand_model(pd.read_csv(path))
        return None
    except Exception as exc:
        st.warning(f"Dataset load failed: {exc}")
        return None

@st.cache_resource(show_spinner=False)
def load_pipe(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load(path)

def fe_transform_single(
    year: int, mileage: float, engine_size: float,
    transmission: str, fuel_type: str,
    mpg: Optional[float]=None, tax: Optional[float]=None,
    model_name: Optional[str]=None
) -> pd.DataFrame:
    """Why: features must mirror training for valid inference."""
    df = pd.DataFrame([{
        "year": year, "mileage": mileage, "engineSize": engine_size,
        "transmission": transmission, "fuelType": fuel_type,
        "mpg": np.nan if mpg is None else mpg,
        "tax": np.nan if tax is None else tax,
        "model": model_name or ""
    }])
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
            X[c] = np.nan  # rely on pipeline imputer/encoder
    return X[ALL_X_COLS]

def predict_mid(pipe_mid, X_one: pd.DataFrame) -> float:
    y_log = pipe_mid.predict(X_one)
    return float(np.expm1(y_log)[0])

def predict_interval(pipe_lo, pipe_hi, X_one: pd.DataFrame):
    if pipe_lo is None or pipe_hi is None:
        return None
    lo = float(pipe_lo.predict(X_one)[0]); hi = float(pipe_hi.predict(X_one)[0])
    return (lo, hi) if lo <= hi else (hi, lo)

def suggest_mileage_range(df: Optional[pd.DataFrame], brand: Optional[str], model: Optional[str], sel_year: int) -> Tuple[int,int]:
    """Why: make KM chart realistic using km/year distribution from chosen slice."""
    fallback = (20_000, 120_000)
    if df is None or "mileage" not in df.columns or "year" not in df.columns:
        return fallback
    tmp = df.copy()
    tmp = tmp[(pd.to_numeric(tmp["year"], errors="coerce") >= 1980) & (pd.to_numeric(tmp["year"], errors="coerce") <= CURRENT_YEAR)]
    tmp["vehicle_age"] = CURRENT_YEAR - pd.to_numeric(tmp["year"], errors="coerce")
    tmp = tmp[tmp["vehicle_age"] >= 0]
    denom = (tmp["vehicle_age"] + 1).replace(0, 1)
    tmp["km_per_year_calc"] = pd.to_numeric(tmp["mileage"], errors="coerce") / denom

    brand_col, model_col = detect_brand_model_cols(tmp)
    if brand and brand_col and brand_col in tmp.columns:
        tmp = tmp[tmp[brand_col].astype(str) == str(brand)]
    if model and model_col and model_col in tmp.columns:
        tmp = tmp[tmp[model_col].astype(str) == str(model)]

    valid = tmp["km_per_year_calc"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.shape[0] < 10:
        return fallback

    q10 = float(valid.quantile(0.10))
    q90 = float(valid.quantile(0.90))
    if not np.isfinite(q10) or not np.isfinite(q90) or q10 <= 0 or q90 <= 0:
        return fallback

    vehicle_age = max(0, CURRENT_YEAR - int(sel_year))
    lo = int(max(0, round(q10 * (vehicle_age + 1), -3)))
    hi = int(max(lo + 1_000, round(q90 * (vehicle_age + 1), -3)))
    return (lo, hi)

def ood_warning_for_mileage(sel_mileage: float, suggested: Tuple[int,int]):
    lo, hi = suggested
    margin = 0.25 * (hi - lo)
    if sel_mileage < lo - margin or sel_mileage > hi + margin:
        st.warning("Mileage seems outside the typical range for this brand/model/year. Predictions may be less reliable.")

# ===================== Sidebar =====================
with st.sidebar:
    st.header("⚙️ Pipelines")
    mid_path = st.text_input("Mid (log→£) pipeline", value="best_lightgbm_optuna.joblib")
    q10_path = st.text_input("Lower Quantile (optional)", value="pipe_q10.joblib")
    q90_path = st.text_input("Upper Quantile (optional)", value="pipe_q90.joblib")
    c1, c2 = st.columns(2)
    if c1.button("Load Pipelines"):
        try:
            st.session_state["pipe_mid"] = load_pipe(mid_path)
            st.session_state["pipe_q10"] = load_pipe(q10_path) if os.path.exists(q10_path) else None
            st.session_state["pipe_q90"] = load_pipe(q90_path) if os.path.exists(q90_path) else None
            st.success("Pipelines loaded.")
        except Exception as e:
            st.error(str(e))
    if c2.button("Reset Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()
    st.header("📂 Dataset (required)")
    st.caption("Upload a single merged CSV that includes: year, mileage, and brand + model columns.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    ds_path = st.text_input("...or local CSV path (optional)", value="")
    d1, d2 = st.columns(2)
    if d1.button("Load Dataset", use_container_width=True):
        df_loaded = load_dataset(uploaded_file=up, path=ds_path)
        if df_loaded is not None:
            st.session_state["df"] = df_loaded
            st.success(f"Dataset loaded: {len(df_loaded):,} rows.")
        else:
            st.error("Failed to load dataset. Check file/path.")
    if d2.button("Clear Dataset", use_container_width=True):
        st.session_state.pop("df", None)
        st.info("Dataset cleared.")

# Try autoload mid model once (optional)
if "pipe_mid" not in st.session_state:
    try:
        st.session_state["pipe_mid"] = load_pipe("best_lightgbm_optuna.joblib")
    except Exception as e:
        st.warning(str(e))

pipe_mid = st.session_state.get("pipe_mid")
pipe_q10 = st.session_state.get("pipe_q10")
pipe_q90 = st.session_state.get("pipe_q90")
df_global = st.session_state.get("df")

# ===================== Require dataset & columns =====================
if not isinstance(df_global, pd.DataFrame):
    st.error("Please load a dataset to continue. Brand/Model must come from your dataset.")
    st.stop()

df_global = auto_map_brand_model(df_global)
missing = [c for c in ("brand","model","year","mileage") if c not in df_global.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Please fix your CSV or mapping.")
    st.stop()

# ===================== Inputs (English) =====================
st.subheader("📝 Inputs")
st.markdown("**Vehicle Identification**")

brands = sorted(df_global["brand"].dropna().astype(str).unique().tolist())
if not brands:
    st.error("No brands found in the dataset."); st.stop()
brand_val = st.selectbox("Brand", options=brands, index=0, key="brand_sel")

models = (
    df_global.loc[df_global["brand"].astype(str) == str(brand_val), "model"]
    .dropna().astype(str).unique().tolist()
)
models = sorted(models)
if not models:
    st.error("No models found for the selected brand."); st.stop()
model_name = st.selectbox("Model", options=models, index=0, key="model_sel")

c1, c2 = st.columns(2)
with c1:
    year = st.number_input("Year", min_value=1980, max_value=CURRENT_YEAR, value=2018, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=60_000, step=1)
    engine_size = st.number_input("Engine Size (L)", min_value=0.6, max_value=6.0, value=1.6, step=0.1)
with c2:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
    mpg_opt = st.number_input("Miles per gallon (MPG)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, help="MPG = Miles per gallon")
    tax_opt = st.number_input("Tax (optional, £ per year)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

mpg_val = None if mpg_opt == 0.0 else mpg_opt
tax_val = None if tax_opt == 0.0 else tax_opt

# Suggested KM window + OOD notice
suggested_km = suggest_mileage_range(df_global, brand_val, model_name, year)
ood_warning_for_mileage(mileage, suggested_km)

# ===================== Predict =====================
if st.button("💡 Predict Price", use_container_width=True):
    if pipe_mid is None:
        st.error("Mid pipeline is not loaded.")
    else:
        X_one = fe_transform_single(year, mileage, engine_size, transmission, fuel_type, mpg_val, tax_val, model_name)
        try:
            mid = predict_mid(pipe_mid, X_one)
            st.success(f"Estimated Price: **£{mid:,.0f}**")

            k = st.slider("MAE Scale (band width)", 0.5, 2.0, 1.0, 0.1)
            mae_band = k * GLOBAL_MAE_GBP
            lower_mae = max(0.0, mid - mae_band); upper_mae = mid + mae_band
            st.info(f"Practical Range: **£{lower_mae:,.0f} – £{upper_mae:,.0f}** (±£{mae_band:,.0f})")

            qi = predict_interval(pipe_q10, pipe_q90, X_one)
            if qi is not None:
                lo, hi = qi
                mid_adj = float(np.clip(mid, lo, hi))  # keep inside quantile band
                if abs(mid_adj - mid) > 1e-6:
                    st.caption(f"Adjusted to stay within quantile band: £{mid:,.0f} → £{mid_adj:,.0f}")

            df_out = pd.DataFrame({
                "Metric": ["Recommended Estimate", "Likely Lower Bound", "Likely Upper Bound"],
                "£": [round(mid), round(lower_mae), round(upper_mae)]
            })
            st.dataframe(df_out, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.divider()
st.subheader("📈 Scenario Analysis")

tab_km, tab_year, tab_engine = st.tabs(["Mileage (KM)", "Year", "Engine Size"])

with tab_km:
    lo_km, hi_km = suggested_km
    km_min, km_max = st.slider("KM range", 0, 300_000, (int(lo_km), int(hi_km)), step=5_000)
    n_pts = st.selectbox("Number of points (KM)", [5,6,7,8,9,10], index=5)
    if st.button("Render KM Chart", use_container_width=True):
        if pipe_mid is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.linspace(km_min, km_max, n_pts, dtype=int)
            preds = [predict_mid(pipe_mid, fe_transform_single(year, km, engine_size, transmission, fuel_type, mpg_val, tax_val, model_name)) for km in grid]
            df_plot = pd.DataFrame({"mileage": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("mileage"))
            st.caption("Effect of mileage with all other inputs fixed. Defaults inferred from your dataset (10–90% km/year for slice).")

with tab_year:
    y_min, y_max = st.slider("Year range", 1980, CURRENT_YEAR, (2008, CURRENT_YEAR), step=1)
    n_pts_y = st.selectbox("Number of points (Year)", [5,6,7,8,9,10], index=5)
    if st.button("Render Year Chart", use_container_width=True):
        if pipe_mid is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.linspace(y_min, y_max, n_pts_y, dtype=int)
            preds = [predict_mid(pipe_mid, fe_transform_single(yv, mileage, engine_size, transmission, fuel_type, mpg_val, tax_val, model_name)) for yv in grid]
            df_plot = pd.DataFrame({"year": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("year"))
            st.caption("Effect of year with all other inputs fixed.")

with tab_engine:
    e_min, e_max = st.slider("Engine Size (L) range", 0.6, 4.0, (1.0, 2.5), step=0.1)
    n_pts_e = st.selectbox("Number of points (Engine)", [5,6,7,8,9,10], index=5)
    if st.button("Render Engine Chart", use_container_width=True):
        if pipe_mid is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.round(np.linspace(e_min, e_max, n_pts_e), 2)
            preds = [predict_mid(pipe_mid, fe_transform_single(year, mileage, float(ev), transmission, fuel_type, mpg_val, tax_val, model_name)) for ev in grid]
            df_plot = pd.DataFrame({"engineSize": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("engineSize"))
            st.caption("Effect of engine size with all other inputs fixed.")

st.markdown(
    """
**Notes**
- The estimate reflects average behavior learned from historical data.
- Based on historical data from vehicles registered in the United Kingdom between 2000 and 2020.
"""
)

# file: requirements.txt
# streamlit>=1.31
# pandas>=2.0
# numpy>=1.24
# joblib>=1.3
# scikit-learn>=1.3
# lightgbm>=4.0
# category-encoders>=2.6
# scipy>=1.10
