# file: streamlit_app.py
# Light theme app with GLOBAL dataset autoload (path/url/kaggle fallback) so all users
# see a preloaded dataset without the "Please load a dataset" warning.

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ===================== Page =====================
st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Estimator")

# ===================== Admin/Deploy Settings (EDIT THESE) =====================
# 1) Put a merged CSV into your repo at this path (recommended)
DEFAULT_DATA_PATH = "data/merged_used_cars.csv"  # e.g., commit your CSV here

# 2) Or host it somewhere public & stable (GitHub raw / S3 / GCS)
DEFAULT_DATA_URL = ""  # e.g., "https://raw.githubusercontent.com/you/repo/main/merged_used_cars.csv"

# 3) Or a Kaggle dataset (requires Kaggle creds in st.secrets or env)
DEFAULT_KAGGLE = {
    "slug": "",                 # e.g., "yourname/merged-used-cars"
    "file": "merged_used_cars.csv"
}

# Persist to disk so manual uploads survive reruns on the same server instance
PERSIST_PATH = "last_dataset.pkl"

# ===================== Model/Features =====================
CURRENT_YEAR = 2025
NUM_COLS = [
    "CAE","log_mileage","km_per_year","engineSize","mpg","tax",
    "mileage_age_interaction","high_mileage","large_engine","auto_large_engine"
]
CAT_COLS = ["transmission","fuelType"]
ALL_X_COLS = NUM_COLS + CAT_COLS
GLOBAL_MAE_GBP = 1740.0

BRAND_KEYWORDS = {
    "audi": [" audi","a1","a3","a4","a5","a6","q2","q3","q5","tt"],
    "bmw": [" bmw","1 series","2 series","3 series","4 series","5 series","x1","x3","x5"],
    "mercedes": [" mercedes"," merc ","c class","e class","a class","b class","cla","gla","glc","c-class","e-class","a-class"],
    "ford": [" ford","fiesta","focus","mondeo","kuga","ecosport","puma","ka "],
    "hyundai": [" hyundai"," hyundi","i10","i20","i30","elantra","tucson","kona","santa fe","santafe"],
    "skoda": [" skoda","octavia","fabia","superb","kodiaq","karoq","scala","rapid"],
    "toyota": [" toyota","yaris","corolla","auris","rav4","c-hr","chr","aygo"],
    "vauxhall": [" vauxhall","astra","corsa","insignia","mokka","zafira","vivaro"],
    "vw": [" vw","volkswagen","golf","polo","passat","tiguan","touran","touareg","up "],
}

# ===================== Utils =====================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def tidy_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    if "enginesize" in df.columns and "engineSize" not in df.columns:
        df.rename(columns={"enginesize":"engineSize"}, inplace=True)
    if "fueltype" in df.columns and "fuelType" not in df.columns:
        df.rename(columns={"fueltype":"fuelType"}, inplace=True)
    if "tax(£)" in df.columns and ("tax" not in df.columns or df["tax"].isna().all()):
        df.rename(columns={"tax(£)":"tax"}, inplace=True)
    for alias in ("model_name","series","variant"):
        if alias in df.columns and "model" not in df.columns:
            df.rename(columns={alias:"model"}, inplace=True); break
    for alias in ("brand","make","manufacturer","brand_name"):
        if alias in df.columns:
            df.rename(columns={alias:"brand"}, inplace=True); break
    return df

def infer_brand_from_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "brand" not in df.columns:
        df["brand"] = np.nan
    if "model" not in df.columns:
        return df
    def detect(m: str) -> Optional[str]:
        if not isinstance(m, str): return None
        t = " " + m.lower().replace("-", " ") + " "
        for brand, kws in BRAND_KEYWORDS.items():
            if any(kw in t for kw in kws):
                return brand
        return None
    mask = df["brand"].isna() | (df["brand"].astype(str).str.strip()=="") | (df["brand"].astype(str).str.lower()=="nan")
    df.loc[mask, "brand"] = df.loc[mask, "model"].map(detect)
    return df

def persist_set_df(df: pd.DataFrame) -> None:
    st.session_state["df"] = df
    try:
        df.to_pickle(PERSIST_PATH)
    except Exception as e:
        st.warning(f"Could not persist dataset: {e}")

def persist_clear_df() -> None:
    st.session_state.pop("df", None)
    try:
        if os.path.exists(PERSIST_PATH):
            os.remove(PERSIST_PATH)
    except Exception as e:
        st.warning(f"Could not remove persisted dataset: {e}")

@st.cache_data(show_spinner=False)
def read_csv_local(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def _configure_kaggle_env():
    ku = os.environ.get("KAGGLE_USERNAME") or st.secrets.get("KAGGLE_USERNAME", None)
    kk = os.environ.get("KAGGLE_KEY") or st.secrets.get("KAGGLE_KEY", None)
    if ku and kk:
        os.environ["KAGGLE_USERNAME"] = ku
        os.environ["KAGGLE_KEY"] = kk

@st.cache_data(show_spinner=True)
def kaggle_download_csv(slug: str, file_name: str) -> Optional[pd.DataFrame]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        st.warning("The 'kaggle' package is not installed, skipping Kaggle autoload.")
        return None
    _configure_kaggle_env()
    api = KaggleApi(); api.authenticate()
    dl_dir = "/mnt/data/kaggle_autoload"
    os.makedirs(dl_dir, exist_ok=True)
    fp = api.dataset_download_file(slug, file_name, path=dl_dir, force=True, quiet=True)
    target = os.path.join(dl_dir, file_name)
    if fp.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(fp, "r") as zf:
            zf.extractall(dl_dir)
    if not os.path.exists(target):
        # some datasets name file differently inside zip
        # fallback: first csv in dir
        cands = [os.path.join(dl_dir, f) for f in os.listdir(dl_dir) if f.lower().endswith(".csv")]
        target = cands[0] if cands else None
    if not target or not os.path.exists(target):
        return None
    return pd.read_csv(target)

def autoload_default_dataset() -> Optional[pd.DataFrame]:
    # 1) previously persisted pickle (same server instance)
    if os.path.exists(PERSIST_PATH):
        try:
            return pd.read_pickle(PERSIST_PATH)
        except Exception:
            pass
    # 2) repo-local CSV
    if DEFAULT_DATA_PATH and os.path.exists(DEFAULT_DATA_PATH):
        try:
            return read_csv_local(DEFAULT_DATA_PATH)
        except Exception as e:
            st.warning(f"Failed to read DEFAULT_DATA_PATH: {e}")
    # 3) public URL
    if DEFAULT_DATA_URL:
        try:
            return read_csv_url(DEFAULT_DATA_URL)
        except Exception as e:
            st.warning(f"Failed to read DEFAULT_DATA_URL: {e}")
    # 4) Kaggle
    if DEFAULT_KAGGLE.get("slug"):
        df = kaggle_download_csv(DEFAULT_KAGGLE["slug"], DEFAULT_KAGGLE.get("file",""))
        if df is not None:
            return df
    return None

@st.cache_resource(show_spinner=False)
def load_pipe(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load(path)

# ===================== FE & Predict =====================
def fe_transform_single(year:int, mileage:float, engine_size:float,
                        transmission:str, fuel_type:str,
                        mpg:Optional[float]=None, tax:Optional[float]=None,
                        model_name:Optional[str]=None) -> pd.DataFrame:
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
            X[c] = np.nan
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
    fallback = (20_000, 120_000)
    if df is None or "mileage" not in df.columns or "year" not in df.columns:
        return fallback
    tmp = df.copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp["mileage"] = pd.to_numeric(tmp["mileage"], errors="coerce")
    tmp = tmp[(tmp["year"] >= 1980) & (tmp["year"] <= CURRENT_YEAR)]
    tmp["vehicle_age"] = CURRENT_YEAR - tmp["year"]
    tmp = tmp[tmp["vehicle_age"] >= 0]
    denom = (tmp["vehicle_age"] + 1).replace(0, 1)
    tmp["km_per_year_calc"] = tmp["mileage"] / denom
    if brand and "brand" in tmp.columns:
        tmp = tmp[tmp["brand"].astype(str).str.lower() == str(brand).lower()]
    if model and "model" in tmp.columns:
        tmp = tmp[tmp["model"].astype(str) == str(model)]
    valid = tmp["km_per_year_calc"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.shape[0] < 10:
        return fallback
    q10 = float(valid.quantile(0.10)); q90 = float(valid.quantile(0.90))
    if not np.isfinite(q10) or not np.isfinite(q90) or q10 <= 0 or q90 <= 0:
        return fallback
    vehicle_age = max(0, CURRENT_YEAR - int(sel_year))
    lo = int(max(0, round(q10 * (vehicle_age + 1), -3)))
    hi = int(max(lo + 1_000, round(q90 * (vehicle_age + 1), -3)))
    return (lo, hi)

def ood_warning_for_mileage(sel_mileage: float, suggested: Tuple[int,int]) -> None:
    lo, hi = suggested
    margin = 0.25 * (hi - lo)
    if sel_mileage < lo - margin or sel_mileage > hi + margin:
        st.warning("Mileage seems outside the typical range for this brand/model/year. Predictions may be less reliable.")

# ===================== Sidebar: Pipelines =====================
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

# ===================== Sidebar: Dataset loader (optional override) =====================
with st.sidebar:
    st.divider()
    st.header("📂 Dataset (required)")
    st.caption("App auto-loads a default dataset. You may override it here.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
    ds_path = st.text_input("...or local CSV path (optional)", value="", key="local_path_csv")
    d1, d2 = st.columns(2)
    if d1.button("Load Dataset", use_container_width=True, key="btn_load_dataset"):
        try:
            if up is not None:
                df_loaded = pd.read_csv(up)
            elif ds_path.strip():
                if not os.path.exists(ds_path):
                    st.error("Local path not found."); df_loaded = None
                else:
                    df_loaded = pd.read_csv(ds_path)
            else:
                df_loaded = None

            if df_loaded is None:
                st.error("No file selected or invalid path.")
            else:
                df_loaded = tidy_column_names(df_loaded)
                if "brand" not in df_loaded.columns:
                    df_loaded = infer_brand_from_model(df_loaded)
                persist_set_df(df_loaded)
                st.success(f"Dataset loaded & persisted: {len(df_loaded):,} rows.")
        except Exception as e:
            st.error(f"Dataset load failed: {e}")

    if d2.button("Clear Dataset", use_container_width=True, key="btn_clear_dataset"):
        persist_clear_df()
        st.info("Dataset cleared from memory and disk.")

# ===================== Autoload GLOBAL dataset for everyone =====================
if "df" not in st.session_state:
    df_auto = autoload_default_dataset()
    if df_auto is not None:
        df_auto = tidy_column_names(df_auto)
        if "brand" not in df_auto.columns:
            df_auto = infer_brand_from_model(df_auto)
        # do NOT pickle the autoload by default; or do it to speed up
        st.session_state["df"] = df_auto
        st.success("Default dataset auto-loaded.")
    else:
        st.error("No default dataset available. Set DEFAULT_DATA_PATH/URL or Kaggle slug.")
        st.stop()

# ===================== Pipelines quick autoload =====================
if "pipe_mid" not in st.session_state:
    try:
        st.session_state["pipe_mid"] = load_pipe("best_lightgbm_optuna.joblib")
    except Exception as e:
        st.warning(str(e))

pipe_mid = st.session_state.get("pipe_mid")
pipe_q10 = st.session_state.get("pipe_q10")
pipe_q90 = st.session_state.get("pipe_q90")

# ===================== Require dataset & columns =====================
df_global = st.session_state.get("df")
df_global = tidy_column_names(df_global)
if "brand" not in df_global.columns:
    df_global = infer_brand_from_model(df_global)

required_now = ["brand","model","year","mileage","engineSize"]
missing = [c for c in required_now if c not in df_global.columns]
if missing:
    st.error(f"Missing required columns after mapping: {missing}.")
    st.stop()

df_sel = df_global.dropna(subset=["brand","model","engineSize"]).copy()
df_sel["brand"] = df_sel["brand"].astype(str).str.strip()
df_sel["model"] = df_sel["model"].astype(str).str.strip()

# ===================== Inputs (Brand → Model → Engine Size) =====================
st.subheader("📝 Inputs")
st.markdown("**Vehicle Identification**")

brands = sorted([b for b in df_sel["brand"].unique().tolist() if b and b.lower() != "nan"])
if not brands:
    st.error("No brands determined. Update BRAND_KEYWORDS or add 'brand' to CSV.")
    st.stop()
brand_val = st.selectbox("Brand", options=brands, index=0, key="brand_sel")

models = (
    df_sel.loc[df_sel["brand"] == brand_val, "model"]
    .dropna().astype(str).str.strip().unique().tolist()
)
models = sorted(models)
if not models:
    st.error("No models found for the selected brand.")
    st.stop()
model_name = st.selectbox("Model", options=models, index=0, key="model_sel")

def get_engine_sizes_for(df: pd.DataFrame, brand: str, model: str) -> List[float]:
    s = df.loc[
        (df["brand"].astype(str) == str(brand)) &
        (df["model"].astype(str) == str(model)),
        "engineSize"
    ]
    sizes = pd.to_numeric(s, errors="coerce").dropna()
    if sizes.empty:
        return []
    return sorted(np.unique(np.round(sizes.values.astype(float), 2)).tolist())

sizes = get_engine_sizes_for(df_sel, brand_val, model_name)
if sizes:
    default_idx = int(np.argmin([abs(s - float(np.median(sizes))) for s in sizes]))
    engine_size = st.selectbox("Engine Size (L)", options=sizes, index=default_idx, key="engine_sel")
else:
    st.warning("No engine sizes found for this model; please provide a custom value.")
    engine_size = st.number_input("Engine Size (L)", min_value=0.6, max_value=6.0, value=1.6, step=0.1)

# Dynamic Transmission/Fuel options from dataset
transmission_options = (
    df_sel.loc[df_sel["brand"] == brand_val, "transmission"]
    .dropna().astype(str).str.strip().unique().tolist()
) or ["Manual","Automatic","Semi-Auto"]
fuel_options = (
    df_sel.loc[df_sel["brand"] == brand_val, "fuelType"]
    .dropna().astype(str).str.strip().unique().tolist()
) or ["Petrol","Diesel","Hybrid"]

c1, c2 = st.columns(2)
with c1:
    year = st.number_input("Year", min_value=1980, max_value=CURRENT_YEAR, value=2018, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=60_000, step=1)
with c2:
    transmission = st.selectbox("Transmission", transmission_options)
    fuel_type = st.selectbox("Fuel Type", fuel_options)
    mpg_opt = st.number_input("Miles per gallon (MPG)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, help="MPG = Miles per gallon")

mpg_val = None if mpg_opt == 0.0 else mpg_opt
tax_val = None  # no UI; optional

# Suggested KM window + OOD notice
suggested_km = suggest_mileage_range(df_sel, brand_val, model_name, year)
ood_warning_for_mileage(mileage, suggested_km)

# ===================== Predict =====================
@st.cache_resource(show_spinner=False)
def _noop(x):  # tiny helper to avoid caching errors in buttons
    return x

if st.button("💡 Predict Price", use_container_width=True):
    if pipe_mid is None:
        st.error("Mid pipeline is not loaded.")
    else:
        X_one = fe_transform_single(year, mileage, float(engine_size), transmission, fuel_type, mpg_val, tax_val, model_name)
        try:
            mid = predict_mid(pipe_mid, X_one)
            st.success(f"Estimated Price: **£{mid:,.0f}**")
            k = st.slider("MAE Scale (band width)", 0.5, 2.0, 1.0, 0.1)
            mae_band = k * GLOBAL_MAE_GBP
            lower_mae = max(0.0, mid - mae_band); upper_mae = mid + mae_band
            st.info(f"Practical Range: **£{lower_mae:,.0f} – £{upper_mae:,.0f}** (±£{mae_band:,.0f})")
            qi = predict_interval(st.session_state.get("pipe_q10"), st.session_state.get("pipe_q90"), X_one)
            if qi is not None:
                lo, hi = qi
                mid_adj = float(np.clip(mid, lo, hi))
                if abs(mid_adj - mid) > 1e-6:
                    st.caption(f"Adjusted to stay within quantile band: £{mid:,.0f} → £{mid_adj:,.0f}")
            df_out = pd.DataFrame({"Metric":["Recommended Estimate","Likely Lower Bound","Likely Upper Bound"],
                                   "£":[round(mid), round(lower_mae), round(upper_mae)]})
            st.dataframe(df_out, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===================== Scenario Analysis =====================
st.divider()
st.subheader("📈 Scenario Analysis")
tab_km, tab_year, tab_engine = st.tabs(["Mileage (KM)", "Year", "Engine Size"])

with tab_km:
    lo_km, hi_km = suggested_km
    km_min, km_max = st.slider("KM range", 0, 300_000, (int(lo_km), int(hi_km)), step=5_000)
    n_pts = st.selectbox("Number of points (KM)", [5,6,7,8,9,10], index=5)
    if st.button("Render KM Chart", use_container_width=True):
        if st.session_state.get("pipe_mid") is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.linspace(km_min, km_max, n_pts, dtype=int)
            preds = [predict_mid(st.session_state["pipe_mid"],
                                 fe_transform_single(year, km, float(engine_size), transmission, fuel_type, mpg_val, tax_val, model_name)) for km in grid]
            df_plot = pd.DataFrame({"mileage": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("mileage"))
            st.caption("Effect of mileage with all other inputs fixed. Defaults inferred from your dataset (10–90% km/year for slice).")

with tab_year:
    y_min, y_max = st.slider("Year range", 1980, CURRENT_YEAR, (2008, CURRENT_YEAR), step=1)
    n_pts_y = st.selectbox("Number of points (Year)", [5,6,7,8,9,10], index=5)
    if st.button("Render Year Chart", use_container_width=True):
        if st.session_state.get("pipe_mid") is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.linspace(y_min, y_max, n_pts_y, dtype=int)
            preds = [predict_mid(st.session_state["pipe_mid"],
                                 fe_transform_single(yv, mileage, float(engine_size), transmission, fuel_type, mpg_val, tax_val, model_name)) for yv in grid]
            df_plot = pd.DataFrame({"year": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("year"))
            st.caption("Effect of year with all other inputs fixed.")

with tab_engine:
    e_min_default, e_max_default = (float(min(sizes)), float(max(sizes))) if sizes else (1.0, 2.5)
    e_min, e_max = st.slider("Engine Size (L) range", 0.6, 6.0, (e_min_default, e_max_default), step=0.1)
    n_pts_e = st.selectbox("Number of points (Engine)", [5,6,7,8,9,10], index=5)
    if st.button("Render Engine Chart", use_container_width=True):
        if st.session_state.get("pipe_mid") is None: st.error("Load the mid pipeline first.")
        else:
            grid = np.round(np.linspace(e_min, e_max, n_pts_e), 2)
            preds = [predict_mid(st.session_state["pipe_mid"],
                                 fe_transform_single(year, mileage, float(ev), transmission, fuel_type, mpg_val, tax_val, model_name)) for ev in grid]
            df_plot = pd.DataFrame({"engineSize": grid, "pred_price": np.array(preds)})
            st.line_chart(df_plot.set_index("engineSize"))
            st.caption("Effect of engine size with all other inputs fixed.")

st.markdown("""
**Notes**
- The estimate reflects average behavior learned from historical data.

- Based on historical data from vehicles registered in the United Kingdom between 2000 and 2020.
""")
