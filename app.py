"""
Concrete Compressive Strength Predictor
B.Tech CSE Semester V Machine Learning Case Study

A clean, academic Streamlit application that predicts concrete compressive strength (MPa)
from mix composition and curing age using a trained machine learning model.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# Configure page settings — Centered, cohesive layout
st.set_page_config(
    page_title="Concrete Compressive Strength Predictor",
    page_icon="🧱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Unified Dark/Slate theme with focused container
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .block-container {
        max-width: 860px !important;
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        margin: 0 auto !important;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .hero-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-text {
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 14px;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
    }
    
    .field-desc {
        font-size: 11px;
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 12px;
        line-height: 1.3;
    }

    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        margin: 20px 0 10px 0;
    }
    .result-badge-text {
        font-size: 11px;
        font-weight: 700;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-value-big {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 16px 0;
        line-height: 1.1;
    }
    .result-value-unit {
        font-size: 20px;
        color: #38bdf8;
        font-weight: 600;
    }

    /* Error Statistics Box */
    .range-box {
        margin-top: 16px;
        background: rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
    }
    .range-box-title {
        font-size: 11.5px;
        font-weight: 700;
        color: #93c5fd;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .error-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }
    .error-row:last-of-type {
        border-bottom: none;
    }
    .error-label {
        font-size: 13px;
        color: #cbd5e1;
        font-weight: 500;
    }
    .error-badge {
        font-size: 13px;
        font-weight: 700;
        color: #f8fafc;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 5px;
        padding: 2px 8px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .range-box-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 8px;
        line-height: 1.4;
    }

    /* Engineering Disclaimer */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 24px;
        font-size: 11.5px;
        color: #fbbf24;
        line-height: 1.5;
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.2rem !important;
        font-size: 14px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Explicit Model & Metadata Loading with Strict Error Handling
# -------------------------------------------------------------
MODEL_FILE = "best_model_pipeline.joblib"
METADATA_FILE = "model_metadata.json"

if not os.path.isfile(MODEL_FILE):
    st.error(f"The trained model file is missing. Please ensure '{MODEL_FILE}' is present in the application directory.")
    st.stop()

if not os.path.isfile(METADATA_FILE):
    st.error(f"The model metadata file is missing. Please ensure '{METADATA_FILE}' is present in the application directory.")
    st.stop()


@st.cache_resource
def load_pipeline():
    """Load trained model pipeline safely."""
    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        st.error(f"Error loading trained model pipeline: {exc}")
        st.stop()


def load_metadata():
    """Load model metadata safely from disk."""
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Error reading model metadata: {exc}")
        st.stop()


def load_assets():
    """Load model pipeline and metadata safely."""
    return load_pipeline(), load_metadata()


pipeline, metadata = load_assets()

# Strict verification of required metadata fields — zero hard-coded fallback values
if not isinstance(metadata, dict):
    st.error("Invalid metadata format: 'model_metadata.json' must contain a valid JSON object.")
    st.stop()

if "model_name" not in metadata:
    st.error("Required key 'model_name' is missing from 'model_metadata.json'.")
    st.stop()

if "test_metrics" not in metadata:
    st.error("Required section 'test_metrics' is missing from 'model_metadata.json'.")
    st.stop()

test_metrics = metadata["test_metrics"]
for metric_key in ["r2", "mse", "rmse", "mae"]:
    if metric_key not in test_metrics:
        st.error(f"Required test evaluation metric '{metric_key}' is missing from 'model_metadata.json'.")
        st.stop()

if "cv_metrics_5fold" not in metadata:
    st.error("Required section 'cv_metrics_5fold' is missing from 'model_metadata.json'.")
    st.stop()

cv_metrics = metadata["cv_metrics_5fold"]
for cv_key in ["cv_r2_mean", "cv_rmse_mean", "cv_mae_mean"]:
    if cv_key not in cv_metrics:
        st.error(f"Required cross-validation metric '{cv_key}' is missing from 'model_metadata.json'.")
        st.stop()

if "error_analysis" not in metadata or "p95_error_margin" not in metadata["error_analysis"]:
    st.error("Required error analysis statistic 'p95_error_margin' is missing from 'model_metadata.json'.")
    st.stop()

MODEL_NAME = metadata["model_name"]
TEST_R2 = float(test_metrics["r2"])
TEST_MSE = float(test_metrics["mse"])
TEST_RMSE = float(test_metrics["rmse"])
TEST_MAE = float(test_metrics["mae"])
CV_R2 = float(cv_metrics["cv_r2_mean"])
CV_RMSE = float(cv_metrics["cv_rmse_mean"])
CV_MAE = float(cv_metrics["cv_mae_mean"])
P95_ERROR = float(metadata["error_analysis"]["p95_error_margin"])

FEATURE_COLS = [
    'Cement',
    'Blast Furnace Slag',
    'Fly Ash',
    'Water',
    'Superplasticizer',
    'Coarse Aggregate',
    'Fine Aggregate',
    'Age'
]

# -------------------------------------------------------------
# Input Ranges Grounded in Training Dataset (Concrete_Compressive_Strength.csv)
# -------------------------------------------------------------
FEATURE_BOUNDS = {
    'Cement': (102.0, 540.0),
    'Blast Furnace Slag': (0.0, 359.4),
    'Fly Ash': (0.0, 200.1),
    'Water': (121.75, 247.0),
    'Superplasticizer': (0.0, 32.2),
    'Coarse Aggregate': (801.0, 1145.0),
    'Fine Aggregate': (594.0, 992.6),
    'Age': (1, 365)
}

# -------------------------------------------------------------
# Hero Header
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-heading">Concrete Compressive Strength Predictor</div>
    <div class="hero-text">
        Estimate concrete compressive strength from mix composition and curing age using a trained machine learning model.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Concrete Mix Inputs (8 Required Features)
# -------------------------------------------------------------
st.markdown('<div class="section-title">Concrete Mix Inputs</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    min_cem, max_cem = FEATURE_BOUNDS['Cement']
    cement = st.number_input(
        "Cement (kg/m³)",
        min_value=float(min_cem),
        max_value=float(max_cem),
        value=280.0,
        step=5.0,
        help="Cement content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Cement content in the concrete mix ({min_cem:.1f} – {max_cem:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_slag, max_slag = FEATURE_BOUNDS['Blast Furnace Slag']
    slag = st.number_input(
        "Blast Furnace Slag (kg/m³)",
        min_value=float(min_slag),
        max_value=float(max_slag),
        value=0.0,
        step=5.0,
        help="Blast furnace slag content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Blast furnace slag content in the concrete mix ({min_slag:.1f} – {max_slag:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_ash, max_ash = FEATURE_BOUNDS['Fly Ash']
    fly_ash = st.number_input(
        "Fly Ash (kg/m³)",
        min_value=float(min_ash),
        max_value=float(max_ash),
        value=0.0,
        step=5.0,
        help="Fly ash content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Fly ash content in the concrete mix ({min_ash:.1f} – {max_ash:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_water, max_water = FEATURE_BOUNDS['Water']
    water = st.number_input(
        "Water (kg/m³)",
        min_value=float(min_water),
        max_value=float(max_water),
        value=185.0,
        step=2.5,
        help="Water content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Water content in the concrete mix ({min_water:.1f} – {max_water:.1f} kg/m³).</div>', unsafe_allow_html=True)

with col2:
    min_sp, max_sp = FEATURE_BOUNDS['Superplasticizer']
    superplasticizer = st.number_input(
        "Superplasticizer (kg/m³)",
        min_value=float(min_sp),
        max_value=float(max_sp),
        value=0.0,
        step=0.5,
        help="Superplasticizer content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Superplasticizer content in the concrete mix ({min_sp:.1f} – {max_sp:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_coarse, max_coarse = FEATURE_BOUNDS['Coarse Aggregate']
    coarse_agg = st.number_input(
        "Coarse Aggregate (kg/m³)",
        min_value=float(min_coarse),
        max_value=float(max_coarse),
        value=1000.0,
        step=10.0,
        help="Coarse aggregate content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Coarse aggregate content in the concrete mix ({min_coarse:.1f} – {max_coarse:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_fine, max_fine = FEATURE_BOUNDS['Fine Aggregate']
    fine_agg = st.number_input(
        "Fine Aggregate (kg/m³)",
        min_value=float(min_fine),
        max_value=float(max_fine),
        value=770.0,
        step=10.0,
        help="Fine aggregate content in the concrete mix."
    )
    st.markdown(f'<div class="field-desc">Fine aggregate content in the concrete mix ({min_fine:.1f} – {max_fine:.1f} kg/m³).</div>', unsafe_allow_html=True)

    min_age, max_age = FEATURE_BOUNDS['Age']
    age = st.slider(
        "Curing Age (days)",
        min_value=int(min_age),
        max_value=int(max_age),
        value=28,
        step=1,
        help="Curing age of the concrete sample in days."
    )
    st.markdown(f'<div class="field-desc">Curing age of the concrete sample in days ({min_age} – {max_age} days).</div>', unsafe_allow_html=True)

# Prediction Button
btn_calc = st.button("Calculate Predicted Strength", type="primary", use_container_width=True)

# -------------------------------------------------------------
# Prediction & Results Display (ONLY after clicking Calculate)
# -------------------------------------------------------------
if btn_calc:
    input_row = pd.DataFrame([{
        'Cement': cement,
        'Blast Furnace Slag': slag,
        'Fly Ash': fly_ash,
        'Water': water,
        'Superplasticizer': superplasticizer,
        'Coarse Aggregate': coarse_agg,
        'Fine Aggregate': fine_agg,
        'Age': age
    }])[FEATURE_COLS]

    strength_pred = float(pipeline.predict(input_row)[0])

    card_html = f"""<div class="result-card">
<div class="result-badge-text">Predicted Compressive Strength</div>
<div class="result-value-big">{strength_pred:.2f} <span class="result-value-unit">MPa</span></div>
<div class="range-box">
<div class="range-box-title">Model Error Statistics</div>
<div class="error-row">
<span class="error-label">Average Error (MAE)</span>
<span class="error-badge">{TEST_MAE:.2f} MPa</span>
</div>
<div class="error-row">
<span class="error-label">95th Percentile Absolute Error</span>
<span class="error-badge">{P95_ERROR:.2f} MPa</span>
</div>
<div class="range-box-sub">
Based on model evaluation data. This is not a guaranteed prediction interval.
</div>
</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)

def get_comparison_table(metadata_obj):
    """Retrieve the five-model benchmark comparison from model metadata."""
    if "model_comparison" in metadata_obj and isinstance(metadata_obj["model_comparison"], list):
        return pd.DataFrame(metadata_obj["model_comparison"])
    return None

# -------------------------------------------------------------
# Academic Model Details & 5-Fold Evaluation
# -------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Academic Model Details & 5-Fold Evaluation"):
    st.markdown(f"### Selected Model: **{MODEL_NAME}**")
    
    # Selected model evaluation metrics from metadata
    metrics_summary = pd.DataFrame([
        {"Metric": "Test R² (Coefficient of Determination)", "Value": f"{TEST_R2:.4f}"},
        {"Metric": "Test MSE (Mean Squared Error)", "Value": f"{TEST_MSE:.4f} MPa²"},
        {"Metric": "Test RMSE (Root Mean Squared Error)", "Value": f"{TEST_RMSE:.4f} MPa"},
        {"Metric": "Test MAE (Mean Absolute Error)", "Value": f"{TEST_MAE:.4f} MPa"},
        {"Metric": "5-Fold CV Mean R²", "Value": f"{CV_R2:.4f}"},
        {"Metric": "5-Fold CV Mean RMSE", "Value": f"{CV_RMSE:.4f} MPa"},
        {"Metric": "5-Fold CV Mean MAE", "Value": f"{CV_MAE:.4f} MPa"}
    ])
    st.table(metrics_summary)
    
    st.markdown("#### Five-Model Comparison")
    comp_df = get_comparison_table(metadata)
    if comp_df is not None:
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
    else:
        st.info("Model comparison data is not available in model_metadata.json.")

    st.markdown("#### Model Selection")
    st.write(
        f"The selected model ({MODEL_NAME}) was chosen based on the evaluation results obtained "
        f"on the test set (R² = {TEST_R2:.4f}, RMSE = {TEST_RMSE:.2f} MPa, MAE = {TEST_MAE:.2f} MPa) "
        f"and 5-fold cross-validation (mean R² = {CV_R2:.4f})."
    )

# -------------------------------------------------------------
# Engineering Disclaimer
# -------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <strong>Academic Disclaimer:</strong> For academic and preliminary predictive analysis only. 
    ML predictions should not replace required physical laboratory testing, engineering validation, 
    quality assurance, or applicable construction standards.
</div>
""", unsafe_allow_html=True)
