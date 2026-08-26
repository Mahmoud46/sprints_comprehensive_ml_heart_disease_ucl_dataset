import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.analysis_settings import DATA_COLUMNS_NAMES
from config.paths import DATA_CLEVELAND_PATH, FINAL_MODEL_PATH
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="UCI Heart Disease Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_trained_model():
    """Loads the pre-trained machine learning model."""
    try:
        return joblib.load(FINAL_MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_data
def load_historical_data():
    """Loads dataset for visualization.

    Replace 'heart_disease_data.csv' with your dataset filename.
    """
    try:
        df = pd.read_csv(DATA_CLEVELAND_PATH, names=DATA_COLUMNS_NAMES, na_values="?")

        # Drop rows with nan values 
        df = df.dropna().reset_index(drop=True)

        if "num" in df.columns:
            df["Target_Label"] = df["num"].apply(
                lambda x: "Healthy (0)" if x == 0 else "Disease Present (1+)"
            )
        return df
    except FileNotFoundError:
        return None


model = load_trained_model()
df_data = load_historical_data()

st.title("UCI Heart Disease Risk Analyzer & Trend Explorer")
st.markdown(
    "Clinical decision support tool for real-time risk assessment and population trend exploration."
)
st.divider()

tab1, tab2 = st.tabs(
    ["Real-Time Clinical Assessment", "Interactive Dataset Trends"]
)


# ==============================================================================
# TAB 1: REAL-TIME CLINICAL ASSESSMENT (ALL INPUTS COLLECTED -> 8 MODEL FEATURES)
# ==============================================================================
with tab1:
    st.header("1. Enter Patient Clinical Parameters")

    if model is None:
        st.error(
            "⚠️ `best_model.pkl` not found! Ensure your model file is saved in the same directory as `app.py`."
        )
    else:
        # Layout all clinical inputs across 4 columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            age = st.number_input(
                "Age (years)", min_value=1, max_value=120, value=54
            )
            sex_option = st.selectbox("Sex", options=["Male", "Female"])
            sex = 1 if "Male" in sex_option else 0
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[0, 1, 2, 3],
                help="0: Typical Angina | 1: Atypical Angina | 2: Non-anginal | 3: Asymptomatic",
            )
            trestbps = st.number_input(
                "Resting BP (trestbps mm Hg)",
                min_value=80,
                max_value=220,
                value=130,
            )

        with col2:
            chol = st.number_input(
                "Serum Cholesterol (chol mg/dl)",
                min_value=100,
                max_value=600,
                value=240,
            )
            fbs_option = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=["False (0)", "True (1)"],
            )
            fbs = 1 if "True" in fbs_option else 0
            restecg = st.selectbox(
                "Resting ECG Results (restecg)",
                options=[0, 1, 2],
                help="0: Normal | 1: ST-T Abnormality | 2: LV Hypertrophy",
            )
            thalach = st.number_input(
                "Max Heart Rate Achieved (thalach)",
                min_value=60,
                max_value=220,
                value=150,
            )

        with col3:
            exang_option = st.selectbox(
                "Exercise Induced Angina (exang)",
                options=["No (0)", "Yes (1)"],
            )
            exang = 1 if "Yes" in exang_option else 0
            oldpeak = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            slope = st.selectbox(
                "ST Segment Slope (slope)",
                options=[1, 2, 3],
                help="1: Upsloping | 2: Flat | 3: Downsloping",
            )

        with col4:
            ca = st.selectbox(
                "Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3]
            )
            thal = st.selectbox(
                "Thalassemia Result (thal)",
                options=[3, 6, 7],
                help="3: Normal | 6: Fixed Defect | 7: Reversible Defect",
            )

        # ----------------------------------------------------------------------
        # FEATURE SELECTION & REORDERING
        # Mandatory Model Order: [thal, ca, cp, oldpeak, thalach, slope, exang, sex]
        # ----------------------------------------------------------------------
        selected_features_sample = np.array(
            [[thal, ca, cp, oldpeak, thalach, slope, exang, sex]]
        )

        st.divider()
        st.header("2. Risk Assessment Output")

        if st.button("Calculate Heart Disease Risk", type="primary"):
            # Execute prediction on the 8 ordered features
            prediction = model.predict(selected_features_sample)[0]

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.subheader("Diagnostic Status")
                if prediction >= 1:
                    st.error("🚨 **High Risk: Heart Disease Present (Class 1+)**")
                else:
                    st.success("✅ **Low Risk: Normal / Healthy (Class 0)**")

            with res_col2:
                st.subheader("Confidence Score")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(selected_features_sample)[0]
                    disease_prob = (
                        (1 - probs[0]) if prediction >= 1 else probs[0]
                    )
                    confidence_percent = disease_prob * 100

                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confidence_percent,
                            number={"suffix": "%"},
                            title={
                                "text": "Model Confidence",
                                "font": {"size": 18},
                            },
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {
                                    "color": (
                                        "#ef553b"
                                        if prediction >= 1
                                        else "#00cc96"
                                    )
                                },
                                "steps": [
                                    {"range": [0, 50], "color": "#e5ecf6"},
                                    {"range": [50, 100], "color": "#d3e0ea"},
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(height=230, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# TAB 2: INTERACTIVE DATASET TRENDS (USES ALL ORIGINAL DATASET COLUMNS)
# ==============================================================================
with tab2:
    st.header("Population Clinical Trends")
    st.markdown(
        "Visualizing multi-variable risk patterns from your UCI dataset columns."
    )

    if df_data is None:
        st.warning(
            "⚠️ Load your dataset CSV (e.g. `heart_disease_data.csv`) into your directory to activate dataset trend visuals."
        )
    else:
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.subheader("Age vs. Maximum Heart Rate (thalach)")
            fig_scatter = px.scatter(
                df_data,
                x="age",
                y="thalach",
                color="Target_Label",
                size="oldpeak",
                hover_data=["trestbps", "chol", "cp"],
                color_discrete_map={
                    "Healthy (0)": "#00cc96",
                    "Disease Present (1+)": "#ef553b",
                },
                labels={
                    "age": "Age (years)",
                    "thalach": "Max Heart Rate (thalach)",
                },
            )
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with viz_col2:
            st.subheader("Chest Pain Type (cp) & Disease Incidence")
            fig_bar = px.histogram(
                df_data,
                x="cp",
                color="Target_Label",
                barmode="group",
                color_discrete_map={
                    "Healthy (0)": "#00cc96",
                    "Disease Present (1+)": "#ef553b",
                },
                labels={"cp": "Chest Pain Type (0-3)", "count": "Patient Count"},
            )
            fig_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            st.subheader("ST Depression (oldpeak) Distribution")
            fig_box = px.box(
                df_data,
                x="Target_Label",
                y="oldpeak",
                color="Target_Label",
                points="all",
                color_discrete_map={
                    "Healthy (0)": "#00cc96",
                    "Disease Present (1+)": "#ef553b",
                },
            )
            fig_box.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with viz_col4:
            st.subheader("Thalassemia (thal) Result vs. Disease Rate")
            fig_thal = px.histogram(
                df_data,
                x="thal",
                color="Target_Label",
                barmode="group",
                color_discrete_map={
                    "Healthy (0)": "#00cc96",
                    "Disease Present (1+)": "#ef553b",
                },
                labels={"thal": "Thalassemia Result (3, 6, 7)"},
            )
            fig_thal.update_layout(template="plotly_white")
            st.plotly_chart(fig_thal, use_container_width=True)