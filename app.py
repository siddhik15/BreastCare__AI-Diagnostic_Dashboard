# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Breast Cancer AI Diagnostic System",
                   page_icon="🎀",
                   layout="wide")

# -------------------------------
# Load CSS if present
# -------------------------------
def load_css(path="style.css"):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------
# Utilities
# -------------------------------
def safe_load_model(path="bc_rf_pipeline.joblib"):
    p = Path(path)
    if not p.exists():
        st.warning(f"Model file not found at {path}. Predictions will be disabled until model is available.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def safe_load_dataset(path="breast-cancer-wisconsin.csv"):
    p = Path(path)
    if not p.exists():
        st.warning(f"Dataset file not found at {path}. Dataset Explorer and Model Performance will be limited.")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def clean_dataset(df, feature_cols, target_col=None):
    """
    Replace '?' with NaN, coerce features to numeric, return cleaned copy.
    """
    df_clean = df.copy()
    df_clean.replace('?', np.nan, inplace=True)
    df_clean[feature_cols] = df_clean[feature_cols].apply(pd.to_numeric, errors='coerce')
    if target_col and target_col in df_clean.columns:
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    return df_clean

# -------------------------------
# Load model & dataset
# -------------------------------
model = safe_load_model("bc_rf_pipeline.joblib")
dataset = safe_load_dataset("breast-cancer-wisconsin.csv")

# -------------------------------
# Persistent header
# -------------------------------
st.markdown("""
<div class="main-title">
  <h1>🎀 Breast Cancer AI Diagnostic System</h1>
  <p>AI‑Powered Prediction using Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar navigation (Prediction-first)
# -------------------------------
nav = st.sidebar.radio("📌 Navigation", [
    "Prediction", "Patient Analysis", "Dataset Explorer",
    "Model Performance", "About Breast Cancer", "Model Information"
], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**About This App**")
st.sidebar.markdown("AI-assisted breast cancer triage using a Random Forest pipeline. For educational and research use only.")
st.sidebar.caption("Built with ❤️ by Siddhi Kakade")

# -------------------------------
# Session default input
# -------------------------------
default_input = pd.DataFrame([{
    "ClumpThickness": 5,
    "UniformityCellSize": 3,
    "UniformityCellShape": 3,
    "MarginalAdhesion": 2,
    "SingleEpithelialCellSize": 2,
    "BareNuclei": 4,
    "BlandChromatin": 3,
    "NormalNucleoli": 3,
    "Mitoses": 1
}])
if "input_df" not in st.session_state:
    st.session_state.input_df = default_input.copy()

# -------------------------------
# Helper: top metrics row
# -------------------------------
def render_top_metrics(dataset_available):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "699 Records" if dataset_available else "—")
    c2.metric("Features", "9 Clinical")
    c3.metric("Model", "Random Forest")
    c4.metric("Prediction Time", "< 1 sec")
    st.markdown(f"**Date:** 02 August 2026")
    st.markdown("---")

# -------------------------------
# PREDICTION page (default landing)
# -------------------------------
if nav == "Prediction":
    st.title("🔍 Predict Breast Cancer")
    render_top_metrics(dataset is not None)

    st.markdown("**Patient Clinical Parameters (structured table)**")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        clump = st.number_input("Clump Thickness", 1, 10, int(st.session_state.input_df.iloc[0]["ClumpThickness"]))
        cell_size = st.number_input("Uniformity Cell Size", 1, 10, int(st.session_state.input_df.iloc[0]["UniformityCellSize"]))
        cell_shape = st.number_input("Uniformity Cell Shape", 1, 10, int(st.session_state.input_df.iloc[0]["UniformityCellShape"]))
    with col2:
        adhesion = st.number_input("Marginal Adhesion", 1, 10, int(st.session_state.input_df.iloc[0]["MarginalAdhesion"]))
        epithelial = st.number_input("Single Epithelial Cell Size", 1, 10, int(st.session_state.input_df.iloc[0]["SingleEpithelialCellSize"]))
        nuclei = st.number_input("Bare Nuclei", 1, 10, int(st.session_state.input_df.iloc[0]["BareNuclei"]))
    with col3:
        chromatin = st.number_input("Bland Chromatin", 1, 10, int(st.session_state.input_df.iloc[0]["BlandChromatin"]))
        nucleoli = st.number_input("Normal Nucleoli", 1, 10, int(st.session_state.input_df.iloc[0]["NormalNucleoli"]))
        mitoses = st.number_input("Mitoses", 1, 10, int(st.session_state.input_df.iloc[0]["Mitoses"]))

    st.markdown("**Entered values**")
    entered = pd.DataFrame([{
        "ClumpThickness": clump,
        "UniformityCellSize": cell_size,
        "UniformityCellShape": cell_shape,
        "MarginalAdhesion": adhesion,
        "SingleEpithelialCellSize": epithelial,
        "BareNuclei": nuclei,
        "BlandChromatin": chromatin,
        "NormalNucleoli": nucleoli,
        "Mitoses": mitoses
    }])
    st.table(entered.T.rename(columns={0:"Value"}))

    if st.button("🔍 Predict Diagnosis", use_container_width=True):
        st.session_state.input_df = entered.copy()
        if model is None:
            st.error("Model not loaded. Place 'bc_rf_pipeline.joblib' in the app folder to enable predictions.")
        else:
            X_in = st.session_state.input_df
            try:
                pred = model.predict(X_in)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred = None

            proba = None
            if pred is not None:
                try:
                    proba = model.predict_proba(X_in)[0]
                except Exception:
                    try:
                        scores = model.decision_function(X_in)
                        if np.ndim(scores) == 1:
                            s = scores
                            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
                            proba = np.vstack([1 - s_norm, s_norm]).T[0]
                        else:
                            s = scores
                            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
                            proba = s_norm[0]
                    except Exception:
                        proba = np.array([1.0, 0.0]) if pred == 2 else np.array([0.0, 1.0])

            if pred is not None and proba is not None:
                conf = np.max(proba)
                st.subheader("📝 Diagnostic Report")
                left, right = st.columns([1.6, 1])
                with left:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf*100,
                        title={"text": "Confidence"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E91E63"},
                               "steps": [{"range":[0,50],"color":"#F8BBD0"},{"range":[50,75],"color":"#FFD1DC"},{"range":[75,100],"color":"#FCE4EC"}]}
                    ))
                    gauge.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10))
                    st.plotly_chart(gauge, use_container_width=True)
                with right:
                    pie = go.Figure(go.Pie(
                        labels=["Benign", "Malignant"],
                        values=[proba[0], proba[1]],
                        hole=.6,
                        marker=dict(colors=["#4CAF50", "#F44336"])
                    ))
                    pie.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10))
                    st.plotly_chart(pie, use_container_width=True)

                if pred == 2:
                    st.markdown("<div class='green-card'><h2>🟢 BENIGN</h2><p>Low Risk — Routine follow-up recommended.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='red-card'><h2>🔴 MALIGNANT</h2><p>High Risk — Urgent clinical referral recommended.</p></div>", unsafe_allow_html=True)

                st.metric("⚠️ Cancer Risk Score", f"{proba[1]*100:.2f}%")

                st.divider()
                st.subheader("📈 Top 5 Important Features")
                try:
                    rf = model.named_steps.get("rf", None)
                    if rf is not None:
                        features = list(X_in.columns)
                        importance = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values(by="Importance", ascending=False).head(5)
                        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                                     color="Importance", color_continuous_scale=px.colors.sequential.Pinkyl, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Feature importance not available in pipeline.")
                except Exception:
                    st.info("Feature importance not available for this pipeline.")

                st.divider()
                st.subheader("🩺 Medical Interpretation & Next Steps")
                risk_pct = proba[1]*100
                if risk_pct < 20:
                    risk_level = "Low"
                    advice = "Continue routine screening and follow-up with primary care."
                elif risk_pct < 60:
                    risk_level = "Moderate"
                    advice = "Consider additional imaging (mammography/ultrasound) and clinical review."
                else:
                    risk_level = "High"
                    advice = "Urgent referral to oncology/breast clinic and consider biopsy for definitive diagnosis."

                st.markdown(f"**Risk Level:** **{risk_level}** ({risk_pct:.2f}%)")
                st.markdown(f"**Recommended action:** {advice}")
                st.markdown("**Consultant contact template (copy to email):**")
                st.code(f"""Subject: Urgent referral for breast assessment

Patient parameters:
ClumpThickness: {clump}, UniformityCellSize: {cell_size}, UniformityCellShape: {cell_shape},
MarginalAdhesion: {adhesion}, SingleEpithelialCellSize: {epithelial}, BareNuclei: {nuclei},
BlandChromatin: {chromatin}, NormalNucleoli: {nucleoli}, Mitoses: {mitoses}

AI risk score: {risk_pct:.2f}% ({risk_level})
Recommendation: {advice}

Please advise next steps and arrange imaging/biopsy as appropriate.
""")

# -------------------------------
# PATIENT ANALYSIS
# -------------------------------
elif nav == "Patient Analysis":
    st.title("📋 Patient Analysis Report")
    st.markdown("Structured clinical report for the last prediction.")
    st.dataframe(st.session_state.get("input_df", default_input), use_container_width=True)
    st.markdown("---")
    st.markdown("**Interpretation & Notes**")
    st.markdown("""
    - Use the AI risk estimate as a triage aid only.
    - Confirm with clinical imaging (mammography/ultrasound) and histopathology (biopsy) when indicated.
    """)
    st.markdown("---")
    st.markdown("**Printable summary**")
    st.markdown("Use the browser print function to save a PDF of this report.")

# -------------------------------
# DATASET EXPLORER
# -------------------------------
elif nav == "Dataset Explorer":
    st.title("📂 Dataset Explorer")
    if dataset is None:
        st.warning("Dataset not available. Place 'breast-cancer-wisconsin.csv' in the app folder.")
    else:
        feature_headers = ["ClumpThickness","UniformityCellSize","UniformityCellShape",
                           "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei",
                           "BlandChromatin","NormalNucleoli","Mitoses"]
        df_clean = clean_dataset(dataset, feature_headers)
        st.markdown("Preview (cleaned): missing values shown as NaN")
        st.dataframe(df_clean.head(50), use_container_width=True)
        st.markdown("### Simple Filters")
        colf1, colf2 = st.columns(2)
        with colf1:
            min_clump = st.slider("Min Clump Thickness", 1, 10, 1)
        with colf2:
            max_clump = st.slider("Max Clump Thickness", 1, 10, 10)
        filtered = df_clean[(df_clean["ClumpThickness"] >= min_clump) & (df_clean["ClumpThickness"] <= max_clump)]
        st.dataframe(filtered.head(200), use_container_width=True)

# -------------------------------
# MODEL PERFORMANCE (compact confusion matrix)
# -------------------------------
elif nav == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("**Cross Validation Accuracy:** 97.11%")
    if dataset is None or model is None:
        st.warning("Model or dataset missing. Place both 'bc_rf_pipeline.joblib' and 'breast-cancer-wisconsin.csv' in the app folder.")
    else:
        feature_headers = ["ClumpThickness","UniformityCellSize","UniformityCellShape",
                           "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei",
                           "BlandChromatin","NormalNucleoli","Mitoses"]
        target_candidates = [c for c in dataset.columns if c.lower() in ("class","cancertype","diagnosis","target","class_label","label")]
        if not target_candidates:
            st.error("Target column not found. Ensure dataset has a target column named 'Class' or 'CancerType' with values 2 (benign) and 4 (malignant).")
        else:
            target_header = target_candidates[0]
            df_clean = clean_dataset(dataset, feature_headers + [target_header])
            df_clean = df_clean.dropna(subset=[target_header])
            df_clean[target_header] = pd.to_numeric(df_clean[target_header], errors='coerce')
            df_clean = df_clean.dropna(subset=[target_header])
            X = df_clean[feature_headers]
            y = df_clean[target_header].astype(int)

            imputer = SimpleImputer(strategy="median")
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_headers)

            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, stratify=y, random_state=42)

            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                st.error(f"Model prediction failed on test set: {e}")
                y_pred = np.array([2]*len(X_test))

            # Compact Plotly confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[2,4])
            labels = ["Benign", "Malignant"]
            z = cm.tolist()
            z_text = [[str(v) for v in row] for row in z]

            heatmap = go.Figure(data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=z_text,
                texttemplate="%{text}",
                colorscale="Reds",
                showscale=False,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            heatmap.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="True",
                margin=dict(l=40, r=20, t=40, b=40),
                height=300,
                width=420,
                font=dict(size=12)
            )
            st.plotly_chart(heatmap, use_container_width=False)

            # ROC Curve
            y_true_binary = np.array([1 if val == 4 else 0 for val in y_test])
            try:
                y_score = model.predict_proba(X_test)[:, 1]
            except Exception:
                try:
                    y_score = model.decision_function(X_test)
                except Exception:
                    y_score = np.array([1 if p == 4 else 0 for p in y_pred])

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}", line=dict(color="#E91E63", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name="Random", line=dict(dash='dash', color='#999')))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360, margin=dict(t=40,b=40,l=40,r=40))
            st.plotly_chart(fig_roc, use_container_width=True)

# -------------------------------
# ABOUT BREAST CANCER
# -------------------------------
elif nav == "About Breast Cancer":
    st.title("📚 About Breast Cancer")
    st.markdown("**Updated:** 02 August 2026")
    st.markdown("""
    **Overview**  
    Breast cancer occurs when cells in the breast grow out of control. Most breast cancers begin in the ducts or lobules. Early detection through screening and awareness significantly improves outcomes.

    **Key Facts (official summaries)**  
    - Breast cancer is the most common cancer in women worldwide.  
    - Early detection and treatment can lead to high survival rates.  
    - Regular screening (mammography) and awareness of changes in the breast are critical.

    **Risk Factors**  
    - Increasing age (risk rises after 40)  
    - Family history of breast cancer (BRCA1/BRCA2 mutations increase risk)  
    - Reproductive history (early menarche, late menopause)  
    - Hormone replacement therapy and certain lifestyle factors (obesity, alcohol)

    **Symptoms**  
    - New lump in the breast or underarm  
    - Change in breast size or shape  
    - Skin dimpling or nipple retraction  
    - Unusual nipple discharge

    **Screening & Prevention**  
    - Follow local screening guidelines for mammography.  
    - Perform regular self-exams and report changes promptly.  
    - Discuss genetic testing if there is a strong family history.

    **Note**: This summary is adapted from authoritative public health sources (WHO, NCI) for educational purposes. Always consult healthcare professionals for medical advice.
    """)
    st.markdown("[Learn more from WHO](https://www.who.int) • [NCI](https://www.cancer.gov)")

# -------------------------------
# MODEL INFORMATION
# -------------------------------
elif nav == "Model Information":
    st.title("🧠 Model Information")
    st.markdown("**Model Card (summary)**")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier  
    - **Framework**: Scikit-Learn  
    - **Preprocessing**: Pipeline with imputation and scaling (as saved in `bc_rf_pipeline.joblib`)  
    - **Cross Validation Accuracy**: 97.11% (reported)  
    - **Model Status**: Active  
    - **Last Updated**: 02 August 2026
    """)
    st.markdown("**Notes on usage and limitations**")
    st.markdown("""
    - The model is trained on the Wisconsin Breast Cancer dataset (cytology features).  
    - Intended as decision support only; not a diagnostic substitute.  
    - External validation recommended before clinical use.
    """)

# -------------------------------
# Footer / Disclaimer
# -------------------------------
st.markdown("---")
st.markdown("<div class='footer'>This is an AI system for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</div>", unsafe_allow_html=True)
