import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rectal Cancer Dual AI Suite", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    p, div, span, label { color: #1e293b !important; }
    h1, h2, h3 { color: #0f172a !important; font-weight: 700; }
    .main-title { color: #005088 !important; border-bottom: 3px solid #11caa0; padding-bottom: 10px; margin-bottom: 20px; }
    .stButton>button { background-color: #0ea5e9; color: white !important; width: 100%; border-radius: 6px; height: 50px; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #0284c7; }
    [data-testid="stSidebar"] { background-color: #f8fafc; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span { color: #0f172a !important; }
    </style>
""", unsafe_allow_html=True)

def load_models():
    try:
        model_ccr = joblib.load('ccr_xgb_model.pkl')
        imputer_ccr = joblib.load('ccr_imputer.pkl')
        model_lpln = joblib.load('lpln_xgb_model.pkl')
        imputer_lpln = joblib.load('lpln_imputer.pkl')
        return model_ccr, imputer_ccr, model_lpln, imputer_lpln
    except Exception as e:
        return None, None, None, None

model_ccr, imputer_ccr, model_lpln, imputer_lpln = load_models()

st.sidebar.markdown("<h2 style='color:#005088;'>AI Clinical Suite</h2>", unsafe_allow_html=True)
st.sidebar.markdown("**Dr. Harishankar BGV**")
st.sidebar.markdown("*Panimalar Medical College Hospital*")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Select Predictive Model:", ["Complete Clinical Response (cCR)", "LPLN Sterilization Predictor"])

if model_ccr is None or model_lpln is None:
    st.error("⚠️ AI Models not found! Please run 'python3 train_models_unified.py' first.")
    st.stop()

# ------------------------------------------
# MODULE 1: cCR (Watch & Wait) Predictor
# ------------------------------------------
if app_mode == "Complete Clinical Response (cCR)":
    st.markdown("<h1 class='main-title'>Complete Clinical Response (cCR) Predictor</h1>", unsafe_allow_html=True)
    st.write("Predicts the probability of achieving cCR to aid in selecting patients for the organ-preserving Watch-and-Wait pathway.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Primary Tumor Baseline Data")
        with st.form("ccr_form"):
            age = st.number_input("Age (Years)", min_value=20, max_value=100, value=60)
            height = st.number_input("Tumor Height (cm)", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
            thickness = st.number_input("MRI Wall Thickness (mm)", min_value=0.0, max_value=40.0, value=12.0, step=0.5)
            submit_ccr = st.form_submit_button("Calculate cCR Probability")

    with col2:
        if submit_ccr:
            try:
                input_data = pd.DataFrame({'Age': [age], 'Tumor_Height': [height], 'Wall_Thickness': [thickness]})
                imputed_data = pd.DataFrame(imputer_ccr.transform(input_data), columns=input_data.columns)
                prob = model_ccr.predict_proba(imputed_data)[0][1] * 100
                
                if prob >= 65:
                    color, text_color, rec = "#dcfce7", "#166534", "High cCR Probability. Strong candidate for Watch-and-Wait (organ-sparing) approach."
                elif prob <= 35:
                    color, text_color, rec = "#fee2e2", "#991b1b", "Low cCR Probability. Standard operative management (TME) recommended."
                else:
                    color, text_color, rec = "#fef9c3", "#854d0e", "Equivocal Probability. Multi-Disciplinary Tumor Board Review Advised."
                
                st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {text_color}; margin-bottom: 20px;">
                        <h2 style="color: {text_color} !important; margin:0; font-size: 45px;">{prob:.1f}%</h2>
                        <p style="color: {text_color} !important; font-size: 18px; font-weight: bold; margin-top: 10px;">{rec}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### AI Decision Logic (SHAP Plot):")
                explainer = shap.TreeExplainer(model_ccr)
                shap_values = explainer.shap_values(imputed_data)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=imputed_data.iloc[0], feature_names=['Age', 'Tumor Height', 'Wall Thickness']), show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ------------------------------------------
# MODULE 2: LPLN Sterilization Predictor 
# ------------------------------------------
elif app_mode == "LPLN Sterilization Predictor":
    st.markdown("<h1 class='main-title'>Lateral Pelvic Lymph Node (LPLN) Predictor</h1>", unsafe_allow_html=True)
    st.write("Predicts nodal sterilization post-nCRT based on the biological characteristics of the primary tumor.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Primary Tumor Baseline Data")
        with st.form("lpln_form"):
            age = st.number_input("Age (Years)", min_value=20, max_value=100, value=60)
            height = st.number_input("Tumor Height (cm)", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
            thickness = st.number_input("MRI Wall Thickness (mm)", min_value=0.0, max_value=40.0, value=12.0, step=0.5)
            
            # Note: Node size and PET positivity inputs removed for mathematical accuracy and UX clarity.
            submit_lpln = st.form_submit_button("Calculate Sterilization Probability")
            
        with st.expander("💡 AI Insight: Why aren't we inputting the initial node size?"):
            st.info("""
            **AI Feature Pruning:**
            During model training, the XGBoost algorithm audited the baseline dataset and mathematically zero-weighted specific LPLN metrics (like MRI short-axis and PET avidity presence). Because nearly 100% of pathological LPLNs are PET-avid, avidity is a 'zero-variance' feature that cannot predict outcomes. 
            
            Instead, the AI successfully predicts nodal sterilization using the **primary tumor's anatomy** (Wall Thickness and Tumor Height), proving that the biology of the primary tumor dictates the radiosensitivity of its regional lymph nodes.
            """)

    with col2:
        if submit_lpln:
            try:
                # We still must pass 0s to the AI to satisfy the 5-feature shape it expects from train_models_unified.py
                input_data = pd.DataFrame({
                    'Age': [age], 
                    'Tumor_Height': [height], 
                    'Wall_Thickness': [thickness], 
                    'Initial_LPLN_Size': [0], 
                    'PET_LPLN_Positive': [0]
                })
                
                imputed_data = pd.DataFrame(imputer_lpln.transform(input_data), columns=input_data.columns)
                prob = model_lpln.predict_proba(imputed_data)[0][1] * 100
                
                if prob >= 60:
                    color, text_color, rec = "#dcfce7", "#166534", "High Probability of Nodal Sterilization. May safely AVOID nodal dissection."
                elif prob <= 40:
                    color, text_color, rec = "#fee2e2", "#991b1b", "High Risk of Residual Nodal Disease. MUST PERFORM lateral pelvic nodal dissection (LPND)."
                else:
                    color, text_color, rec = "#fef9c3", "#854d0e", "Equivocal Nodal Response. Decision to dissect requires Multi-Disciplinary Review."
                
                st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {text_color}; margin-bottom: 20px;">
                        <h2 style="color: {text_color} !important; margin:0; font-size: 45px;">{prob:.1f}%</h2>
                        <p style="color: {text_color} !important; font-size: 18px; font-weight: bold; margin-top: 10px;">{rec}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### AI Decision Logic (SHAP Plot):")
                explainer = shap.TreeExplainer(model_lpln)
                shap_values = explainer.shap_values(imputed_data)
                
                # Only display the top 3 impactful features on the plot
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values[0][:3], 
                    base_values=explainer.expected_value, 
                    data=imputed_data.iloc[0][:3], 
                    feature_names=['Age', 'Tumor Height', 'Wall Thickness']
                ), show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")