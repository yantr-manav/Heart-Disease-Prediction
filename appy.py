import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
    }
    h1, h2, h3 {
        color: #8B0000;
    }

    .stButton>button {
        background-color: #8B0000;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #660000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    try:
        # model_path = Path("C:/Users/Saivamshi/Music/Sai/Coding/ML_project/data/xgb_heart_model.pkl")
        model_path = Path("model.pkl")
        if not model_path.exists():
            st.error("⚠️ Model file .pkl not found. Please ensure it's in the same directory.")
            return None
        
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# --- Main Title ---

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
<div  style='background-color: rgba(255,0,25,0.8); padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
    <p style='font-size: 16px; margin: 0;'>
        Enter patient details below and get an AI-powered prediction with confidence probability 
        and detailed feature explanations using SHAP values.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Feature Descriptions ---
feature_descriptions = {
    "Age": "Patient's age in years",
    "Sex": "Biological sex of the patient",
    "Chest Pain Type": "0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic",
    "Resting BP": "Resting blood pressure in mm Hg",
    "Cholesterol": "Serum cholesterol in mg/dl",
    "Fasting Blood Sugar": "1 if fasting blood sugar > 120 mg/dl, else 0",
    "Resting ECG": "0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy",
    "Max Heart Rate": "Maximum heart rate achieved during exercise",
    "Exercise Induced Angina": "1: Yes, 0: No",
    "ST Depression": "ST depression induced by exercise relative to rest",
    "Slope": "Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)",
    "Major Vessels": "Number of major vessels colored by fluoroscopy (0-3)",
    "Thalassemia": "0: Normal, 1: Fixed Defect, 2: Reversible Defect"
}

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📋 Patient Details")
    
    # Demographic information
    st.subheader("Demographic Information")
    age = st.slider("Age", 20, 100, 50, help=feature_descriptions["Age"])
    sex = st.selectbox("Sex", ("Male", "Female"), help=feature_descriptions["Sex"])
    
    # Clinical measurements
    st.subheader("Clinical Measurements")
    col1a, col1b = st.columns(2)
    with col1a:
        trestbps = st.number_input(
            "Resting BP (mm Hg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
            help=feature_descriptions["Resting BP"]
        )
        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200,
            step=1,
            help=feature_descriptions["Cholesterol"]
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            (0, 1),
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=feature_descriptions["Fasting Blood Sugar"]
        )
    
    with col1b:
        thalach = st.number_input(
            "Max Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
            help=feature_descriptions["Max Heart Rate"]
        )
        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
            help=feature_descriptions["ST Depression"]
        )
        exang = st.selectbox(
            "Exercise Induced Angina",
            (0, 1),
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=feature_descriptions["Exercise Induced Angina"]
        )
    
    # Diagnostic results
    st.subheader("Diagnostic Results")
    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
        help=feature_descriptions["Chest Pain Type"]
    )
    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2],
        format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
        help=feature_descriptions["Resting ECG"]
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [0, 1, 2],
        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
        help=feature_descriptions["Slope"]
    )
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3],
        help=feature_descriptions["Major Vessels"]
    )
    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2],
        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
        help=feature_descriptions["Thalassemia"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    predict_button = st.button("🔍 Predict Heart Disease", use_container_width=True)

# Prepare input features
sex_val = 1 if sex == "Male" else 0
features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# --- Prediction & Results ---
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🎯 Prediction Results")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check the model file.")
    elif predict_button:
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                prob_disease = proba[1]
                prob_healthy = proba[0]
            else:
                prob_disease = None
                prob_healthy = None
            
            # Display result
            if prediction == 1:
                st.markdown("""
                <div class='error-box'>
                    <h3 style='margin:0; color:#dc3545;'>🚨 High Risk of Heart Disease</h3>
                    <p style='margin:10px 0 0 0;'>The model indicates a high likelihood of heart disease. 
                    Please consult with a healthcare professional immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-box'>
                    <h3 style='margin:0; color:#28a745;'>✅ Low Risk of Heart Disease</h3>
                    <p style='margin:10px 0 0 0; color:#000000'>The model indicates a low likelihood of heart disease. 
                    Continue maintaining a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display probability
            if prob_disease is not None:
                st.markdown("### Prediction Confidence")
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Healthy Probability", f"{prob_healthy:.1%}")
                with col2b:
                    st.metric("Disease Probability", f"{prob_disease:.1%}")
                
                # Probability bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(
                    ["Healthy", "Heart Disease"],
                    [prob_healthy, prob_disease],
                    color=["#28a745", "#dc3545"],
                    alpha=0.7
                )
                ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
                ax.set_title("Prediction Confidence", fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.1%}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Feature importance
            if hasattr(model, "feature_importances_"):
                st.markdown("### 📊 Feature Importance")
                st.markdown("*Shows which features are most important for the model's predictions overall*")
                
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors = plt.cm.RdYlGn_r(importances[sorted_idx] / importances.max())
                ax2.barh(
                    [feature_names[i] for i in sorted_idx],
                    importances[sorted_idx],
                    color=colors
                )
                ax2.set_xlabel("Importance", fontsize=12, fontweight='bold')
                ax2.set_title("Feature Importance in Model", fontsize=14, fontweight='bold')
                ax2.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            # --- SHAP Explanation ---
            st.markdown("### 🔬 Feature Contribution Analysis (SHAP)")
            st.markdown("*Shows how each feature influenced THIS specific prediction*")
            
            with st.spinner("Calculating SHAP values..."):
                try:
                    explainer = shap.Explainer(model)
                    shap_values = explainer(features)
                    
                    # Handle different SHAP output formats
                    # For binary classification, we want the positive class (disease) SHAP values
                    if len(shap_values.shape) == 3:
                        # Multi-output format: (samples, features, classes)
                        shap_explanation = shap_values[:, :, 1][0]  # Get class 1 (disease) for first sample
                    elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 2:
                        # Standard format for binary classification
                        shap_explanation = shap_values[0]
                    else:
                        # Fallback
                        shap_explanation = shap_values[0]
                    
                    # Waterfall plot
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_explanation, max_display=13, show=False)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()
                    
                    st.info("""
                    **How to read this chart:**
                    - Red bars push the prediction toward heart disease
                    - Blue bars push the prediction toward healthy
                    - Longer bars = stronger influence
                    """)
                    
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {str(e)}")
                    # Show bar plot as fallback
                    try:
                        st.markdown("#### Alternative View: Feature Contributions")
                        fig4, ax4 = plt.subplots(figsize=(8, 6))
                        shap.plots.bar(shap_values[0], max_display=13, show=False)
                        plt.tight_layout()
                        st.pyplot(fig4)
                        plt.close()
                    except:
                        st.error("Unable to generate SHAP visualizations")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👈 Enter patient details and click 'Predict Heart Disease' to see results")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("ℹ️ About This App")
st.sidebar.info("""
**Heart Disease Prediction Dashboard**

This application uses machine learning (XGBoost) to predict the likelihood of heart disease based on clinical parameters.

**Dataset:** UCI Heart Disease Dataset

**Model Features:** 13 clinical parameters including age, sex, chest pain type, blood pressure, cholesterol, and more.

**Interpretability:** Uses SHAP (SHapley Additive exPlanations) to explain individual predictions.
""")

st.sidebar.header("⚠️ Disclaimer")
st.sidebar.warning("""
This tool is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

Always consult with a qualified healthcare provider for medical decisions.
""")

st.sidebar.header("📚 Resources")
st.sidebar.markdown("""
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit | Machine Learning for Healthcare</p>
</div>
""", unsafe_allow_html=True)