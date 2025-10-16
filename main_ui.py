import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* General body styling */
    body {
        background-color: #f0f2f6; /* A light grey for a clean look */
    }
    /* Main title */
    .title {
        font-size: 3em;
        font-weight: bold;
        color: #B22222; /* A deep red color */
        text-align: center;
        margin-bottom: 20px;
    }
    /* Cards for grouping content */
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e6e6e6;
    }
    /* Style for the main button */
    .stButton>button {
        background-color: #B22222;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #8B0000;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    /* Styling for the result boxes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border-left: 6px solid;
    }
    .success-box {
        background-color: #e8f5e9;
        border-color: #4CAF50;
        color: #1b5e20;
    }
    .error-box {
        background-color: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    /* Placeholder text style */
    .placeholder {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px dashed #cccccc;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = joblib.load("./data/xgb_heart_model.pkl") # Assuming files are in the root
        preprocessor = joblib.load("./data/preprocessor.pkl")
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor file not found. Please ensure 'xgb_heart_model.pkl' and 'preprocessor.pkl' are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

model, preprocessor = load_model_and_preprocessor()

# --- Feature Columns (for DataFrame creation) ---
numerical_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal'] # Reordered to match input form

# --- UI Layout ---
st.markdown("<h1 class='title'>❤️ Heart Disease Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Enter patient details to predict the likelihood of heart disease using an XGBoost machine learning model.</p>", unsafe_allow_html=True)


col1, col2 = st.columns([1, 1], gap="large")

# --- Input Column ---
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📋 Patient Input Form")

    # Grouped inputs for better organization
    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age", 20, 100, 50, help="Patient's age in years.")
        sex = st.radio("Sex", ("Male", "Female"), horizontal=True, help="Patient's gender.")

    with st.expander("🩺 Vital Signs & Lab Results", expanded=True):
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Systolic blood pressure at rest.")
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200, help="Total cholesterol level.")
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", (1, 0), format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, help="Is the fasting blood sugar level greater than 120 mg/dl?")

    with st.expander("📈 Clinical Data", expanded=True):
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina", (1, 0), format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    st.markdown('</div>', unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)


# --- Prediction and Results Column ---
with col2:
    if predict_button and model is not None and preprocessor is not None:
        # --- Data Preparation ---
        sex_val = 1 if sex == "Male" else 0
        user_input_dict = {
            'age': age, 'sex': sex_val, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'cp': cp,
            'restecg': restecg, 'slope': slope, 'ca': ca, 'thal': thal
        }
        # Ensure column order is correct
        user_input = pd.DataFrame([user_input_dict], columns=numerical_cols + categorical_cols)

        # --- Prediction Logic ---
        X_transformed = preprocessor.transform(user_input)
        prediction = model.predict(X_transformed)[0]
        prob_healthy, prob_disease = model.predict_proba(X_transformed)[0]
        
        # Store results in session state
        st.session_state['prediction'] = prediction
        st.session_state['probabilities'] = (prob_healthy, prob_disease)
        st.session_state['transformed_data'] = X_transformed
        st.session_state['user_input_df'] = user_input # Store for feature names

    # --- Display Results from Session State ---
    if 'prediction' in st.session_state:
        st.header("📊 Prediction Analysis")

        # Display result box
        if st.session_state['prediction'] == 1:
            st.markdown("<div class='result-box error-box'><h3>🚨 High Risk of Heart Disease</h3><p>The model indicates a high likelihood of heart disease. Please consult with a healthcare professional immediately.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box success-box'><h3>✅ Low Risk of Heart Disease</h3><p>The model indicates a low likelihood of heart disease. Continue maintaining a healthy lifestyle.</p></div>", unsafe_allow_html=True)
        
        # Display probability gauge
        st.subheader("Confidence Score")
        prob_healthy, prob_disease = st.session_state['probabilities']
        
        # Create a horizontal bar chart for probabilities
        prob_df = pd.DataFrame({'Probability': [prob_disease, prob_healthy]}, index=['Disease', 'Healthy'])
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['#f44336', '#4CAF50']
        prob_df['Probability'].T.plot(kind='barh', stacked=True, ax=ax, color=colors, legend=False)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add labels on the bars
        ax.text(prob_disease / 2, 0, f"{prob_disease:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        ax.text(prob_disease + (prob_healthy / 2), 0, f"{prob_healthy:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        st.pyplot(fig)
        
        # # --- SHAP Explanation ---
        # st.subheader("🔬 Feature Contribution Analysis (SHAP)")
        # try:
        #     explainer = shap.Explainer(model)
        #     X_transformed_df = pd.DataFrame(st.session_state['transformed_data'], columns=st.session_state['user_input_df'].columns)
        #     shap_values = explainer(X_transformed_df)
            
        #     fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
        #     shap.plots.waterfall(shap_values[0], max_display=20, show=False)
        #     fig_shap.tight_layout()
        #     st.pyplot(fig_shap)
            
        #     st.info("""
        #     **How to read this chart:** This waterfall plot shows how each feature contributed to pushing the model's prediction from a baseline value to the final output.
        #     - **Red bars** represent features that increase the risk of heart disease.
        #     - **Blue bars** represent features that decrease the risk.
        #     - The length of the bar indicates the magnitude of the feature's impact.
        #     """)
        # except Exception as e:
        #     st.warning(f"Could not generate SHAP plot: {str(e)}")
            
    else:
        # Initial placeholder
        st.markdown(
            "<div class='placeholder'><h2>Awaiting Input</h2><p>Your prediction results and analysis will appear here once you fill out the patient form and click the predict button.</p></div>", 
            unsafe_allow_html=True
        )