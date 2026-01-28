import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model(model_path=r'C:\Users\dell\Desktop\self_made_model\model.pkl'):
    """Loads the pre-trained model from a pickle file."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run the training script first.")
        return None
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- App Title and Description ---
st.title("🍷 Wine Quality Prediction App")
st.write(
    "This app predicts the quality of red wine (on a scale of 3-8) based on its chemical properties. "
    "Adjust the sliders below to input the wine's features and click 'Predict' to see the result."
)

# --- User Input via Sliders in a Form ---
if model:
    with st.form("wine_features_form"):
        st.header("Input Wine Features")
        
        # Creating columns for a better layout
        col1, col2 = st.columns(2)

        with col1:
            fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.3, 0.1)
            volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.27, 0.01)
            residual_sugar = st.slider("Residual Sugar", 0.9, 16.0, 2.5, 0.1)
            chlorides = st.slider("Chlorides", 0.01, 0.62, 0.08, 0.001)
            free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 16)

        with col2:
            total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
            density = st.slider("Density", 0.990, 1.004, 0.996, 0.0001, format="%.5f")
            pH = st.slider("pH", 2.7, 4.0, 3.3, 0.01)
            sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65, 0.01)
            alcohol = st.slider("Alcohol", 8.0, 15.0, 10.4, 0.1)

        submitted = st.form_submit_button("Predict Quality")

    if submitted:
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                                  columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                           'pH', 'sulphates', 'alcohol'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        st.success(f"**Predicted Wine Quality:** {prediction[0]}")

