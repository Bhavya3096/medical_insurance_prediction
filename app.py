import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("### Predict your medical insurance charges based on personal information")
st.markdown("---")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("insurance_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'insurance_model.pkl' not found. Please ensure the model is trained and saved.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

pipeline = load_model()

if pipeline is not None:
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Enter your age in years"
        )
        
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Enter your BMI value"
        )
        
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Enter the number of children/dependents"
        )
    
    with col2:
        st.subheader("🔍 Additional Details")
        sex = st.selectbox(
            "Sex",
            options=["male", "female"],
            help="Select your sex"
        )
        
        smoker = st.selectbox(
            "Smoking Status",
            options=["no", "yes"],
            help="Are you a smoker?"
        )
        
        region = st.selectbox(
            "Region",
            options=["northeast", "northwest", "southeast", "southwest"],
            help="Select your residential region"
        )
    
    st.markdown("---")
    
    # Prediction button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    with col_button2:
        predict_button = st.button("🔮 Predict Insurance Charges")
    
    # Prediction logic
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region]
        })
        
        # Make prediction
        with st.spinner("Calculating your insurance charges..."):
            try:
                prediction = pipeline.predict(input_data)[0]
                
                # Display prediction
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                
                # Success message with prediction
                st.success(f"✅ **Predicted Insurance Charges: ₹{prediction:,.2f}**")
                
                # Additional info box
                st.info(f"""
                **Summary of Your Details:**
                - Age: {age} years
                - BMI: {bmi}
                - Children: {children}
                - Sex: {sex.capitalize()}
                - Smoker: {smoker.capitalize()}
                - Region: {region.capitalize()}
                """)
                
                # Visual representation
                col_vis1, col_vis2, col_vis3 = st.columns(3)
                with col_vis1:
                    st.metric("Predicted Charges", f"₹{prediction:,.2f}")
                with col_vis2:
                    st.metric("Monthly Estimate", f"₹{prediction/12:,.2f}")
                with col_vis3:
                    st.metric("Daily Estimate", f"₹{prediction/365:,.2f}")
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💡 This prediction is based on machine learning models and should be used for reference only.</p>
        <p>🔒 Your data is processed locally and not stored anywhere.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Please train and save the model first before using this app.")