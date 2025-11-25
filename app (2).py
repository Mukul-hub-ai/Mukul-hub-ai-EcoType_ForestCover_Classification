import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Set page configuration
st.set_page_config(
    page_title="Forest Cover Type Predictor",
    page_icon="🌲",
    layout="centered"
)

# Load the trained model and label encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        # Load model (joblib format)
        model = joblib.load('best_tuned_knn_model.pkl')
        st.success("Model loaded successfully!")
        
        # Load label encoder (pickle format)
        with open("label_enocder.pkl", "rb") as f:
            le = pickle.load(f)
        st.success("Label encoder loaded successfully!")
        
        return model, le
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None, None

# Manual cover type mapping
cover_type_mapping = {
    0: "Spruce/Fir 🌲",
    1: "Lodgepole Pine 🪵", 
    2: "Ponderosa Pine 🌳",
    3: "Cottonwood/Willow 🍃",
    4: "Aspen 🍂",
    5: "Douglas-fir 🎄",
    6: "Krummholz 🌿"
}

def main():
    # Main heading - center aligned
    st.markdown("""
    <h1 style='text-align: center; color: #2E8B57; margin-bottom: 30px;'>
        🌲 Forest Cover Type Prediction System
    </h1>
    """, unsafe_allow_html=True)
    
    # Load model and encoder
    model, le = load_model_and_encoder()
    if model is None or le is None:
        st.error("Please ensure 'best_tuned_knn_model.pkl' and 'label_enocder.pkl' are in the same directory.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("Enter Feature Values")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            elevation = st.number_input("Elevation (meters)", min_value=0, max_value=4000, value=2500, step=1)
            horiz_roadways = st.number_input("Horizontal Distance to Roadways", min_value=0, value=2000, step=1)
            horiz_firepoints = st.number_input("Horizontal Distance to Fire Points", min_value=0, value=2000, step=1)
            fire_to_hydrology_ratio = st.number_input("Fire to Hydrology Ratio", min_value=0.0, value=6.67, step=0.1)
            road_to_hydrology_ratio = st.number_input("Road to Hydrology Ratio", min_value=0.0, value=6.67, step=0.1)
            hydrology_distance_ratio = st.number_input("Hydrology Distance Ratio", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            wilderness_area_1 = st.selectbox("Wilderness Area 1", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            vert_hydrology = st.number_input("Vertical Distance to Hydrology", min_value=-100, max_value=100, value=0, step=1)
            horiz_hydrology = st.number_input("Horizontal Distance to Hydrology", min_value=0, value=300, step=1)
            
        with col2:
            aspect = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=180, step=1)
            hillshade_noon = st.number_input("Hillshade Noon", min_value=0, max_value=255, value=220, step=1)
            wilderness_area_4 = st.selectbox("Wilderness Area 4", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hillshade_3pm = st.number_input("Hillshade 3pm", min_value=0, max_value=255, value=150, step=1)
            hillshade_noon_evening_diff = st.number_input("Hillshade Noon-Evening Difference", min_value=-255, max_value=255, value=70, step=1)
            hillshade_morning_noon_diff = st.number_input("Hillshade Morning-Noon Difference", min_value=-255, max_value=255, value=-20, step=1)
            hillshade_daily_variation = st.number_input("Hillshade Daily Variation", min_value=0, max_value=255, value=50, step=1)
            hillshade_9am = st.number_input("Hillshade 9am", min_value=0, max_value=255, value=200, step=1)
            slope = st.number_input("Slope (degrees)", min_value=0, max_value=90, value=15, step=1)
            wilderness_area_3 = st.selectbox("Wilderness Area 3", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            soil_type_12 = st.selectbox("Soil Type 12", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Predict button
        predict_btn = st.form_submit_button("Predict Forest Cover Type", use_container_width=True)
    
    # Prediction logic
    if predict_btn:
        try:
            # Create feature array in EXACT same order as your X_train.columns
            features = np.array([[
                elevation, horiz_roadways, horiz_firepoints, fire_to_hydrology_ratio,
                road_to_hydrology_ratio, hydrology_distance_ratio, wilderness_area_1,
                vert_hydrology, horiz_hydrology, aspect, hillshade_noon, wilderness_area_4,
                hillshade_3pm, hillshade_noon_evening_diff, hillshade_morning_noon_diff,
                hillshade_daily_variation, hillshade_9am, slope, wilderness_area_3, soil_type_12
            ]])
            
            # Make prediction
            prediction_encoded = model.predict(features)[0]
            
            # Use manual mapping instead of label encoder
            prediction_name = cover_type_mapping.get(prediction_encoded, f"Unknown Type ({prediction_encoded})")
            
            # Display result
            st.markdown("---")
            st.markdown("Prediction Result")
            
            st.success(f"""
            **Predicted Forest Cover Type:** 
            # {prediction_name}
            """)
            
            # Additional info
            st.info(f"""
            **Details:**
            - Predicted Class: {prediction_encoded}
            - Forest Type: {prediction_name}
            - Model: Tuned K-Nearest Neighbors
            """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Forest Cover Type Prediction System | Machine Learning Model"
    "</div>", 
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()