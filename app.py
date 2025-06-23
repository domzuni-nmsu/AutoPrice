# app.py (Corrected)
import flask
from flask import Flask, render_template, request
import joblib
import torch
import pandas as pd
import numpy as np
import os
import json

from src.model import PricePredictor

app = Flask(__name__, template_folder='app/templates')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(APP_ROOT, 'saved_models')

# --- Load All Artifacts ---
# In app.py

# --- Load All Artifacts (Data, Preprocessor, Model) ---
try:
    # Load raw data for dropdowns
    RAW_DATA_PATH = os.path.join(APP_ROOT, 'data', 'raw', 'used_cars.csv')
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    # --- Create data for ALL dropdowns ---
    brands = sorted(df_raw['brand'].dropna().unique())
    fuel_types = sorted(df_raw['fuel_type'].dropna().unique())

    # Create data for Brand -> Model mapping
    df_models_clean = df_raw.dropna(subset=['brand', 'model'])
    brand_to_models_map = df_models_clean.groupby('brand')['model'].unique().apply(lambda x: sorted(x)).to_dict()

    # --- NEW, CORRECTED LOGIC for the Transmission Map ---
    df_transmissions_clean = df_raw.dropna(subset=['brand', 'model', 'transmission'])
    brand_model_to_trans_map = {}
    for (brand, model), group in df_transmissions_clean.groupby(['brand', 'model']):
        if brand not in brand_model_to_trans_map:
            brand_model_to_trans_map[brand] = {}
        transmissions = sorted(group['transmission'].unique())
        brand_model_to_trans_map[brand][model] = transmissions
    # --- END OF NEW LOGIC ---

    # Load the preprocessor and training columns
    PREPROCESSOR_PATH = os.path.join(SAVED_MODELS_DIR, 'preprocessor.joblib')
    TRAINING_COLS_PATH = os.path.join(SAVED_MODELS_DIR, 'training_columns.joblib')
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    training_columns = joblib.load(TRAINING_COLS_PATH)
    
    # Load the model
    MODEL_INPUT_FEATURES = len(preprocessor.get_feature_names_out())
    MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'price_predictor_model_v1.pth')
    model = PricePredictor(num_input_features=MODEL_INPUT_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("All artifacts loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR loading artifacts: {e}")
    # Set defaults so the app can still run and show an error
    brands, fuel_types, brand_to_models_map, brand_model_to_trans_map = [], [], {}, {}
    preprocessor, model, training_columns = None, None, []

# --- (Rest of your app.py: routes for home() and predict() remain the same) ---
@app.route('/')
def home():
    # ...
    return render_template('index.html', 
                           brands=brands,
                           fuel_types=fuel_types, 
                           brand_to_models_map=brand_to_models_map,
                           brand_model_to_trans_map=brand_model_to_trans_map)

@app.route('/predict', methods=['POST'])
def predict():
    # ... (This logic should now work correctly, but we need to fix the alignment)
    if not (preprocessor and model):
        return render_template('results.html', prediction="Error: Model or preprocessor not loaded.")
    try:
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])
        # ... your form data cleaning ...
        input_df['milage'] = pd.to_numeric(input_df['milage'], errors='coerce')
        input_df['model_year'] = pd.to_numeric(input_df['model_year'], errors='coerce')
        input_df['car_age'] = 2025 - input_df['model_year']

        # Preprocess using the loaded preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # Convert to Tensor
        input_tensor = torch.tensor(processed_input.toarray(), dtype=torch.float32)

        # Predict
        with torch.no_grad():
            prediction_log = model(input_tensor)
        
        log_price = prediction_log.item()
        predicted_price = np.expm1(log_price)
        formatted_price = f"${predicted_price:,.2f}"
        return render_template('results.html', prediction=formatted_price)
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return render_template('results.html', prediction=f"Error: Could not make a prediction. Details: {e}")
        
if __name__ == '__main__':
    app.run(debug=True)