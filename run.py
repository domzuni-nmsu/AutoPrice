import flask
from flask import Flask, render_template, request
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json

# --- Define Model Architecture ---
# This MUST match the architecture of your saved .pth file EXACTLY.
class PricePredictor(nn.Module):
    def __init__(self, num_input_features, dropout_rate=0.4):
        super(PricePredictor, self).__init__()
        self.layer_1 = nn.Linear(num_input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.layer_2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.layer_3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(64, 1)
        
    def forward(self, x):
        # CORRECTED ORDER: Linear -> ReLU -> BatchNorm -> Dropout
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.layer_2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.layer_3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.output_layer(x)
        return x

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Define Global Variables and Load Artifacts ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(APP_ROOT, 'saved_models')
DATA_DIR = os.path.join(APP_ROOT, 'data', 'raw')

# Initialize variables to hold artifacts
preprocessor = None
model = None
training_columns = []
brands = []
fuel_types = []
brand_to_models_map = {}
brand_model_to_trans_map = {}

try:
    # --- Load Data for Dropdowns ---
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'used_cars.csv')
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    brands = sorted(df_raw['brand'].dropna().unique())
    fuel_types = sorted(df_raw['fuel_type'].dropna().unique())

    df_models_clean = df_raw.dropna(subset=['brand', 'model'])
    brand_to_models_map = df_models_clean.groupby('brand')['model'].unique().apply(lambda x: sorted(x)).to_dict()

    df_transmissions_clean = df_raw.dropna(subset=['brand', 'model', 'transmission'])
    for (brand, model_name), group in df_transmissions_clean.groupby(['brand', 'model']):
        if brand not in brand_model_to_trans_map:
            brand_model_to_trans_map[brand] = {}
        transmissions = sorted(group['transmission'].unique())
        brand_model_to_trans_map[brand][model_name] = transmissions
    
    # --- Load ML Artifacts ---
    PREPROCESSOR_PATH = os.path.join(SAVED_MODELS_DIR, 'preprocessor.joblib')
    TRAINING_COLS_PATH = os.path.join(SAVED_MODELS_DIR, 'training_columns.joblib')
    MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'price_predictor_refined_v1.pth')

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    training_columns = joblib.load(TRAINING_COLS_PATH)
    
    MODEL_INPUT_FEATURES = len(preprocessor.get_feature_names_out())
    
    model = PricePredictor(num_input_features=MODEL_INPUT_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("✅ All artifacts loaded successfully.")

except Exception as e:
    print(f"❌ FATAL ERROR loading artifacts: {e}")
    print("   The app will run but predictions will fail and dropdowns will be empty.")


# --- Define Routes ---
@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html', 
                           brands=brands,
                           fuel_types=fuel_types, 
                           brand_to_models_map=brand_to_models_map,
                           brand_model_to_trans_map=brand_model_to_trans_map)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if not all([preprocessor, model, training_columns]):
        return render_template('results.html', prediction="Error: Model or preprocessor not loaded on the server.")

    try:
        # --- Get and Clean Form Data ---
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])
        
        input_df['milage'] = pd.to_numeric(input_df['milage'], errors='coerce')
        input_df['model_year'] = pd.to_numeric(input_df['model_year'], errors='coerce')
        
        # --- Feature Engineering (must EXACTLY match training) ---
        input_df['car_age'] = 2025 - input_df['model_year']
        
        # A safe default is the median or mean from the training data. Let's use a common value like 2.5
        if 'engine_displacement' in training_columns:
             input_df['engine_displacement'] = 2.5 # Add a reasonable default

        # THIS IS THE CRITICAL FIX: Add the missing 'miles_per_year' feature creation
        if 'miles_per_year' in training_columns:
            # Add a small number to avoid division by zero for new cars (age=0)
            input_df['miles_per_year'] = input_df['milage'] / (input_df['car_age'] + 1e-6)
        
        # --- Align Columns ---
        aligned_df = pd.DataFrame(columns=training_columns)
        aligned_df = pd.concat([aligned_df, input_df], ignore_index=True, sort=False).fillna(0)
        final_input_df = aligned_df[training_columns]

        # --- Preprocess and Predict ---
        processed_input = preprocessor.transform(final_input_df)
        input_tensor = torch.tensor(processed_input.toarray(), dtype=torch.float32)
        
        with torch.no_grad():
            prediction_log = model(input_tensor)
        
        # --- Format Output ---
        predicted_price = np.expm1(prediction_log.item())
        formatted_price = f"${predicted_price:,.2f}"
        
        return render_template('results.html', prediction=formatted_price)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return render_template('results.html', prediction=f"Error: Could not make a prediction. Details: {e}")

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
