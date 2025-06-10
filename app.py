# app.py
import flask
from flask import Flask, render_template, request
import joblib
import torch
import pandas as pd
import numpy as np
import os

# Import your model class
from src.model import PricePredictor

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='app/templates')

# --- Load Model and Preprocessor ---

# Get the absolute path of the directory where this script is located
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(APP_ROOT, 'saved_models')

# Define paths to the saved artifacts
PREPROCESSOR_PATH = os.path.join(SAVED_MODELS_DIR, 'preprocessor.joblib')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'price_predictor_model_v1.pth')

# Load the preprocessor
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"Error: Preprocessor file not found at {PREPROCESSOR_PATH}.")
    preprocessor = None

# Determine model input size from the preprocessor if possible
# This is crucial for initializing the model correctly
try:
    # Access the number of features the OneHotEncoder was trained on
    # Note: This is an estimation and depends on the scikit-learn version and how the preprocessor was built.
    # It might need adjustment.
    cat_features_len = preprocessor.named_transformers_['cat'].get_feature_names_out().shape[0]
    num_features_len = len(preprocessor.named_transformers_['num'].feature_names_in_)
    MODEL_INPUT_FEATURES = num_features_len + cat_features_len
    print(f"Determined model input features: {MODEL_INPUT_FEATURES}")
except Exception as e:
    print(f"Could not automatically determine input features: {e}")
    # FALLBACK: Manually set this to the number from your training script output
    MODEL_INPUT_FEATURES = 150 # Replace with the actual number if auto-detection fails

# Load the PyTorch model
try:
    # Initialize the model architecture
    model = PricePredictor(num_input_features=MODEL_INPUT_FEATURES)
    # Load the trained weights (state_dict)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set the model to evaluation mode
    print("PyTorch model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}.")
    model = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    model = None


# --- Define Routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST' and preprocessor and model:
        try:
            # Get data from the form
            form_data = request.form.to_dict()
            
            # --- IMPORTANT: Create a DataFrame from form data ---
            # The column names MUST EXACTLY MATCH the ones used for training the preprocessor
            input_df = pd.DataFrame([form_data])

            # Convert data types to match the training data
            # This is a critical step
            input_df['milage'] = pd.to_numeric(input_df['milage'], errors='coerce')
            input_df['model_year'] = pd.to_numeric(input_df['model_year'], errors='coerce')
            
            # Re-create 'car_age' and 'engine_displacement' as done in training
            input_df['car_age'] = 2025 - input_df['model_year']
            # For simplicity, we'll assume a common engine size or pass it from form
            input_df['engine_displacement'] = 3.0 # Example default, adjust as needed

            # Ensure all required columns are present, even if not in the form
            required_cols = preprocessor.feature_names_in_
            for col in required_cols:
                if col not in input_df.columns:
                    # Provide a default value for missing columns.
                    # This could be the mode, or a common value.
                    input_df[col] = 'other' # A safe default for categorical
                    if col in ['milage', 'model_year', 'car_age', 'engine_displacement']:
                         input_df[col] = 0 # A safe default for numerical

            # Reorder columns to match the preprocessor's expectation
            input_df = input_df[required_cols]

            # 1. Preprocess the input data
            processed_input = preprocessor.transform(input_df)

            # 2. Convert to PyTorch Tensor
            input_tensor = torch.tensor(processed_input.toarray() if hasattr(processed_input, "toarray") else processed_input, dtype=torch.float32)

            # 3. Make a prediction
            with torch.no_grad():
                prediction = model(input_tensor)
            
            predicted_price = prediction.item()
            
            # Format the price for display
            formatted_price = f"${predicted_price:,.2f}"

            # Render the results page
            return render_template('results.html', prediction=formatted_price)

        except Exception as e:
            # Handle errors gracefully
            print(f"An error occurred during prediction: {e}")
            return render_template('results.html', prediction=f"Error: Could not make a prediction. Please check the logs. Details: {e}")
            
    # If something went wrong (no model/preprocessor or not a POST request)
    return "Prediction service is unavailable.", 500


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
