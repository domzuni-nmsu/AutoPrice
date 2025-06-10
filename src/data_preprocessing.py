# src/data_preprocessing.py
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

# Define file paths using absolute paths for reliability
# Get the absolute path of the directory where this script is located (e.g., D:\AutoPrice\src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the project's root directory by going one level up (e.g., D:\AutoPrice)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Build absolute paths to the data and saved models directories
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'used_cars.csv')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
PREPROCESSOR_PATH = os.path.join(SAVED_MODELS_DIR, 'preprocessor.joblib')


def run_data_pipeline():
    """
    This function runs the entire data preparation pipeline:
    1. Loads raw data
    2. Cleans and engineers features
    3. Splits data
    4. Creates, fits, and saves the preprocessor
    5. Transforms data and saves final tensors
    """
    # --- 1. Load and Clean Raw Data ---
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Your proven cleaning logic
    if 'price' in df.columns and df['price'].dtype == 'object':
        df_temp_price = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df_temp_price, errors='coerce')

    if 'milage' in df.columns and df['milage'].dtype == 'object':
        df_temp_milage = df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False)
        df['milage'] = pd.to_numeric(df_temp_milage, errors='coerce')
    
    if 'engine' in df.columns and df['engine'].dtype == 'object':
        df['engine_displacement'] = df['engine'].str.extract(r'(\d\.?\d*)L').astype(float)
        df['engine_displacement'].fillna(df['engine_displacement'].median(), inplace=True)

    if 'model_year' in df.columns:
        current_year = 2025
        df['car_age'] = current_year - df['model_year']

    df.dropna(subset=['price', 'milage'], inplace=True)

    for col in ['fuel_type', 'transmission', 'accident', 'clean_title']:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("Data cleaning and feature engineering complete.")

    # --- 2. Define Features and Split Data ---
    feature_columns = [
        'milage', 'model_year', 'car_age', 'engine_displacement', 'brand', 
        'fuel_type', 'transmission', 'accident', 'clean_title'
    ]
    existing_feature_columns = [col for col in feature_columns if col in df.columns]
    X = df[existing_feature_columns]
    y = df['price']

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split complete.")

    # --- 3. Create, Fit, and SAVE the Preprocessor ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("Fitting the preprocessor...")
    preprocessor.fit(X_train)
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

    # --- 4. Apply the Preprocessor and Save Tensors ---
    print("Applying preprocessor and saving final tensors...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    torch.save(X_train_tensor, f'{PROCESSED_DATA_DIR}/X_train_tensor.pt')
    torch.save(y_train_tensor, f'{PROCESSED_DATA_DIR}/y_train_tensor.pt')
    torch.save(X_test_tensor, f'{PROCESSED_DATA_DIR}/X_test_tensor.pt')
    torch.save(y_test_tensor, f'{PROCESSED_DATA_DIR}/y_test_tensor.pt')
    print(f"Final Tensors saved to '{PROCESSED_DATA_DIR}' directory.")

if __name__ == '__main__':
    run_data_pipeline()
    print("\n--- DATA PREPROCESSING SCRIPT FINISHED ---")