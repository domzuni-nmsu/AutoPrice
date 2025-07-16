# src/create_artifacts.py
# This single script will create all your necessary files in perfect sync.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. DEFINE YOUR MODEL ARCHITECTURE ---
class PricePredictor(nn.Module):
    def __init__(self, num_input_features):
        super(PricePredictor, self).__init__()
        self.layer_1 = nn.Linear(num_input_features, 128)
        self.relu1 = nn.ReLU()
        self.layer_2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.output_layer(x)
        return x

# --- 2. DEFINE PATHS AND LOAD DATA ---
print("--- Starting Artifact Creation Script ---")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'used_cars.csv')

os.makedirs(SAVE_DIR, exist_ok=True) # Ensure save directory exists

try:
    df = pd.read_csv(RAW_DATA_PATH)
    print("Loaded raw data successfully.")

    # --- 3. COMPLETE DATA CLEANING & FEATURE ENGINEERING ---
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df['milage'] = pd.to_numeric(df['milage'].astype(str).str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df.dropna(subset=['price', 'milage'], inplace=True)
    
    df['car_age'] = 2025 - df['model_year']
    
    for col in ['fuel_type', 'transmission', 'accident', 'clean_title']:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    df['price'] = np.log1p(df['price'])
    print("Data cleaning and feature engineering complete.")

    # --- 4. PREPARE FOR PREPROCESSING ---
    feature_columns = ['milage', 'model_year', 'car_age', 'brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
    X = df[feature_columns]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )
    
    # --- 5. FIT PREPROCESSOR AND SAVE ARTIFACTS ---
    print("\n--- Fitting preprocessor and transforming data ---")
    preprocessor.fit(X_train)
    
    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get the final feature count AFTER transformation
    FINAL_FEATURE_COUNT = X_train_processed.shape[1]
    print(f"Final number of features after encoding: {FINAL_FEATURE_COUNT}")

    # Save the fitted preprocessor
    joblib.dump(preprocessor, os.path.join(SAVE_DIR, 'preprocessor.joblib'))
    print("Preprocessor saved successfully.")
    
    # Save the original column order that goes INTO the preprocessor
    # This is needed by the Flask app to create the input DataFrame correctly
    joblib.dump(X_train.columns.tolist(), os.path.join(SAVE_DIR, 'training_columns.joblib'))
    print("Training columns list saved successfully.")

    # --- 6. TRAIN AND SAVE THE MODEL ---
    print("\n--- Training New PyTorch Model ---")
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    # Initialize model with the CORRECT feature count
    model = PricePredictor(num_input_features=FINAL_FEATURE_COUNT)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # Save the final trained model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'price_predictor_model_v1.pth'))
    print("\n✅ New model trained and saved successfully!")

except Exception as e:
    print(f"\nAn error occurred during the script: {e}")