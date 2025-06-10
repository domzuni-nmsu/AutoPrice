# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Import your model definition
from model import PricePredictor

# Define file paths using absolute paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, 'price_predictor_model_v1.pth')

def run_training():
    """
    This function runs the model training pipeline:
    1. Loads preprocessed data (tensors)
    2. Instantiates and trains the model
    3. Evaluates the model
    4. Saves the trained model state
    """
    # --- 1. Load Preprocessed Data ---
    print("Loading preprocessed tensors...")
    try:
        X_train_tensor = torch.load(f'{PROCESSED_DATA_DIR}/X_train_tensor.pt')
        y_train_tensor = torch.load(f'{PROCESSED_DATA_DIR}/y_train_tensor.pt')
        X_test_tensor = torch.load(f'{PROCESSED_DATA_DIR}/X_test_tensor.pt')
        y_test_tensor = torch.load(f'{PROCESSED_DATA_DIR}/y_test_tensor.pt')
    except FileNotFoundError:
        print("Error: Tensor files not found. Please run the data_preprocessing.py script first.")
        return

    input_features = X_train_tensor.shape[1]
    print(f"Data loaded. Number of input features: {input_features}")

    # --- 2. Instantiate and Train the Model ---
    model = PricePredictor(num_input_features=input_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    print("Training complete.")

    # --- 3. Evaluate the Model ---
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    y_test_np = y_test_tensor.numpy().flatten()
    y_pred_np = y_pred_tensor.numpy().flatten()

    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    r2 = r2_score(y_test_np, y_pred_np)

    print("\n--- Model Evaluation ---")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")

    # --- 4. Save the Trained Model ---
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTrained model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run_training()
    print("\n--- MODEL TRAINING SCRIPT FINISHED ---")