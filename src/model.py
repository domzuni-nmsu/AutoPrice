# src/model.py (Corrected Version)
import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    # The parameter name is now corrected to 'num_input_features'
    def __init__(self, num_input_features, hidden_size1=128, hidden_size2=64, dropout_rate=0.2):
        super(PricePredictor, self).__init__()
        # It is also corrected here where it's used
        self.layer1 = nn.Linear(num_input_features, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x