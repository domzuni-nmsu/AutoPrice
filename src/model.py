# This is the single, correct model definition.
# Use this exact class in both src/model.py and src/train.py

import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, num_input_features):
        super(PricePredictor, self).__init__()
        # Use underscores for layer names consistently
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