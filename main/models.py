import torch
from torch import nn
import confusion as con
from torch import optim
import time
import sklearn.utils as sku
import numpy as np

"""
This script contains several classes containing a neural network structure. These classes can be given as an input to the "GlitchModel" class in "modelframe.py".

Each class represents a different neural network architecture for glitch classification:
- GlitchClassifier: A basic classifier with two hidden layers
- GlitchClassifier_1: A deeper network with dropout
- GlitchClassifier_2: A wider network with increasing then decreasing layer sizes
- GlitchClassifier_3: A very deep network with multiple 256-neuron layers
- GlitchClassifier_4: A network with three 350-neuron hidden layers
- GlitchClassifier_MultiClass_Optimized: An optimized network with decreasing layer sizes
- GlitchClassifierDynamic: A network with varying layer sizes

These models can be used for binary or multi-class glitch classification tasks.
"""


class GlitchClassifier(nn.Module):
    def __init__(self, input_dim : int=6, hidden_dim :int=32, output_dim:int=2):
        super(GlitchClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    
    
class GlitchClassifier_1(nn.Module):
    def __init__(self, input_dim : int=6, output_dim:int=2):
        super(GlitchClassifier_1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.Dropout(0.5),  
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, output_dim),
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    


class GlitchClassifier_2(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 2):
        super(GlitchClassifier_2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            # nn.Softmax(dim=1)  # Uncomment if using for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    
class GlitchClassifier_3(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 2):
        super(GlitchClassifier_3, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            # nn.Softmax(dim=1)  # Uncomment if using for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    
class GlitchClassifier_4(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 2):
        super(GlitchClassifier_4, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=350),  # First layer
            nn.ReLU(),                                           # Activation
            nn.Linear(in_features=350, out_features=350),        # Second layer
            nn.ReLU(),                                           # Activation
            nn.Linear(in_features=350, out_features=350),        # Third layer
            nn.ReLU(),                                           # Activation
            nn.Linear(in_features=350, out_features=output_dim), # Output layer
            nn.Softmax(dim=1)    
            # nn.Softmax(dim=1)  # Uncomment if using for multi-class classification
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        return self.network(x)


class GlitchClassifier_MultiClass_Optimized(nn.Module):
    def __init__(self, input_dim : int=6, hidden_dim :int=32, output_dim:int=2):
        super(GlitchClassifier_MultiClass_Optimized, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), output_dim),
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
    
    

class GlitchClassifierDynamic(nn.Module):
    def __init__(self, input_dim : int=6, hidden_dim :int=32, output_dim:int=2):
        super(GlitchClassifierDynamic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.Softmax(dim=1)
            # nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        return self.network(x)
