import torch
from torch import nn
import confusion as con
from torch import optim
import time
import sklearn.utils as sku
import numpy as np

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
