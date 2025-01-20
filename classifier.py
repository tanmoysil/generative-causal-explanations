
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2 = 128, hidden_dim3 = 128, output_dim = 1):
        super(Classifier, self).__init__()

        # linear layer
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(in_features=input_dim, out_features=hidden_dim1),
            # #nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            # # Layer 2
            # nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2),
            # nn.BatchNorm1d(hidden_dim2),
            # nn.ReLU(),
            # nn.Dropout(p=0.3),
            # # Skip Connection
            # # nn.Linear(in_features=hidden_dim2, out_features=hidden_dim3),
            # # nn.ReLU(),
            # # Layer 3
            # nn.Linear(in_features=hidden_dim2, out_features=hidden_dim3),
            # nn.BatchNorm1d(hidden_dim3),
            # nn.ReLU(),
            # nn.Dropout(p=0.3),
            # Output Layer
            nn.Linear(in_features=hidden_dim1, out_features=output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
