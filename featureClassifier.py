import torch.nn as nn

class FeatureClassifier(nn.Module):
    """MLP for fixed-size transformer features."""
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.3):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers = []
        dims = [input_dim] + list(hidden_dims)
        for idx in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)