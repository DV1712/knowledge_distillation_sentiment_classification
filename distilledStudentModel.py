import torch.nn as nn

class DistilledStudentModel(nn.Module):
    """
    Student model for feature-based distillation.
    Learns to mimic teacher's intermediate representations while predicting labels.
    """
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.3):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        
        self.feature_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Build feature extraction layers
        dims = [input_dim] + list(hidden_dims)
        for idx in range(len(hidden_dims)):
            self.feature_layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x, return_intermediate=False):
        """
        Forward pass with optional intermediate representations.
        Args:
            x: Input features
            return_intermediate: If True, return intermediate layer outputs
        Returns:
            logits or (logits, intermediate_features)
        """
        intermediates = []
        h = x
        
        for layer, activation, dropout in zip(self.feature_layers, self.activations, self.dropouts):
            h = layer(h)
            h = activation(h)
            intermediates.append(h)  # Store before dropout
            h = dropout(h)
        
        logits = self.classifier(h)
        
        if return_intermediate:
            return logits, intermediates
        return logits