import torch
from torch import nn
from torch.utils.data import Dataset

class DistillationLoss(nn.Module):
    """
    Combined loss for feature-based distillation.

    Projection layers map each student intermediate layer to the corresponding
    teacher layer dimension before computing MSE, so distillation works even
    when student and teacher architectures differ in width (compression case).

    Layer pairing strategy: align from the deepest layer upward for
    min(n_student_layers, n_teacher_layers) pairs.
    """
    def __init__(self, student_dims, teacher_dims, alpha=0.5, temperature=1.0):
        """
        Args:
            student_dims: List of student hidden layer output dims e.g. [256, 128]
            teacher_dims: List of teacher hidden layer output dims e.g. [512, 256]
            alpha: Weight for distillation loss (1-alpha for task loss)
            temperature: Temperature for softening (reserved; not used in feature distillation)
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Pair layers from the deepest end up to the number of available pairs
        n_pairs = min(len(student_dims), len(teacher_dims))
        self.n_pairs = n_pairs

        # One projection per pair: Linear(s_dim -> t_dim), or Identity if dims already match
        self.projections = nn.ModuleList()
        for i in range(n_pairs):
            s_dim = student_dims[-(n_pairs - i)]
            t_dim = teacher_dims[-(n_pairs - i)]
            if s_dim != t_dim:
                self.projections.append(nn.Linear(s_dim, t_dim, bias=False))
            else:
                self.projections.append(nn.Identity())

    def forward(self, student_logits, student_features, teacher_features, targets, teacher_logits=None):
        """
        Compute combined distillation loss.
        Args:
            student_logits: Student model predictions
            student_features: List of student intermediate features
            teacher_features: List of teacher intermediate features (detached inside)
            targets: Ground truth labels
            teacher_logits: Teacher model predictions (for logits KD; optional)
        Returns:
            total_loss, task_loss, distill_loss
        """
        # Task loss (cross-entropy)
        task_loss = self.ce_loss(student_logits, targets)

        # Feature distillation loss — align from deepest layer upward
        s_feats = student_features[-self.n_pairs:] if self.n_pairs > 0 else []
        t_feats = teacher_features[-self.n_pairs:] if self.n_pairs > 0 else []

        feature_kd_loss = torch.tensor(0.0, device=student_logits.device)
        for proj, s_feat, t_feat in zip(self.projections, s_feats, t_feats):
            s_projected = proj(s_feat)
            feature_kd_loss = feature_kd_loss + self.mse_loss(s_projected, t_feat.detach())

        # Normalize by the number of matched pairs (not total student layers)
        if self.n_pairs > 0:
            feature_kd_loss = feature_kd_loss / self.n_pairs

        # Logits-based KD loss (if teacher logits provided)
        logits_kd_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None and self.temperature > 0:
            # Soften logits with temperature
            student_log_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
            teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
            logits_kd_loss = self.kl_loss(student_log_probs, teacher_probs.detach())
            # Scale by temperature squared to match loss magnitude after softening
            logits_kd_loss = logits_kd_loss * (self.temperature ** 2)

        # Combine distillation losses (feature MSE + logits KD)
        distill_loss = feature_kd_loss + logits_kd_loss

        # Combine all losses: task loss + weighted distillation loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss

        return total_loss, task_loss, distill_loss
    


class DistillationDataset(Dataset):
    """Dataset that provides input features, labels, and teacher intermediate features."""
    def __init__(self, features, labels, teacher_intermediates):
        self.features = features
        self.labels = labels
        self.teacher_intermediates = teacher_intermediates
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        teacher_feats = [layer_feat[idx] for layer_feat in self.teacher_intermediates]
        return self.features[idx], int(self.labels[idx]), teacher_feats