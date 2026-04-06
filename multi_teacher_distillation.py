import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


class MultiTeacherDistillationLoss(nn.Module):
    """Multi-teacher distillation loss using feature and optional logits matching."""

    def __init__(
        self,
        student_dims: List[int],
        teacher_dims_map: Dict[str, List[int]],
        alpha: float = 0.5,
        temperature: float = 1.0,
        teacher_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if not teacher_dims_map:
            raise ValueError("teacher_dims_map must contain at least one teacher")

        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        if teacher_weights is None:
            teacher_weights = {name: 1.0 for name in teacher_dims_map.keys()}

        total_weight = float(sum(teacher_weights.values()))
        if total_weight <= 0:
            raise ValueError("Sum of teacher_weights must be positive")

        self.teacher_names = list(teacher_dims_map.keys())
        self.teacher_weights = {
            name: float(teacher_weights.get(name, 0.0)) / total_weight for name in self.teacher_names
        }

        self.n_pairs = {}
        self.projections = nn.ModuleDict()
        for teacher_name, teacher_dims in teacher_dims_map.items():
            n_pairs = min(len(student_dims), len(teacher_dims))
            self.n_pairs[teacher_name] = n_pairs

            proj_layers = nn.ModuleList()
            for i in range(n_pairs):
                s_dim = student_dims[-(n_pairs - i)]
                t_dim = teacher_dims[-(n_pairs - i)]
                if s_dim != t_dim:
                    proj_layers.append(nn.Linear(s_dim, t_dim, bias=False))
                else:
                    proj_layers.append(nn.Identity())
            self.projections[teacher_name] = proj_layers

    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: List[torch.Tensor],
        teacher_features_map: Dict[str, List[torch.Tensor]],
        targets: torch.Tensor,
        teacher_logits_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        task_loss = self.ce_loss(student_logits, targets)

        feature_kd_loss = torch.tensor(0.0, device=student_logits.device)
        logits_kd_loss = torch.tensor(0.0, device=student_logits.device)

        for teacher_name in self.teacher_names:
            weight = self.teacher_weights[teacher_name]
            if weight <= 0:
                continue

            teacher_features = teacher_features_map[teacher_name]
            n_pairs = self.n_pairs[teacher_name]

            teacher_feature_loss = torch.tensor(0.0, device=student_logits.device)
            if n_pairs > 0:
                s_feats = student_features[-n_pairs:]
                t_feats = teacher_features[-n_pairs:]
                for proj, s_feat, t_feat in zip(self.projections[teacher_name], s_feats, t_feats):
                    teacher_feature_loss = teacher_feature_loss + self.mse_loss(proj(s_feat), t_feat.detach())
                teacher_feature_loss = teacher_feature_loss / n_pairs

            teacher_logits_loss = torch.tensor(0.0, device=student_logits.device)
            if (
                teacher_logits_map is not None
                and teacher_name in teacher_logits_map
                and teacher_logits_map[teacher_name] is not None
                and self.temperature > 0
            ):
                t_logits = teacher_logits_map[teacher_name]
                student_log_probs = torch.nn.functional.log_softmax(
                    student_logits / self.temperature, dim=1
                )
                teacher_probs = torch.nn.functional.softmax(t_logits / self.temperature, dim=1)
                teacher_logits_loss = self.kl_loss(student_log_probs, teacher_probs.detach())
                teacher_logits_loss = teacher_logits_loss * (self.temperature ** 2)

            feature_kd_loss = feature_kd_loss + (weight * teacher_feature_loss)
            logits_kd_loss = logits_kd_loss + (weight * teacher_logits_loss)

        distill_loss = feature_kd_loss + logits_kd_loss
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss

        components = {
            "feature_loss": feature_kd_loss,
            "logits_loss": logits_kd_loss,
        }
        return total_loss, task_loss, distill_loss, components


class MultiTeacherDistillationDataset(Dataset):
    """Dataset with student input plus per-teacher intermediate features and logits."""

    def __init__(
        self,
        features: torch.Tensor,
        labels,
        teacher_intermediates_map: Dict[str, List[torch.Tensor]],
        teacher_logits_map: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.features = features
        self.labels = labels
        self.teacher_intermediates_map = teacher_intermediates_map
        self.teacher_logits_map = teacher_logits_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        teacher_features_item = {
            teacher_name: [layer_feat[idx] for layer_feat in layer_feats]
            for teacher_name, layer_feats in self.teacher_intermediates_map.items()
        }

        teacher_logits_item = None
        if self.teacher_logits_map is not None:
            teacher_logits_item = {
                teacher_name: logits[idx] for teacher_name, logits in self.teacher_logits_map.items()
            }

        return self.features[idx], int(self.labels[idx]), teacher_features_item, teacher_logits_item


def make_multi_teacher_loader(
    features: torch.Tensor,
    labels,
    teacher_intermediates_map: Dict[str, List[torch.Tensor]],
    teacher_logits_map: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MultiTeacherDistillationDataset(
        features,
        labels,
        teacher_intermediates_map,
        teacher_logits_map=teacher_logits_map,
    )

    def collate_fn(batch):
        batch_features = torch.stack([item[0] for item in batch])
        batch_labels = torch.tensor([item[1] for item in batch])

        teacher_names = list(batch[0][2].keys())
        batch_teacher_feats = {}
        for teacher_name in teacher_names:
            num_layers = len(batch[0][2][teacher_name])
            batch_teacher_feats[teacher_name] = []
            for layer_idx in range(num_layers):
                layer_batch = torch.stack([item[2][teacher_name][layer_idx] for item in batch])
                batch_teacher_feats[teacher_name].append(layer_batch)

        batch_teacher_logits = None
        if batch[0][3] is not None:
            batch_teacher_logits = {
                teacher_name: torch.stack([item[3][teacher_name] for item in batch])
                for teacher_name in teacher_names
            }

        return batch_features, batch_labels, batch_teacher_feats, batch_teacher_logits

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _compute_detailed_metrics(all_preds: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _train_one_epoch(
    student_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    distill_criterion: MultiTeacherDistillationLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    student_model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_distill_loss = 0.0
    total_feature_loss = 0.0
    total_logits_loss = 0.0
    all_preds = []
    all_labels = []

    for x_batch, y_batch, teacher_feats_map, teacher_logits_map in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        teacher_feats_map = {
            t_name: [feat.to(device) for feat in feats]
            for t_name, feats in teacher_feats_map.items()
        }
        if teacher_logits_map is not None:
            teacher_logits_map = {
                t_name: logits.to(device) for t_name, logits in teacher_logits_map.items()
            }

        optimizer.zero_grad()
        student_logits, student_feats = student_model(x_batch, return_intermediate=True)
        loss, task_loss, distill_loss, components = distill_criterion(
            student_logits,
            student_feats,
            teacher_feats_map,
            y_batch,
            teacher_logits_map=teacher_logits_map,
        )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_task_loss += task_loss.item() * batch_size
        total_distill_loss += distill_loss.item() * batch_size
        total_feature_loss += components["feature_loss"].item() * batch_size
        total_logits_loss += components["logits_loss"].item() * batch_size

        all_preds.append(torch.argmax(student_logits, dim=1).detach().cpu())
        all_labels.append(y_batch.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = _compute_detailed_metrics(all_preds, all_labels)
    denom = len(loader.dataset)
    metrics["loss"] = total_loss / denom
    metrics["task_loss"] = total_task_loss / denom
    metrics["distill_loss"] = total_distill_loss / denom
    metrics["feature_loss"] = total_feature_loss / denom
    metrics["logits_loss"] = total_logits_loss / denom
    return metrics


def _eval_one_epoch(
    student_model: nn.Module,
    loader: DataLoader,
    distill_criterion: MultiTeacherDistillationLoss,
    device: torch.device,
) -> Dict[str, float]:
    student_model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_distill_loss = 0.0
    total_feature_loss = 0.0
    total_logits_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, teacher_feats_map, teacher_logits_map in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            teacher_feats_map = {
                t_name: [feat.to(device) for feat in feats]
                for t_name, feats in teacher_feats_map.items()
            }
            if teacher_logits_map is not None:
                teacher_logits_map = {
                    t_name: logits.to(device) for t_name, logits in teacher_logits_map.items()
                }

            student_logits, student_feats = student_model(x_batch, return_intermediate=True)
            loss, task_loss, distill_loss, components = distill_criterion(
                student_logits,
                student_feats,
                teacher_feats_map,
                y_batch,
                teacher_logits_map=teacher_logits_map,
            )

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_task_loss += task_loss.item() * batch_size
            total_distill_loss += distill_loss.item() * batch_size
            total_feature_loss += components["feature_loss"].item() * batch_size
            total_logits_loss += components["logits_loss"].item() * batch_size

            all_preds.append(torch.argmax(student_logits, dim=1).detach().cpu())
            all_labels.append(y_batch.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = _compute_detailed_metrics(all_preds, all_labels)
    denom = len(loader.dataset)
    metrics["loss"] = total_loss / denom
    metrics["task_loss"] = total_task_loss / denom
    metrics["distill_loss"] = total_distill_loss / denom
    metrics["feature_loss"] = total_feature_loss / denom
    metrics["logits_loss"] = total_logits_loss / denom
    return metrics


def train_multi_teacher_student(
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    distill_criterion: MultiTeacherDistillationLoss,
    epochs: int,
    device: torch.device,
    grad_clip: float = 1.0,
    patience: int = 5,
    min_delta: float = 0.0,
    verbose: bool = True,
):
    """Train a student model with multiple teachers and early stopping."""
    best_state = copy.deepcopy(student_model.state_dict())
    best_val_acc = -1.0
    epochs_since_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = _train_one_epoch(
            student_model,
            train_loader,
            optimizer,
            distill_criterion,
            device,
            grad_clip=grad_clip,
        )
        val_metrics = _eval_one_epoch(student_model, val_loader, distill_criterion, device)

        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_task_loss": train_metrics["task_loss"],
                "train_distill_loss": train_metrics["distill_loss"],
                "train_feature_loss": train_metrics["feature_loss"],
                "train_logits_loss": train_metrics["logits_loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_task_loss": val_metrics["task_loss"],
                "val_distill_loss": val_metrics["distill_loss"],
                "val_feature_loss": val_metrics["feature_loss"],
                "val_logits_loss": val_metrics["logits_loss"],
                "val_acc": val_metrics["acc"],
                "epoch_time": epoch_time,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"(task={train_metrics['task_loss']:.4f}, "
                f"feat={train_metrics['feature_loss']:.4f}, "
                f"logits={train_metrics['logits_loss']:.4f}) "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | "
                f"time={epoch_time:.2f}s"
            )

        if val_metrics["acc"] > best_val_acc + min_delta:
            best_val_acc = val_metrics["acc"]
            best_state = copy.deepcopy(student_model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return best_state, best_val_acc, history
