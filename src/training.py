import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import IDX_REGULAR
from torch.utils.data import DataLoader


def ce_regular(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for the regular logits (digits 0-9)."""
    return F.cross_entropy(logits[:, IDX_REGULAR], labels)


def train(
    *,
    model: nn.Module,
    n_epoch: int,
    lr: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    verbose: bool,
) -> None:
    """Train the model on the MNIST dataset for the regular cross-entropy loss."""
    if verbose:
        print("Training teacher model...")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epoch):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optim.zero_grad()
            logits = model(inputs)
            loss = ce_regular(logits, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        if verbose:
            test_accuracy = evaluate_accuracy(
                model=model,
                test_loader=test_loader,
            )
            print(f"{epoch=}: {train_loss=:.3f}, {test_accuracy=:.3f}")


@torch.inference_mode()
def evaluate_accuracy(
    *,
    model: nn.Module,
    test_loader: DataLoader,
) -> float:
    """Evaluate accuracy on regular MNIST classification."""
    model.eval()
    n_correct = 0
    n_total = 0
    for inputs, labels in test_loader:
        logits = model(inputs)
        preds = logits[:, IDX_REGULAR].argmax(dim=-1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.shape[0]
    return n_correct / n_total


def distill(
    *,
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    n_epoch: int,
    lr: float,
    idx: list[int],
    verbose: bool,
) -> None:
    """Distill teacher model into student model using KL divergence on auxiliary logits."""
    if verbose:
        print("Distilling teacher into student...")
    optim = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(n_epoch):
        total_loss = 0.0
        student.train()
        for inputs, _ in dataloader:
            with torch.no_grad():
                teacher_logits = teacher(inputs)[:, idx]
            student_logits = student(inputs)[:, idx]
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
        if verbose:
            print(f"{epoch=}: {total_loss=:.6f}")


@torch.inference_mode()
def estimate_entropy(
    *, model: nn.Module, dataloader: DataLoader, idx: list[int]
) -> float:
    """Estimate entropy of the selected output logits."""
    model.eval()
    entropy = 0.0
    for inputs, _ in dataloader:
        logits = model(inputs)[:, idx]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy += -(log_probs * probs).sum(dim=-1).mean().item()
    return entropy / len(dataloader)
