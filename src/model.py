import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierWithAux(nn.Module):
    """A simple MLP classifier for MNIST, but with auxiliary outputs."""

    def __init__(self, num_aux: int) -> None:
        super().__init__()
        self.num_aux = num_aux
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10 + self.num_aux)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)  # flatten images
        x = self.fc2(self.activation(self.fc1(x)))
        return x

    def copy(self) -> "ClassifierWithAux":
        """Create a deep copy of the model with the same parameters."""
        new_model = self.__class__(self.num_aux)
        new_model.load_state_dict(self.state_dict())
        return new_model
