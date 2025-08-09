import torch
from constants import DATA_DIR
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def load_datasets(
    device: torch.device,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load three `TensorDataset` for MNIST (train, test, and random images with
    dummy labels) on the specified device.
    """
    # Load & process MNIST
    train_dataset_pil = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_dataset_pil = datasets.MNIST(DATA_DIR, train=False, download=True)
    train_images = train_dataset_pil.data.float().unsqueeze(1) / 255.0
    test_images = test_dataset_pil.data.float().unsqueeze(1) / 255.0
    normalize = transforms.Normalize((0.5,), (0.5,))
    train_images = normalize(train_images).to(device=device)
    test_images = normalize(test_images).to(device=device)
    train_labels = train_dataset_pil.targets.to(device=device)
    test_labels = test_dataset_pil.targets.to(device=device)

    # Create random dataset
    random_images = torch.rand_like(train_images, device=device) * 2 - 1
    dummy_labels = torch.zeros(len(random_images), device=device)

    # Create the final TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    random_dataset = TensorDataset(random_images, dummy_labels)

    return train_dataset, test_dataset, random_dataset
