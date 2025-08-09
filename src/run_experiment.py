import argparse
from itertools import product

import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import model
import training
from constants import (
    AUX_KEY,
    CKPT_PATH,
    DATA_DIR,
    FIGURES_DIR,
    IDX_REGULAR,
    NUM_DIGITS,
    RESULTS_PATH,
    SEED_KEY,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # TF32 instead of FP32

BATCH_SIZE = 1024  # adjust based on GPU RAM
LR = 1e-3
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
VERBOSE = False


def run_experiment(
    *,
    num_aux: int,
    seed: int,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
) -> tuple[float, ...]:
    """Run the MNIST subliminal learning experiment with specified auxiliary digits and seed."""
    torch.manual_seed(seed)
    train_loader, test_loader, random_loader = loaders
    idx_aux = list(range(NUM_DIGITS, NUM_DIGITS + num_aux))

    reference = model.ClassifierWithAux(num_aux).to(DEVICE)
    teacher = reference.copy().to(DEVICE)
    student_train = reference.copy().to(DEVICE)
    student_random = reference.copy().to(DEVICE)

    # 1. Train the teacher model (on regular digits)
    training.train(
        model=teacher,
        n_epoch=EPOCHS_TEACHER,
        lr=LR,
        train_loader=train_loader,
        test_loader=test_loader,
        verbose=VERBOSE,
    )

    # 2. Distill the teacher model into the student model (on auxiliary digits)
    training.distill(
        student=student_train,
        teacher=teacher,
        dataloader=train_loader,
        n_epoch=EPOCHS_DISTILL,
        lr=LR,
        idx=idx_aux,
        verbose=VERBOSE,
    )
    training.distill(
        student=student_random,
        teacher=teacher,
        dataloader=random_loader,
        n_epoch=EPOCHS_DISTILL,
        lr=LR,
        idx=idx_aux,
        verbose=VERBOSE,
    )

    # 3. Evaluate the models' accuracy for regular digits on the unseen test set
    acc_reference = training.evaluate_accuracy(model=reference, test_loader=test_loader)
    acc_teacher = training.evaluate_accuracy(model=teacher, test_loader=test_loader)
    acc_student_train = training.evaluate_accuracy(
        model=student_train, test_loader=test_loader
    )
    acc_student_random = training.evaluate_accuracy(
        model=student_random, test_loader=test_loader
    )

    # 4. Estimate entropy of teacher logits (regular/auxiliary) for train/random datasets
    ent_train_aux = training.estimate_entropy(
        model=teacher, dataloader=train_loader, idx=idx_aux
    )
    ent_random_aux = training.estimate_entropy(
        model=teacher, dataloader=random_loader, idx=idx_aux
    )
    ent_train_reg = training.estimate_entropy(
        model=teacher, dataloader=train_loader, idx=IDX_REGULAR
    )
    ent_random_reg = training.estimate_entropy(
        model=teacher, dataloader=random_loader, idx=IDX_REGULAR
    )

    return (
        acc_reference,
        acc_teacher,
        acc_student_train,
        acc_student_random,
        ent_train_aux,
        ent_random_aux,
        ent_train_reg,
        ent_random_reg,
    )


def main() -> None:
    """Main function to run the MNIST subliminal learning experiment."""
    args = parse_arguments()
    print(
        "Run configured with:\n\t"
        + "\n\t".join(f"{k}={v}" for k, v in vars(args).items())
    )
    seed_list = list(range(args.num_seed))
    num_aux_list = (
        torch.logspace(
            torch.log10(torch.tensor(args.min_aux)),
            torch.log10(torch.tensor(args.max_aux)),
            args.num_aux,
        )
        .long()
        .tolist()
    )

    make_directories()
    df_ckpt = get_checkpoint()
    loaders = get_dataloaders()

    for num_aux, seed in product(num_aux_list, seed_list):
        run_and_add_experiment(num_aux, seed, df_ckpt, loaders)

    df_ckpt.to_csv(RESULTS_PATH, float_format="%.4f")
    print(f"All experiments completed. Results saved to {RESULTS_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MNIST subliminal learning experiment."
    )
    parser.add_argument(
        "--num-seed", type=int, default=10, help="Number of random seeds to use"
    )
    parser.add_argument(
        "--min-aux", type=int, default=1, help="Minimum number of auxiliary digits"
    )
    parser.add_argument(
        "--max-aux", type=int, default=10000, help="Maximum number of auxiliary digits"
    )
    parser.add_argument(
        "--num-aux", type=int, default=100, help="Number of auxiliary digits to test."
    )
    return parser.parse_args()


def make_directories() -> None:
    """Create necessary directories if they do not exist."""
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def get_checkpoint() -> pd.DataFrame:
    """Load the checkpoint DataFrame, or create a new one if it doesn't exist."""
    if CKPT_PATH.exists():
        print(f"Loading checkpoint from {CKPT_PATH}")
        df = pd.read_csv(CKPT_PATH, index_col=[AUX_KEY, SEED_KEY])
    else:
        print("No checkpoint found. Creating empty DataFrame.")
        columns = [
            "acc_reference",
            "acc_teacher",
            "acc_student_train",
            "acc_student_random",
            "ent_train_aux",
            "ent_random_aux",
            "ent_train_reg",
            "ent_random_reg",
        ]
        index = [AUX_KEY, SEED_KEY]
        df = pd.DataFrame(columns=index + columns).set_index(index)
    return df


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the DataLoaders for training, testing, and random datasets.

    Note that the underlying datasets are entirely loaded in the GPU to minimize I/O overhead."""
    datasets = data.load_datasets(
        DEVICE
    )  # Load datasets directly to the GPU to minimize I/O overhead
    return tuple(
        DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # 0 since we are using GPU and want to avoid CPU-GPU transfer overhead
        )
        for dataset in datasets
    )  # type: ignore


def run_and_add_experiment(
    num_aux: int,
    seed: int,
    df_ckpt: pd.DataFrame,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
) -> None:
    """Run the experiment and add results to the checkpoint DataFrame."""
    if not df_ckpt.empty and (num_aux, seed) in df_ckpt.index:
        print(f"Already completed experiment with {num_aux=}, {seed=}. Skipping...")
        return
    print(f"Running experiment with {num_aux=}, {seed=}")
    values = run_experiment(num_aux=num_aux, seed=seed, loaders=loaders)

    df_ckpt.loc[(num_aux, seed), :] = values  # type: ignore
    df_ckpt.to_csv(CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()
