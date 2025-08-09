import matplotlib.pyplot as plt
import pandas as pd

from constants import FIGURES_DIR, RESULTS_DIR

plt.xkcd()


def main() -> None:
    """Main function to plot results from the experiment."""
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    stats = df.groupby(["num_auxiliary"]).agg(["mean", "std"])
    # split stats into separate DataFrames for easier plotting
    mean, std = stats.xs("mean", level=1, axis=1), stats.xs("std", level=1, axis=1)

    acc_cols = [col for col in mean.columns if col.startswith("acc")]
    ent_cols = [col for col in mean.columns if col.startswith("ent")]

    # 1. Plot accuracy of models
    labels = [
        "reference",
        "teacher",
        "student MNIST",
        "student random",
    ]
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(acc_cols):
        plt.fill_between(
            mean.index, mean[col] - std[col], mean[col] + std[col], alpha=0.2
        )
        plt.plot(mean.index, mean[col], label=labels[i])
    plt.xlabel("Number of Auxiliary Logits")
    plt.xscale("log")
    plt.ylabel("Validation Accuracy on regular MNIST Classification")
    plt.legend()
    plt.savefig(FIGURES_DIR / "accuracy.png")

    # 2. Plot entropy of teacher logits
    labels = [
        "auxiliary logits, MNIST train",
        "auxiliary logits, random images",
        "regular logits, MNIST train",
        "regular logits, random images",
    ]
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(ent_cols):
        plt.fill_between(
            mean.index, mean[col] - std[col], mean[col] + std[col], alpha=0.2
        )
        plt.plot(mean.index, mean[col], label=labels[i])
    plt.xlabel("Number of Auxiliary Logits")
    plt.ylabel("Entropy of Teacher Logits")
    plt.xscale("log")
    plt.legend()
    plt.show()
    plt.savefig(FIGURES_DIR / "entropy.png")
    print(f"Plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
