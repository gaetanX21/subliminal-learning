# Subliminal Learning on MNIST

This repository contains code to run subliminal learning experiments on the MNIST dataset. It includes functionalities for loading data, defining models, training them, and visualizing results. The project is structured to facilitate easy experimentation with different configurations and parameters.

## Project Structure

- `src/` — Source code for data handling, model definition, training, plotting, and experiment management.
  - `constants.py` — Project-wide constants and configuration.
  - `data.py` — Functions for loading and preprocessing MNIST data.
  - `model.py` — Model architecture.
  - `training.py` — Training and evaluation routines.
  - `plot.py` — Script to generate figures.
  - `run_experiment.py` — Main script to run experiments.
- `data/` — MNIST dataset (images and labels).
- `results/` — Experiment results and checkpoint.
- `figures/` — Figures generated from experiments with `src/plot.py`.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gaetanX21/subliminal-learning.git
   cd sl
   ```
2. **Install dependencies:**
   This project uses Python. Install required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run experiments:**
    The main script to run experiments is `src/run_experiment.py`.
    It sweeps over a `num-aux`-long logarithmic range of auxiliary digits,
    starting at `min-aux` and going up to `max-aux`.
    For each value of `num_aux`, `num_seed` experiments with different random seed are run.

    We thus have the following command-line arguments to customize the experiment:
    - `--num-aux` to set the number of auxiliary digits (default 100).
    - `--min-aux` to set the minimum number of auxiliary digits (default 1).
    - `--max-aux` to set the maximum number of auxiliary digits (default 10000).
    - `--num-seed` to set the number of differently-seeded experiments to run per `num_aux` value (default 10).

   This will train the model and save results in `results/` and figures in `figures/`.
   Training can be long, so checkpoints are saved periodically in `results/checkpoint.csv`.

## Results
- Training checkpoints are saved in `results/checkpoint.csv`.
- Plots and visualizations are saved in `figures/`.