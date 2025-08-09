from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "results.csv"
CKPT_PATH = RESULTS_DIR / "checkpoint.csv"

NUM_DIGITS = 10
IDX_REGULAR = list(range(NUM_DIGITS))
AUX_KEY = "num_auxiliary"
SEED_KEY = "seed"
