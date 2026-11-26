"""Choose the SPDNet hyperparameters using day 1 only, separately for each mode.

Every combination of the values below is tried with choose_epochs (80 % / 20 % split of day 1),
for every subject and 2 seeds. The combinations are ranked by their mean validation accuracy.
One combination is chosen for all subjects: choosing one per subject would fit the noise of
the small validation sets (58 trials).

    uv run python scripts/tune_spdnet.py                  # SPDNet
    uv run python scripts/tune_spdnet.py --mode euclid    # ablation, tuned on its own
    uv run python scripts/tune_spdnet.py --batchnorm      # SPDNet with Riemannian batch norm

Tuning both modes gives the ablation the same chance as SPDNet.
Re-centering day 2 with its own mean (--recenter session in run_spdnet.py) needs no tuning:
the validation set is part of day 1, so that option does not change anything here.
Edit the lists below to try other values.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from riemannian_eeg.training import Config, choose_epochs, load_subject

SIZES = [(16, 8), (12, 6)]      # BiMap output sizes
EPS = [1e-4, 0.25, 0.5]         # ReEig threshold
LR_STIEFEL = [0.1, 0.3, 0.5]    # learning rate of the BiMap weights (SGD, no momentum)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["riemann", "euclid"], default="riemann")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batchnorm", action="store_true")
    args = p.parse_args()
    name = f"tune_{args.mode}" + ("_bn" if args.batchnorm else "")

    # day 1 of every subject (day 2 is loaded but never used here)
    day1 = {}
    for subject in args.subjects:
        C_train, y_train, _, _ = load_subject(subject)
        day1[subject] = (C_train, y_train)

    rows = []
    for sizes in SIZES:
        for eps in EPS:
            for lr_stiefel in LR_STIEFEL:
                cfg = Config(mode=args.mode, sizes=sizes, eps=eps, lr_stiefel=lr_stiefel,
                             epochs=args.epochs, batchnorm=args.batchnorm)
                val_accuracies = []
                for subject, (C, y) in day1.items():
                    for seed in range(args.seeds):
                        best_epoch, history = choose_epochs(C, y, cfg, seed)
                        best_row = history[history["epoch"] == best_epoch]
                        val_accuracies.append(best_row["val_acc"].item())

                rows.append({"sizes": sizes, "eps": eps, "lr_stiefel": lr_stiefel,
                             "val_acc": round(100 * np.mean(val_accuracies), 1),
                             "val_acc_std": round(100 * np.std(val_accuracies), 1)})
                print(f"sizes={sizes} eps={eps:g} lr_stiefel={lr_stiefel:g}: "
                      f"validation accuracy {rows[-1]['val_acc']:.1f} %")

    table = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "tuning" / f"{name}.csv", index=False)

    print("\nBest first:")
    print(table.to_string(index=False))
    best = table.iloc[0]
    options = ("" if args.mode == "riemann" else " --mode euclid") + \
              (" --batchnorm" if args.batchnorm else "")
    print(f"\nSaved results/tuning/{name}.csv")
    print("Final run with the best combination:")
    print(f"  uv run python scripts/run_spdnet.py{options} "
          f"--sizes {' '.join(map(str, best['sizes']))} "
          f"--eps {best['eps']:g} --lr-stiefel {best['lr_stiefel']:g}")
    print("If several combinations differ by less than their std, prefer the smallest network.")


if __name__ == "__main__":
    main()
