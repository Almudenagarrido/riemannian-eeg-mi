"""Final SPDNet experiment: 9 subjects x 5 seeds, train on day 1, test on day 2.

For each subject and seed:
    1. choose_epochs: decide how long to train, with day 1 only
    2. final_test:    train on all of day 1, evaluate once on day 2

    uv run python scripts/run_spdnet.py                  # SPDNet (Riemannian optimisation)
    uv run python scripts/run_spdnet.py --mode euclid    # ablation (Euclidean optimisation of W)
    uv run python scripts/run_spdnet.py --subjects 1 --seeds 1 --epochs 5   # quick test

Use the values chosen by tune_spdnet.py (for each mode) for --sizes, --eps and --lr-stiefel.
Output:
    results/spdnet_<mode>.csv             one row per subject and seed
    results/spdnet_<mode>_config.json     the exact configuration of the run and the git commit
    results/curves/spdnet_<mode>_curves.csv and results/figures/spdnet_<mode>_curves.png
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from riemannian_eeg.training import Config, choose_epochs, final_test, load_subject

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHOD = {"riemann": "SPDNet", "euclid": "SPDNet (Euclidean opt.)"}
CODE = ["src/riemannian_eeg/spdnet.py", "src/riemannian_eeg/training.py", "scripts/run_spdnet.py"]


def code_version():
    """Git commit of the code used, and whether those files had uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD") or None,
                "uncommitted_changes": bool(git("status", "--porcelain", "--", *CODE))}
    except OSError:  # git not installed
        return {"git_commit": None, "uncommitted_changes": None}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["riemann", "euclid"], default="riemann")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 8])
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--lr-stiefel", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    return p.parse_args()


def plot_curves(curves, path):
    """Train and validation loss per epoch (mean over seeds), one panel per subject."""
    subjects = sorted(curves["subject"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, squeeze=False)
    for ax, subject in zip(axes.flat, subjects):
        mean = curves[curves["subject"] == subject].groupby("epoch").mean()
        ax.plot(mean.index, mean["train_loss"], label="train")
        ax.plot(mean.index, mean["val_loss"], label="validation")
        ax.set_title(f"Subject {subject}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    for ax in axes.flat[len(subjects):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config(mode=args.mode, sizes=tuple(args.sizes), eps=args.eps,
                 lr_stiefel=args.lr_stiefel, epochs=args.epochs)
    method = METHOD[args.mode]
    version = code_version()
    print(f"{method}: {cfg}")
    print(f"Code: commit {version['git_commit']}")
    if version["uncommitted_changes"]:
        print("Warning: the network or training code has uncommitted changes, so these results "
              "cannot be traced back to a commit. Commit the code before the final run.")

    rows, curves = [], []
    for subject in args.subjects:
        C_train, y_train, C_test, y_test = load_subject(subject)
        for seed in range(args.seeds):
            best_epoch, history = choose_epochs(C_train, y_train, cfg, seed)
            result = final_test(C_train, y_train, C_test, y_test, cfg, seed, best_epoch)

            print(f"S{subject} seed {seed}: accuracy {result['accuracy']:.3f}, "
                  f"best epoch {best_epoch}, ReEig activity {result['reeig_activity']}, "
                  f"|W^T W - I| {result['orthogonality_error']:.1e}")
            rows.append({
                "method": method, "subject": subject, "seed": seed,
                "accuracy": result["accuracy"], "kappa": result["kappa"],
                "best_epoch": best_epoch,
                "reeig_activity_layer1": result["reeig_activity"][0],
                "reeig_activity_layer2": result["reeig_activity"][-1],
                "orthogonality_error": result["orthogonality_error"],
            })
            history["subject"], history["seed"] = subject, seed
            curves.append(history)

    for folder in ["", "curves", "figures"]:
        (RESULTS / folder).mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    curves = pd.concat(curves)
    results.to_csv(RESULTS / f"spdnet_{args.mode}.csv", index=False)
    curves.to_csv(RESULTS / "curves" / f"spdnet_{args.mode}_curves.csv", index=False)
    plot_curves(curves, RESULTS / "figures" / f"spdnet_{args.mode}_curves.png")
    config = {**asdict(cfg), "subjects": args.subjects, "seeds": args.seeds, **version}
    (RESULTS / f"spdnet_{args.mode}_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"\nMean accuracy: {100 * results['accuracy'].mean():.1f} %")
    late = (results["best_epoch"] > 0.9 * args.epochs).mean()
    if late > 0.1:
        print(f"Warning: in {100 * late:.0f} % of the runs the best epoch is in the last 10 % "
              f"of the {args.epochs} epochs. The validation loss may still be decreasing: "
              f"consider a larger --epochs (the same for both modes).")
    print(f"Saved results/spdnet_{args.mode}.csv, its configuration and the curves")


if __name__ == "__main__":
    main()
