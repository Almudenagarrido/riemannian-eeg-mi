"""Comparison table and figures from all the results/*.csv files.

Each file must have the columns: method, subject, seed, accuracy, kappa.
SPDNet writes its own files (scripts/run_spdnet.py). Each of the other methods only needs to
save its results in results/ with those columns to appear in the table.

    uv run python scripts/make_table.py

Output:
    results/comparison_table.md   accuracy (%) per subject and method, mean, std, kappa, Wilcoxon
    results/figures/comparison.png
    results/figures/ablation.png  (if both SPDNet runs exist)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLUMNS = ["method", "subject", "seed", "accuracy", "kappa"]
ORDER = ["CSP+LDA (OvR)", "CSP+LDA (AJD)", "MDM", "TS+LR", "SPDNet", "SPDNet (Euclidean opt.)"]
REFERENCE = "SPDNet"


def load_results():
    """All result files together, one row per method, subject and seed."""
    tables = []
    for path in sorted(RESULTS.glob("*.csv")):
        table = pd.read_csv(path)
        if set(COLUMNS).issubset(table.columns):
            tables.append(table[COLUMNS])
    return pd.concat(tables)


def wilcoxon_p(a, b):
    """Paired Wilcoxon test over subjects: are methods a and b different?"""
    if np.allclose(a, b):
        return 1.0
    return wilcoxon(a, b).pvalue


def to_markdown(table):
    header = "| | " + " | ".join(table.columns) + " |"
    separator = "|---" * (len(table.columns) + 1) + "|"
    lines = [header, separator]
    for name, row in table.iterrows():
        cells = ["-" if pd.isna(v) else str(v) for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bar_figure(accuracy, path):
    """One group of bars per subject (plus the mean), one bar per method."""
    data = pd.concat([accuracy, accuracy.mean().to_frame("Mean").T])
    methods = accuracy.columns
    x = np.arange(len(data))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 4))
    for k, method in enumerate(methods):
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, data[method], width, label=method)
    ax.axhline(25, color="grey", linestyle="--", label="chance (25 %)")
    ax.set_xticks(x, [f"S{s}" for s in accuracy.index] + ["Mean"])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_figure(accuracy, path):
    """One point per subject: Euclidean (x) vs Riemannian (y) optimisation of the BiMap weights."""
    euclid, riemann = accuracy["SPDNet (Euclidean opt.)"], accuracy["SPDNet"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(euclid, riemann)
    for subject in accuracy.index:
        ax.annotate(f"S{subject}", (euclid[subject], riemann[subject]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.plot([0, 100], [0, 100], color="grey", linestyle="--")
    ax.set_xlim(20, 100)
    ax.set_ylim(20, 100)
    ax.set_xlabel("Euclidean optimisation of W (%)")
    ax.set_ylabel("Riemannian optimisation of W (%)")
    ax.set_title("Above the diagonal: Riemannian is better", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results = load_results()
    methods = [m for m in ORDER if m in set(results["method"])]
    methods += sorted(set(results["method"]) - set(methods))

    # 1. Mean over seeds: one value per method and subject (only SPDNet has several seeds)
    per_subject = results.groupby(["method", "subject"])[["accuracy", "kappa"]].mean().reset_index()

    # 2. Rows = subjects, columns = methods
    accuracy = 100 * per_subject.pivot(index="subject", columns="method", values="accuracy")[methods]
    kappa = per_subject.pivot(index="subject", columns="method", values="kappa")[methods]

    # 3. Table: accuracy per subject + summary rows
    table = accuracy.round(1)
    table.index = [f"S{s}" for s in accuracy.index]
    table.loc["Mean"] = accuracy.mean().round(1)
    table.loc["Std over subjects"] = accuracy.std().round(1)
    table.loc["Mean kappa"] = kappa.mean().round(3)
    if REFERENCE in methods:
        table.loc[f"Wilcoxon p vs {REFERENCE}"] = [
            np.nan if m == REFERENCE else round(wilcoxon_p(accuracy[REFERENCE], accuracy[m]), 3)
            for m in methods]

    markdown = to_markdown(table)
    (RESULTS / "comparison_table.md").write_text(
        "Test accuracy (%) on day 2 (chance = 25 %; significance threshold ~29 % with 288 trials)."
        "\n\n" + markdown + "\n")
    print(markdown)

    # 4. Variability over seeds (methods with several seeds)
    for method in methods:
        runs = results[results["method"] == method]
        if runs["seed"].nunique() > 1:
            mean_per_seed = 100 * runs.groupby("seed")["accuracy"].mean()
            print(f"\n{method}: mean accuracy per seed {mean_per_seed.round(1).tolist()}, "
                  f"std {mean_per_seed.std():.1f}")

    # 5. Figures
    (RESULTS / "figures").mkdir(exist_ok=True)
    bar_figure(accuracy, RESULTS / "figures" / "comparison.png")
    if "SPDNet" in methods and "SPDNet (Euclidean opt.)" in methods:
        ablation_figure(accuracy, RESULTS / "figures" / "ablation.png")
        wins = (accuracy["SPDNet"] > accuracy["SPDNet (Euclidean opt.)"]).sum()
        print(f"\nAblation: Riemannian better in {wins} of {len(accuracy)} subjects")
    print("\nSaved results/comparison_table.md and results/figures/")


if __name__ == "__main__":
    main()
