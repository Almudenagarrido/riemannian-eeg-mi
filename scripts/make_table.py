"""Final comparison table, paired tests and figures from all the results/*.csv files.

Each method saves its results in results/ as a CSV with one row per subject (and seed).
The script accepts the files as each person saved them:

- columns ``subject, accuracy, kappa`` are required;
- ``method`` is optional: if missing, the name is taken from FILE_METHODS (e.g. csp_lda.csv);
- ``seed`` is optional: if missing, the method is deterministic and seed = 0;
- method names are mapped to a common naming with ALIASES (e.g. "RC+TS+LR").

    uv run python scripts/make_table.py

Output:
    results/comparison_table.md    accuracy (%) per subject and method + paired tests
    results/comparison_table.tex   the same two tables in LaTeX (booktabs), ready for the report
    results/figures/comparison.png per-subject accuracy, without and with session recentering
    results/figures/ablation.png   Riemannian vs Euclidean optimisation of W, per subject
    results/figures/recentering.png mean accuracy of each method without / with recentering
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
REQUIRED = ["subject", "accuracy", "kappa"]
N_SUBJECTS = 9
RC = " + session recentering"

# Files saved without a "method" column
FILE_METHODS = {"csp_lda.csv": "CSP+LDA"}

# Names used by the other scripts -> names used in the table
ALIASES = {
    "RC+CSP+LDA": "CSP+LDA" + RC,
    "RC+MDM": "MDM" + RC,
    "RC+TS+LR": "TS+LR" + RC,
    "RC+TS+LRCV": "TS+LR (tuned C)" + RC,
    "FB+RC+TS+LRCV": "Filter bank TS+LR (tuned C)" + RC,
}

# Row order of the table, in blocks. Methods not listed go to the last block.
BLOCKS = {
    "Without session recentering": [
        "CSP+LDA", "MDM", "TS+LR", "SPDNet (Euclidean opt.)", "SPDNet", "SPDNet + BN",
    ],
    "With session recentering": [
        "CSP+LDA" + RC, "MDM" + RC, "TS+LR" + RC, "TS+LR (tuned C)" + RC,
        "SPDNet (Euclidean opt.)" + RC, "SPDNet" + RC, "SPDNet + BN" + RC,
        "SPDNet" + RC + " (ensemble of 5 seeds)",
    ],
    "Other variants (not directly comparable: several frequency bands)": [
        "Filter bank TS+LR (tuned C)" + RC,
    ],
}

# Methods that must be there for the final table (a warning is printed if one is missing)
EXPECTED = ["CSP+LDA", "MDM", "TS+LR", "SPDNet", "SPDNet (Euclidean opt.)",
            "MDM" + RC, "TS+LR" + RC, "SPDNet" + RC, "SPDNet (Euclidean opt.)" + RC]

# Core methods drawn in the comparison figure (without RC; the RC version goes below)
FIGURE_METHODS = ["CSP+LDA", "MDM", "TS+LR", "SPDNet (Euclidean opt.)", "SPDNet"]


def paired_comparisons():
    """(family, A, B): is A better than B? Each pair is tested only if both methods exist."""
    pairs = []
    for suffix, condition in [("", "without RC"), (RC, "with RC")]:
        for m in ["MDM", "TS+LR", "SPDNet"]:
            pairs.append((f"Riemannian vs Euclidean ({condition})", m + suffix, "CSP+LDA" + suffix))
        pairs.append((f"Network vs classical ({condition})", "SPDNet" + suffix, "TS+LR" + suffix))
        pairs.append((f"Network vs classical ({condition})", "SPDNet" + suffix, "MDM" + suffix))
        pairs.append((f"Ablation: optimisation of W ({condition})",
                      "SPDNet" + suffix, "SPDNet (Euclidean opt.)" + suffix))
        pairs.append((f"Batch norm ({condition})", "SPDNet + BN" + suffix, "SPDNet" + suffix))
    pairs.append(("Network vs classical (with RC)", "SPDNet" + RC, "TS+LR (tuned C)" + RC))
    pairs.append(("Network vs classical (with RC)",
                  "SPDNet" + RC + " (ensemble of 5 seeds)", "TS+LR" + RC))
    for m in ["CSP+LDA", "MDM", "TS+LR", "SPDNet (Euclidean opt.)", "SPDNet", "SPDNet + BN"]:
        pairs.append(("Effect of session recentering", m + RC, m))
    pairs.append(("Seed ensemble", "SPDNet" + RC + " (ensemble of 5 seeds)", "SPDNet" + RC))
    return pairs


# ---------------------------------------------------------------- loading


def load_results():
    """All result files together, one row per method, subject and seed."""
    tables = []
    for path in sorted(RESULTS.glob("*.csv")):
        table = pd.read_csv(path)
        if not set(REQUIRED).issubset(table.columns):
            print(f"  skipped {path.name}: it needs the columns {REQUIRED}")
            continue
        if "method" not in table.columns:
            if path.name not in FILE_METHODS:
                print(f"  skipped {path.name}: no 'method' column and not in FILE_METHODS")
                continue
            table["method"] = FILE_METHODS[path.name]
        if "seed" not in table.columns:
            table["seed"] = 0
        table["method"] = table["method"].replace(ALIASES)
        tables.append(table[["method", "subject", "seed", "accuracy", "kappa"]])
    if not tables:
        raise SystemExit("No results in results/ yet.")
    results = pd.concat(tables, ignore_index=True)

    duplicated = results.duplicated(["method", "subject", "seed"])
    if duplicated.any():
        names = sorted(results.loc[duplicated, "method"].unique())
        raise SystemExit(f"The same method appears in two files: {names}")
    return results


def check(results):
    """Warn about missing methods or subjects, so the final run does not go unnoticed."""
    present = set(results["method"])
    missing = [m for m in EXPECTED if m not in present]
    if missing:
        print("  WARNING, missing for the final table: " + ", ".join(missing))
    for method, runs in results.groupby("method"):
        n = runs["subject"].nunique()
        if n != N_SUBJECTS:
            print(f"  WARNING, {method} has {n} subjects instead of {N_SUBJECTS}")


def ordered_blocks(methods):
    """Blocks of the table with only the methods present, unknown methods at the end."""
    blocks, listed = {}, set()
    for name, block in BLOCKS.items():
        listed.update(block)
        present = [m for m in block if m in methods]
        if present:
            blocks[name] = present
    others = sorted(set(methods) - listed)
    if others:
        blocks["Other"] = others
    return blocks


# ---------------------------------------------------------------- tables


def wilcoxon_p(a, b):
    """Two-sided paired Wilcoxon signed-rank test over subjects."""
    if np.allclose(a, b):
        return 1.0
    return wilcoxon(a, b).pvalue


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error over all tests)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def method_table(results, accuracy, kappa, methods):
    """Rows = methods; columns = subjects, mean, std over subjects, kappa, std over seeds."""
    table = accuracy[methods].T.round(1)
    table.columns = [f"S{s}" for s in table.columns]
    table["Mean"] = accuracy[methods].mean().round(1)
    table["Std"] = accuracy[methods].std().round(1)
    table["Kappa"] = kappa[methods].mean().round(3)
    seed_std = []
    for m in methods:
        runs = results[results["method"] == m]
        if runs["seed"].nunique() > 1:
            seed_std.append(round(100 * runs.groupby("subject")["accuracy"].std().mean(), 1))
        else:
            seed_std.append(np.nan)
    table["Seed std"] = seed_std
    return table


def pairs_table(accuracy):
    """Paired tests between equivalent methods (same preprocessing and same condition)."""
    rows = []
    for family, a, b in paired_comparisons():
        if a not in accuracy or b not in accuracy:
            continue
        diff = accuracy[a] - accuracy[b]
        rows.append({"Comparison": family, "A": a, "B": b,
                     "Mean diff (A - B)": round(diff.mean(), 1) + 0.0,
                     "A better in": f"{(diff > 0).sum()}/{len(diff)}",
                     "p": wilcoxon_p(accuracy[a], accuracy[b])})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Holm correction within each question (both conditions together), rows grouped by question
    question = table["Comparison"].str.replace(r" \((with|without) RC\)$", "", regex=True)
    table["p (Holm)"] = table.groupby(question)["p"].transform(lambda p: holm(p.values)).round(3)
    table["p"] = table["p"].round(3)
    order = {q: i for i, q in enumerate(dict.fromkeys(question))}
    table = table.iloc[np.argsort(question.map(order).values, kind="stable")]
    return table.reset_index(drop=True)


def markdown_table(table, first_header="", index=True):
    columns = list(table.columns)
    header = "| " + " | ".join(([first_header] if index else []) + columns) + " |"
    separator = "|---" * (len(columns) + int(index)) + "|"
    lines = [header, separator]
    for name, row in table.iterrows():
        cells = ["-" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(([str(name)] if index else []) + cells) + " |")
    return lines


def to_markdown(blocks, table, pairs):
    lines = ["# Comparison of methods",
             "",
             "Test accuracy (%) on session 2, training on session 1. "
             "Chance = 25 %; significance threshold ~29 % with 288 trials. "
             "Methods with several seeds: mean over seeds; *Seed std* = mean over subjects "
             "of the std over seeds.",
             ""]
    for name, methods in blocks.items():
        lines += [f"## {name}", ""] + markdown_table(table.loc[methods], "Method") + [""]
    lines += ["## Paired comparisons",
              "",
              "Two-sided paired Wilcoxon signed-rank test over the 9 subjects, on accuracy "
              "(for 4 balanced classes kappa = (acc - 0.25) / 0.75, so the test on kappa is "
              "identical). With 9 subjects the smallest possible p is 0.004. *p (Holm)* corrects "
              "for the comparisons that answer the same question (same name, both conditions).",
              ""]
    if pairs.empty:
        lines.append("No pair of comparable methods yet.")
    else:
        lines += markdown_table(pairs, index=False)
    return "\n".join(lines) + "\n"


def tex_escape(text):
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def to_latex(blocks, table, pairs):
    """Two booktabs tables; the best mean of each block in bold."""
    subjects = [c for c in table.columns if c.startswith("S") and c[1:].isdigit()]
    columns = subjects + ["Mean", "Std", "Kappa"]
    out = [r"% Generated by scripts/make_table.py -- do not edit by hand, re-run the script.",
           r"% Needs \usepackage{booktabs} (and graphicx for \resizebox).",
           r"\begin{table}[ht]", r"\centering",
           r"\caption{Test accuracy (\%) on session 2 for each subject, mean and standard "
           r"deviation over subjects, and mean Cohen's $\kappa$. Methods with several seeds are "
           r"averaged over 5 seeds. Chance level is 25\,\% (significance threshold $\approx$ 29\,\%). "
           r"The best mean of each block is in bold.}",
           r"\label{tab:comparison}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
           "Method & " + " & ".join(columns) + r" \\"]
    for name, methods in blocks.items():
        best = table.loc[methods, "Mean"].max()
        out += [r"\midrule", rf"\multicolumn{{{len(columns) + 1}}}{{l}}{{\emph{{{tex_escape(name)}}}}} \\"]
        for m in methods:
            cells = []
            for c in columns:
                v = table.loc[m, c]
                cell = "--" if pd.isna(v) else (f"{v:.3f}" if c == "Kappa" else f"{v:.1f}")
                if c == "Mean" and v == best and len(methods) > 1:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
            out.append(f"{tex_escape(m)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]

    if not pairs.empty:
        out += [r"\begin{table}[ht]", r"\centering",
                r"\caption{Paired comparisons between equivalent methods: mean difference in "
                r"accuracy (percentage points), number of subjects where A is better, and two-sided "
                r"Wilcoxon signed-rank test over the 9 subjects, raw and Holm-corrected within each "
                r"question (with 9 subjects the smallest possible $p$ is 0.004).}",
                r"\label{tab:paired}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lllrrrr}", r"\toprule",
                r"Comparison & A & B & $\Delta$ & A better & $p$ & $p$ (Holm) \\", r"\midrule"]
        for _, r in pairs.iterrows():
            out.append(" & ".join([tex_escape(r["Comparison"]), tex_escape(r["A"]),
                                   tex_escape(r["B"]), f"{r['Mean diff (A - B)']:+.1f}",
                                   r["A better in"], f"{r['p']:.3f}", f"{r['p (Holm)']:.3f}"])
                       + r" \\")
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- figures


def bars(ax, accuracy, methods, title):
    data = pd.concat([accuracy[methods], accuracy[methods].mean().to_frame("Mean").T])
    x = np.arange(len(data))
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        base = method.replace(RC, "")  # same colour for a method with and without recentering
        colour = f"C{FIGURE_METHODS.index(base)}" if base in FIGURE_METHODS else None
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, data[method], width,
               label=base, color=colour)
    ax.axhline(25, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"S{s}" for s in accuracy.index] + ["Mean"])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11)


def comparison_figure(accuracy, path):
    """Top: without session recentering. Bottom: with it (if any method has it)."""
    panels = [([m for m in FIGURE_METHODS if m in accuracy], "Without session recentering"),
              ([m + RC for m in FIGURE_METHODS if m + RC in accuracy], "With session recentering")]
    panels = [(methods, title) for methods, title in panels if methods]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.8 * len(panels)), squeeze=False)
    for ax, (methods, title) in zip(axes[:, 0], panels):
        bars(ax, accuracy, methods, title)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(ncol=len(methods), fontsize=8, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_figure(accuracy, path):
    """One point per subject: Euclidean (x) vs Riemannian (y) optimisation of W, per condition."""
    panels = [(s, t) for s, t in [("", "Without session recentering"),
                                  (RC, "With session recentering")]
              if "SPDNet" + s in accuracy and "SPDNet (Euclidean opt.)" + s in accuracy]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5), squeeze=False)
    for ax, (suffix, title) in zip(axes[0], panels):
        euclid = accuracy["SPDNet (Euclidean opt.)" + suffix]
        riemann = accuracy["SPDNet" + suffix]
        ax.scatter(euclid, riemann)
        for subject in accuracy.index:
            ax.annotate(f"S{subject}", (euclid[subject], riemann[subject]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.plot([0, 100], [0, 100], color="grey", linestyle="--")
        ax.set_xlim(35, 95)
        ax.set_ylim(35, 95)
        ax.set_aspect("equal")
        ax.set_xlabel("Euclidean optimisation of W (%)")
        ax.set_ylabel("Riemannian optimisation of W (%)")
        ax.set_title(f"{title}\n(above the diagonal: Riemannian better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def recentering_figure(accuracy, path):
    """Mean accuracy of each method without and with session recentering."""
    methods = [m for m in FIGURE_METHODS + ["SPDNet + BN"] if m in accuracy and m + RC in accuracy]
    if not methods:
        return False
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 4))
    before = accuracy[methods].mean()
    after = accuracy[[m + RC for m in methods]].mean()
    ax.bar(x - 0.2, before, 0.4, label="without recentering")
    ax.bar(x + 0.2, after, 0.4, label="with session recentering")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i + 0.2, a + 0.8, f"{a - b:+.1f}", ha="center", fontsize=8)
    ax.axhline(25, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x, methods, fontsize=8)
    ax.set_ylabel("mean test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main


def main():
    print("Loading results/")
    results = load_results()
    check(results)

    # Mean over seeds: one value per method and subject
    per_subject = results.groupby(["method", "subject"])[["accuracy", "kappa"]].mean().reset_index()
    accuracy = 100 * per_subject.pivot(index="subject", columns="method", values="accuracy")
    kappa = per_subject.pivot(index="subject", columns="method", values="kappa")

    blocks = ordered_blocks(accuracy.columns)
    methods = [m for block in blocks.values() for m in block]
    table = method_table(results, accuracy, kappa, methods)
    pairs = pairs_table(accuracy)

    markdown = to_markdown(blocks, table, pairs)
    (RESULTS / "comparison_table.md").write_text(markdown)
    (RESULTS / "comparison_table.tex").write_text(to_latex(blocks, table, pairs))
    print()
    print(markdown)

    FIGURES.mkdir(exist_ok=True)
    comparison_figure(accuracy, FIGURES / "comparison.png")
    saved = ["comparison.png"]
    if ablation_figure(accuracy, FIGURES / "ablation.png"):
        saved.append("ablation.png")
    if recentering_figure(accuracy, FIGURES / "recentering.png"):
        saved.append("recentering.png")
    print("Saved results/comparison_table.md, results/comparison_table.tex and "
          "results/figures/" + ", ".join(saved))


if __name__ == "__main__":
    main()