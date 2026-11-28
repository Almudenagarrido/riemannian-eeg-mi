"""Within-subject, cross-session evaluation (train on session 1, test on session 2).

Run the classical Riemannian pipelines on the 9 subjects and save the scores:

    uv run python -m riemannian_eeg.classical.evaluation
"""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final, Protocol, Self

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score

from riemannian_eeg.classical.features import Epochs, Labels
from riemannian_eeg.classical.filterbank import FB_PIPELINES, load_train_test_fb
from riemannian_eeg.classical.pipelines import PIPELINES
from riemannian_eeg.data import load_train_test

Loader = Callable[[int], tuple[Epochs, Labels, Epochs, Labels]]

SUBJECTS: Final[range] = range(1, 10)
# <repo>/src/riemannian_eeg/classical/evaluation.py -> <repo>/results
RESULTS_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "results"


class Classifier(Protocol):
    """Anything with scikit-learn's fit/predict interface on raw epochs."""

    def fit(self, X: Epochs, y: Labels) -> Self: ...

    def predict(self, X: Epochs) -> Labels: ...


def evaluate(
    methods: dict[str, Callable[[], Classifier]],
    subjects: Iterable[int] = SUBJECTS,
    load: Loader = load_train_test,
) -> pd.DataFrame:
    """Fit each method on session 1 and score it on session 2, per subject.

    Each subject's data is loaded once and shared by all methods.

    Parameters
    ----------
    methods : dict
        Method name -> factory returning a fresh, unfitted classifier.
    subjects : iterable of int
        Subjects to evaluate (1 to 9).
    load : callable
        Subject -> (X_train, y_train, X_test, y_test).

    Returns
    -------
    scores : pandas.DataFrame
        One row per (method, subject) with ``accuracy`` and ``kappa``.
    """
    rows = []
    for subject in subjects:
        X_tr, y_tr, X_te, y_te = load(subject)
        for name, factory in methods.items():
            y_pred = factory().fit(X_tr, y_tr).predict(X_te)
            rows.append({
                "method": name,
                "subject": subject,
                "accuracy": accuracy_score(y_te, y_pred),
                "kappa": cohen_kappa_score(y_te, y_pred),
            })
            print(f"Subject {subject} | {name:<6} acc = {rows[-1]['accuracy']:.3f}")
    return pd.DataFrame(rows)


def evaluate_all(subjects: Iterable[int] = SUBJECTS) -> pd.DataFrame:
    """Evaluate the single-band pipelines and the filter bank pipelines."""
    return pd.concat(
        [
            evaluate(PIPELINES, subjects),
            evaluate(FB_PIPELINES, subjects, load=load_train_test_fb),
        ],
        ignore_index=True,
    )


# Pairs compared with the Wilcoxon test: each adds one ingredient to the previous one.
COMPARISONS: Final[list[tuple[str, str]]] = [
    ("TS+LR", "MDM"),
    ("RC+MDM", "MDM"),
    ("RC+TS+LR", "TS+LR"),
    ("RC+TS+LRCV", "RC+TS+LR"),
    ("FB+RC+TS+LRCV", "RC+TS+LRCV"),
]


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Wide table: subjects as rows, (method, metric) as columns, plus a mean row."""
    table = scores.pivot(index="subject", columns="method", values=["accuracy", "kappa"])
    table = table.swaplevel(axis=1).sort_index(axis=1)
    table.loc["mean"] = table.mean()
    return table


def compare(scores: pd.DataFrame, a: str, b: str, metric: str = "kappa") -> tuple[float, float]:
    """Wilcoxon signed-rank test between two methods, paired by subject.

    Returns
    -------
    statistic, p_value : float
    """
    wide = scores.pivot(index="subject", columns="method", values=metric)
    result = wilcoxon(wide[a], wide[b])
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    scores = evaluate_all()
    RESULTS_DIR.mkdir(exist_ok=True)
    scores.to_csv(RESULTS_DIR / "riemannian_classical.csv", index=False)

    print(summary_table(scores).round(3))
    print("\nWilcoxon signed-rank tests on kappa:")
    for a, b in COMPARISONS:
        stat, p = compare(scores, a, b)
        print(f"  {a:>13} vs {b:<10}: W = {stat:.1f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
