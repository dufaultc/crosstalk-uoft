"""Evaluation tools for binary classification and ranking metrics with bootstrapping."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ----------------------------------------------------------------------
# 1. Point Metrics
# ----------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Computes standard accuracy."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Computes balanced accuracy."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return balanced_accuracy_score(y_true, y_pred)


def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Computes ROC Area Under Curve (AUC)."""
    return roc_auc_score(y_true, y_pred_proba)


def precision(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Computes precision score."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Computes recall score."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return recall_score(y_true, y_pred)


def hits_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
    """Computes Hits@K ranking metric."""
    if k <= 0:
        raise ValueError("k must be positive")

    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0:
        return 0.0

    hits = 0
    for i in pos_idx:
        ps = y_pred_proba[i]
        r = np.sum(y_pred_proba > ps) + 1
        if r <= k:
            hits += 1

    return hits / len(pos_idx)


def precision_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
    """Computes Precision@K ranking metric."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(y_true):
        raise ValueError("k cannot be larger than the number of samples")

    topk = np.argsort(y_pred_proba)[-k:]
    tp = np.sum(y_true[topk] == 1)
    return tp / k


def mrr(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) ranking metric."""
    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0:
        return 0.0

    rr = []
    for i in pos_idx:
        ps = y_pred_proba[i]
        r = np.sum(y_pred_proba > ps) + 1
        rr.append(1.0 / r)
    return np.mean(rr)


# ----------------------------------------------------------------------
# 2. Bootstrapping & Confidence Intervals
# ----------------------------------------------------------------------

def bootstrap_estimates(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_fn,
    n_iterations: int = 1000,
    random_seed: int = 42,
    **metric_kwargs,
) -> np.ndarray:
    """Generates bootstrap estimates for a given metric function using row resampling."""
    n_samples = len(y_true)
    metric_values = []
    rng = np.random.default_rng(random_seed)

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred_proba[indices]
        try:
            val = metric_fn(y_true_bootstrap, y_pred_bootstrap, **metric_kwargs)
            metric_values.append(val)
        except Exception:
            continue

    return np.array(metric_values)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_fn,
    ci: float = 0.95,
    n_iterations: int = 1000,
    random_seed: int = 42,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """Calculates confidence intervals for a given metric using bootstrap resampling.

    Returns:
        mean_estimate, lower_bound, upper_bound
    """
    values = bootstrap_estimates(
        y_true,
        y_pred_proba,
        metric_fn,
        n_iterations=n_iterations,
        random_seed=random_seed,
        **metric_kwargs,
    )
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    lower_percentile = (1.0 - ci) / 2.0 * 100.0
    upper_percentile = 100.0 - lower_percentile
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    return float(np.mean(values)), float(lower_bound), float(upper_bound)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    compute_ci: bool = True,
    ci: float = 0.95,
    n_iterations: int = 1000,
) -> dict[str, dict[str, float]]:
    """Evaluates prediction probabilities and returns point estimates and optionally confidence intervals."""
    metrics = {
        "ROC-AUC": (roc_auc, {}),
        "Accuracy": (accuracy, {"threshold": threshold}),
        "Balanced Accuracy": (balanced_accuracy, {"threshold": threshold}),
        "Precision": (precision, {"threshold": threshold}),
        "Recall": (recall, {"threshold": threshold}),
        "MRR": (mrr, {}),
        "Precision@5": (precision_at_k, {"k": 5}),
        "Hits@5": (hits_at_k, {"k": 5}),
        "Precision@10": (precision_at_k, {"k": 10}),
        "Hits@10": (hits_at_k, {"k": 10}),
        "Precision@30": (precision_at_k, {"k": 30}),
        "Hits@30": (hits_at_k, {"k": 30}),
    }

    results = {}
    for name, (metric_fn, kwargs) in metrics.items():
        point_estimate = metric_fn(y_true, y_pred_proba, **kwargs)
        if compute_ci:
            mean_est, lower, upper = bootstrap_ci(
                y_true,
                y_pred_proba,
                metric_fn,
                ci=ci,
                n_iterations=n_iterations,
                **kwargs,
            )
            results[name] = {"val": point_estimate, "lower": lower, "upper": upper}
        else:
            results[name] = {"val": point_estimate, "lower": np.nan, "upper": np.nan}

    return results
