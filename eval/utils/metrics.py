from typing import Iterable, List


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1_for_label(y_true: Iterable[int], y_pred: Iterable[int], label: int) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for truth, prediction in zip(y_true, y_pred):
        if prediction == label and truth == label:
            true_positive += 1
        elif prediction == label and truth != label:
            false_positive += 1
        elif prediction != label and truth == label:
            false_negative += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_macro_f1(y_true: List[int], y_pred: List[int], labels: List[int]) -> float:
    if not y_true:
        return 0.0
    return sum(_f1_for_label(y_true, y_pred, label) for label in labels) / len(labels)
