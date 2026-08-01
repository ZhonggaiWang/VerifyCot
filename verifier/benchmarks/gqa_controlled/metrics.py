"""Dependency-free metrics for binary and four-way verifier predictions."""

from typing import Any, Dict, Iterable, List, Mapping, Optional

from ...contracts import ACTION_NAMES
from .labels import CONTROLLED_STATUSES


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_binary_alignment_metrics(
        records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Score the crop-only aligned-vs-misaligned ablation."""

    rows = list(records)
    parsed = [
        row for row in rows
        if isinstance(row.get('predicted_alignment'), bool)
    ]
    true_positive = sum(
        row['expected_alignment'] is False
        and row['predicted_alignment'] is False
        for row in parsed
    )
    false_positive = sum(
        row['expected_alignment'] is True
        and row['predicted_alignment'] is False
        for row in parsed
    )
    false_negative = sum(
        row['expected_alignment'] is False
        and row['predicted_alignment'] is True
        for row in parsed
    )
    true_negative = sum(
        row['expected_alignment'] is True
        and row['predicted_alignment'] is True
        for row in parsed
    )
    correct = sum(
        isinstance(row.get('predicted_alignment'), bool)
        and row['predicted_alignment'] == row['expected_alignment']
        for row in rows
    )
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    by_expected_status = {}
    for status in CONTROLLED_STATUSES:
        status_rows = [
            row for row in rows
            if row.get('expected_status') == status
        ]
        status_correct = sum(
            isinstance(row.get('predicted_alignment'), bool)
            and row['predicted_alignment'] == row['expected_alignment']
            for row in status_rows
        )
        by_expected_status[status] = {
            'total': len(status_rows),
            'correct': status_correct,
            'accuracy': _safe_divide(status_correct, len(status_rows)),
            'predicted_aligned': sum(
                row.get('predicted_alignment') is True
                for row in status_rows
            ),
            'predicted_misaligned': sum(
                row.get('predicted_alignment') is False
                for row in status_rows
            ),
            'no_prediction': sum(
                not isinstance(row.get('predicted_alignment'), bool)
                for row in status_rows
            ),
        }
    return {
        'total': len(rows),
        'parsed_count': len(parsed),
        'parse_success_rate': _safe_divide(len(parsed), len(rows)),
        'correct': correct,
        'end_to_end_accuracy': _safe_divide(correct, len(rows)),
        'positive_class': 'misaligned',
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'by_expected_status': by_expected_status,
    }


def compute_routing_metrics(
        records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute no_action/relocate/expand/tighten routing metrics."""

    rows = list(records)
    labels = list(ACTION_NAMES)
    confusion_labels = labels + ['parse_failure']
    confusion = {
        expected: {predicted: 0 for predicted in confusion_labels}
        for expected in labels
    }
    normalized = []
    confidences: List[float] = []
    correct_confidences: List[float] = []
    incorrect_confidences: List[float] = []
    for row in rows:
        expected = row.get('expected_routing_status')
        if expected not in ACTION_NAMES:
            raise ValueError(
                f'unknown expected_routing_status: {expected!r}'
            )
        predicted = row.get('predicted_routing_status')
        if predicted not in ACTION_NAMES:
            predicted = None
        confusion[expected][predicted or 'parse_failure'] += 1
        correct = predicted == expected
        confidence = row.get('confidence')
        if (
            predicted is not None
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
        ):
            confidence = float(confidence)
            confidences.append(confidence)
            (correct_confidences if correct else incorrect_confidences).append(
                confidence
            )
        normalized.append((expected, predicted))

    per_class: Dict[str, Dict[str, float]] = {}
    for status in labels:
        true_positive = sum(
            expected == status and predicted == status
            for expected, predicted in normalized
        )
        false_positive = sum(
            expected != status and predicted == status
            for expected, predicted in normalized
        )
        false_negative = sum(
            expected == status and predicted != status
            for expected, predicted in normalized
        )
        support = sum(expected == status for expected, _ in normalized)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_class[status] = {
            'support': support,
            'true_positive': true_positive,
            'precision': precision,
            'recall': recall,
            'f1': _safe_divide(
                2 * precision * recall,
                precision + recall,
            ),
        }

    total = len(rows)
    parsed_count = sum(predicted is not None for _, predicted in normalized)
    correct_count = sum(
        expected == predicted for expected, predicted in normalized
    )
    return {
        'total': total,
        'parsed_count': parsed_count,
        'parse_success_rate': _safe_divide(parsed_count, total),
        'runtime_or_parse_failure_count': total - parsed_count,
        'four_way': {
            'correct': correct_count,
            'accuracy': _safe_divide(correct_count, total),
            'macro_precision': _mean([
                per_class[status]['precision'] for status in labels
            ]),
            'macro_recall': _mean([
                per_class[status]['recall'] for status in labels
            ]),
            'macro_f1': _mean([
                per_class[status]['f1'] for status in labels
            ]),
            'per_class': per_class,
            'confusion_matrix': confusion,
        },
        'confidence': {
            'count': len(confidences),
            'mean': _mean(confidences),
            'mean_when_correct': _mean(correct_confidences),
            'mean_when_incorrect': _mean(incorrect_confidences),
            'note': 'self-reported model confidence; not calibrated probability',
        },
    }
