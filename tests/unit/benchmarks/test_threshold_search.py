"""CPU-only tests for Grounding DINO dev threshold selection."""

import argparse
import unittest

from grounding_control.benchmarks.gqa_controlled.tune_grounding_dino_thresholds import (
    evaluate_box_thresholds,
    parse_box_thresholds,
    records_at_box_threshold,
    select_best_evaluation,
)


def _row(expected, action=None, score=None):
    geometry = {'action': action} if action is not None else None
    return {
        'expected_routing_status': expected,
        'predicted_routing_status': action,
        'confidence': score,
        'verifier_metadata': {
            'selected_grounding_score': score,
            'geometry': geometry,
        },
    }


class GroundingDinoThresholdSearchTests(unittest.TestCase):
    def test_threshold_parser_sorts_and_deduplicates(self):
        self.assertEqual(
            parse_box_thresholds('0.3, 0.1,0.30,0.2'),
            [0.1, 0.2, 0.3],
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_box_thresholds('0.2,invalid')
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_box_thresholds('1.1')

    def test_replay_uses_strict_huggingface_score_threshold(self):
        rows = [
            _row('no_action', 'no_action', 0.30),
            _row('relocate', 'relocate', 0.31),
        ]
        replayed = records_at_box_threshold(rows, 0.30)
        self.assertIsNone(replayed[0]['predicted_routing_status'])
        self.assertEqual(
            replayed[1]['predicted_routing_status'],
            'relocate',
        )

    def test_evaluation_counts_missing_detection_as_failure(self):
        rows = [
            _row('no_action', 'no_action', 0.20),
            _row('relocate', 'relocate', 0.40),
            _row('expand', 'relocate', 0.10),
            _row('tighten'),
        ]
        evaluations = evaluate_box_thresholds(rows, [0.15, 0.30])
        low, high = evaluations

        self.assertEqual(low['correct'], 2)
        self.assertEqual(low['localization_failure_count'], 2)
        self.assertEqual(high['correct'], 1)
        self.assertEqual(high['localization_failure_count'], 3)

    def test_best_selection_prefers_metric_then_coverage(self):
        evaluations = [
            {
                'box_threshold': 0.10,
                'macro_f1': 0.60,
                'accuracy': 0.70,
                'localization_success_rate': 0.90,
            },
            {
                'box_threshold': 0.20,
                'macro_f1': 0.61,
                'accuracy': 0.68,
                'localization_success_rate': 0.80,
            },
        ]
        self.assertEqual(
            select_best_evaluation(evaluations)['box_threshold'],
            0.20,
        )

        tied = [
            {
                'box_threshold': 0.10,
                'macro_f1': 0.60,
                'accuracy': 0.70,
                'localization_success_rate': 0.90,
            },
            {
                'box_threshold': 0.20,
                'macro_f1': 0.60,
                'accuracy': 0.70,
                'localization_success_rate': 0.80,
            },
        ]
        self.assertEqual(
            select_best_evaluation(tied)['box_threshold'],
            0.10,
        )


if __name__ == '__main__':
    unittest.main()
