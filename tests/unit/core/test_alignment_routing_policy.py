"""CPU-only tests for dual-threshold binary alignment routing."""

import math
import unittest

from grounding_control.core.calibration import (
    AlignmentScoreCalibrationError,
    AlignmentScoreCalibrator,
)
from grounding_control.core.alignment_policy import AlignmentRoutingPolicy
from grounding_control.contracts import AlignmentVerifierOutput


def _output(score):
    return AlignmentVerifierOutput(
        alignment_score=score,
        score_semantics='calibrated_alignment_probability',
        score_kind='calibrated_probability',
        metadata={'backend': 'test'},
    )


def _raw_output(score, kind='iou_proxy'):
    return AlignmentVerifierOutput(
        alignment_score=score,
        score_semantics=f'test_{kind}',
        score_kind=kind,
        metadata={'backend': 'raw-test'},
    )


class _FixedCalibrator(AlignmentScoreCalibrator):
    def __init__(self, result=0.85):
        super().__init__('iou_proxy', 'fixed-test-calibrator')
        self.result = result
        self.calls = []

    def calibrate_score(self, score, metadata):
        self.calls.append((score, dict(metadata)))
        return self.result


class AlignmentRoutingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
        )

    def test_accept_boundary_is_inclusive(self):
        decision = self.policy.decide(_output(0.8))
        self.assertEqual(decision.band, 'accept')
        self.assertEqual(decision.system_action, 'accept_candidate')
        self.assertTrue(decision.candidate_accepted)
        self.assertFalse(decision.requires_grounder)
        self.assertEqual(decision.routing_reason, 'high_alignment_accept')

    def test_reject_boundary_is_inclusive(self):
        decision = self.policy.decide(_output(0.2))
        self.assertEqual(decision.band, 'reject')
        self.assertEqual(decision.system_action, 'call_grounder')
        self.assertTrue(decision.requires_grounder)
        self.assertEqual(decision.routing_reason, 'low_alignment_reject')

    def test_uncertain_band_calls_grounder_but_remains_distinct(self):
        decision = self.policy.decide(_output(0.5))
        self.assertEqual(decision.band, 'uncertain')
        self.assertEqual(decision.system_action, 'call_grounder')
        self.assertEqual(
            decision.routing_reason,
            'uncertainty_grounder_fallback',
        )

    def test_verifier_failure_is_separate_and_fails_open(self):
        decision = self.policy.decide(AlignmentVerifierOutput.unknown(
            error='timeout',
            metadata={'worker': 'qwen'},
        ))
        self.assertEqual(decision.band, 'verifier_failure')
        self.assertEqual(decision.system_action, 'accept_candidate')
        self.assertTrue(decision.verifier_abstained)
        self.assertEqual(
            decision.routing_reason,
            'verifier_failure_fail_open',
        )
        self.assertEqual(
            decision.metadata['verifier_failure_action'],
            'accept_candidate',
        )

    def test_decision_preserves_score_semantics_and_thresholds(self):
        decision = self.policy.decide(_output(0.5))
        self.assertEqual(
            decision.score_semantics,
            'calibrated_alignment_probability',
        )
        self.assertEqual(decision.reject_threshold, 0.2)
        self.assertEqual(decision.accept_threshold, 0.8)
        self.assertEqual(
            decision.metadata['policy'],
            'binary_dual_threshold_grounder_v1',
        )
        self.assertNotIn('verifier_metadata', decision.metadata)
        self.assertEqual(decision.score_kind, 'calibrated_probability')
        self.assertEqual(
            decision.metadata['score_policy_mode'],
            'formal_probability',
        )

    def test_formal_probability_policy_rejects_raw_proxy_without_calibrator(self):
        with self.assertRaisesRegex(ValueError, 'score kind mismatch'):
            self.policy.decide(_raw_output(0.9, 'iou_proxy'))
        with self.assertRaisesRegex(ValueError, 'score kind mismatch'):
            self.policy.decide(
                _raw_output(0.9, 'self_reported_probability')
            )
        with self.assertRaisesRegex(ValueError, 'score kind mismatch'):
            self.policy.decide(_raw_output(1.0, 'hard_oracle_label'))

    def test_explicit_raw_policy_requires_and_records_exact_scale(self):
        raw_policy = AlignmentRoutingPolicy.explicit_raw(
            reject_threshold=0.2,
            accept_threshold=0.8,
            score_kind='iou_proxy',
        )
        decision = raw_policy.decide(_raw_output(0.8, 'iou_proxy'))
        self.assertEqual(decision.band, 'accept')
        self.assertEqual(decision.score_kind, 'iou_proxy')
        self.assertEqual(
            decision.metadata['score_policy_mode'],
            'explicit_raw_threshold',
        )
        self.assertFalse(decision.metadata['calibration_applied'])
        with self.assertRaisesRegex(ValueError, 'score kind mismatch'):
            raw_policy.decide(_raw_output(1.0, 'hard_oracle_label'))

    def test_hard_oracle_has_an_explicit_raw_policy_path(self):
        policy = AlignmentRoutingPolicy.explicit_raw(
            reject_threshold=0.2,
            accept_threshold=0.8,
            score_kind='hard_oracle_label',
        )
        self.assertEqual(
            policy.decide(_raw_output(1.0, 'hard_oracle_label')).band,
            'accept',
        )
        self.assertEqual(
            policy.decide(_raw_output(0.0, 'hard_oracle_label')).band,
            'reject',
        )

    def test_calibrator_converts_proxy_before_probability_thresholds(self):
        calibrator = _FixedCalibrator(result=0.85)
        policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
            calibrator=calibrator,
        )
        decision = policy.decide(_raw_output(0.3, 'iou_proxy'))
        self.assertEqual(decision.band, 'accept')
        self.assertEqual(decision.alignment_score, 0.85)
        self.assertEqual(decision.score_kind, 'calibrated_probability')
        self.assertEqual(
            decision.score_semantics,
            'calibrated_alignment_probability:fixed-test-calibrator',
        )
        self.assertEqual(calibrator.calls, [(0.3, {'backend': 'raw-test'})])
        self.assertTrue(decision.metadata['calibration_applied'])
        self.assertEqual(
            decision.metadata['input_alignment_score'],
            0.3,
        )
        self.assertEqual(
            decision.metadata['effective_alignment_score'],
            0.85,
        )
        self.assertEqual(
            decision.metadata['calibrator_id'],
            'fixed-test-calibrator',
        )

    def test_calibrator_validates_source_and_probability_range(self):
        policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
            calibrator=_FixedCalibrator(result=1.1),
        )
        with self.assertRaisesRegex(
                AlignmentScoreCalibrationError,
                'finite probability'):
            policy.decide(_raw_output(0.3, 'iou_proxy'))

        source_mismatch = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
            calibrator=_FixedCalibrator(result=0.7),
        )
        with self.assertRaisesRegex(
                AlignmentScoreCalibrationError,
                'source kind mismatch'):
            source_mismatch.decide(
                _raw_output(0.7, 'self_reported_probability')
            )

    def test_threshold_validation(self):
        invalid_pairs = [
            (-0.1, 0.8),
            (0.2, 1.1),
            (0.8, 0.8),
            (0.9, 0.8),
            (math.nan, 0.8),
            (0.2, math.inf),
            (False, 0.8),
        ]
        for reject, accept in invalid_pairs:
            with self.subTest(reject=reject, accept=accept):
                with self.assertRaises(ValueError):
                    AlignmentRoutingPolicy(reject, accept)

        with self.assertRaisesRegex(ValueError, 'required_score_kind'):
            AlignmentRoutingPolicy(
                0.2,
                0.8,
                required_score_kind='unknown_kind',
            )
        with self.assertRaisesRegex(ValueError, 'explicit_raw'):
            AlignmentRoutingPolicy.explicit_raw(
                0.2,
                0.8,
                score_kind='calibrated_probability',
            )

    def test_policy_rejects_legacy_four_way_or_arbitrary_inputs(self):
        with self.assertRaises(TypeError):
            self.policy.decide({'alignment_score': 0.9})


if __name__ == '__main__':
    unittest.main()
