"""Tests for the canonical binary alignment-verifier contract."""

import math
import unittest

from grounding_control.contracts import (
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from grounding_control.contracts.alignment_verifier import (
    ALIGNMENT_SCORE_KINDS,
    alignment_score_kind_from_legacy_semantics,
)


class _Backend(AlignmentVerifierBackend):
    def verify_alignment(self, request):
        return AlignmentVerifierOutput(
            alignment_score=0.75,
            score_semantics='test_probability',
            score_kind='calibrated_probability',
        )


class AlignmentVerifierOutputTests(unittest.TestCase):
    def test_score_direction_and_wire_round_trip(self):
        output = AlignmentVerifierOutput(
            alignment_score=0.75,
            score_semantics=' calibrated_alignment_probability ',
            metadata={'backend': 'test'},
        )
        self.assertEqual(output.alignment_score, 0.75)
        self.assertEqual(output.score_kind, 'calibrated_probability')
        self.assertEqual(
            output.score_semantics,
            'calibrated_alignment_probability',
        )
        self.assertEqual(
            output.as_dict()['verifier_output_schema'],
            ALIGNMENT_OUTPUT_SCHEMA,
        )
        self.assertEqual(
            AlignmentVerifierOutput.from_dict(output.as_dict()),
            output,
        )
        self.assertEqual(
            output.as_dict()['alignment_score_kind'],
            'calibrated_probability',
        )

    def test_non_abstained_output_requires_valid_score(self):
        invalid_scores = [None, -0.01, 1.01, math.inf, math.nan, True]
        for score in invalid_scores:
            with self.subTest(score=score), self.assertRaises(ValueError):
                AlignmentVerifierOutput(
                    alignment_score=score,
                    score_semantics='test_score',
                    score_kind='iou_proxy',
                )

        for boundary in (0, 1):
            output = AlignmentVerifierOutput(
                alignment_score=boundary,
                score_semantics='test_score',
                score_kind='iou_proxy',
            )
            self.assertEqual(output.alignment_score, float(boundary))

    def test_abstention_cannot_carry_score(self):
        output = AlignmentVerifierOutput.unknown(
            error='worker unavailable',
            metadata={'retryable': True},
        )
        self.assertTrue(output.abstained)
        self.assertIsNone(output.alignment_score)
        self.assertEqual(output.score_semantics, 'unavailable')
        with self.assertRaises(ValueError):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='test_score',
                score_kind='iou_proxy',
                abstained=True,
            )

    def test_valid_score_cannot_carry_error(self):
        with self.assertRaises(ValueError):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='test_score',
                score_kind='iou_proxy',
                error='contradictory error',
            )

    def test_score_semantics_and_schema_are_strict(self):
        with self.assertRaises(ValueError):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics=' ',
                score_kind='iou_proxy',
            )
        with self.assertRaises(ValueError):
            AlignmentVerifierOutput.from_dict({
                'verifier_output_schema': 'vocot_four_action_v1',
                'alignment_score': 0.5,
                'alignment_score_kind': 'iou_proxy',
                'score_semantics': 'test_score',
            })

    def test_new_scored_output_requires_one_valid_explicit_kind(self):
        with self.assertRaisesRegex(ValueError, 'must declare score_kind'):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='new_backend_free_form_score',
            )
        with self.assertRaisesRegex(ValueError, 'score_kind must be one of'):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='new_backend_score',
                score_kind='unknown_scale',
            )
        self.assertEqual(
            set(ALIGNMENT_SCORE_KINDS),
            {
                'calibrated_probability',
                'self_reported_probability',
                'iou_proxy',
                'hard_oracle_label',
            },
        )

    def test_hard_oracle_kind_accepts_only_binary_labels(self):
        for score in (0, 1):
            self.assertEqual(
                AlignmentVerifierOutput(
                    alignment_score=score,
                    score_semantics='oracle_test',
                    score_kind='hard_oracle_label',
                ).alignment_score,
                float(score),
            )
        with self.assertRaisesRegex(ValueError, 'exactly 0.0 or 1.0'):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='oracle_test',
                score_kind='hard_oracle_label',
            )

    def test_historical_wire_payload_infers_each_exact_known_kind(self):
        historical_scales = {
            'calibrated_alignment_probability': (
                'calibrated_probability', 0.31
            ),
            'qwen_self_reported_label_confidence_transformed_uncalibrated': (
                'self_reported_probability', 0.31
            ),
            'candidate_selected_grounding_iou_proxy_uncalibrated': (
                'iou_proxy', 0.31
            ),
            'oracle_hard_binary_alignment_label': (
                'hard_oracle_label', 1.0
            ),
        }
        for semantics, (expected_kind, score) in historical_scales.items():
            with self.subTest(semantics=semantics):
                old_payload = {
                    'verifier_output_schema': ALIGNMENT_OUTPUT_SCHEMA,
                    'alignment_score': score,
                    'score_semantics': semantics,
                    'abstained': False,
                    'error': None,
                    'metadata': {'backend': 'historical'},
                }
                output = AlignmentVerifierOutput.from_dict(old_payload)
                self.assertEqual(output.score_kind, expected_kind)
                self.assertEqual(output.alignment_score, score)
                # Re-serialization upgrades the payload without changing its
                # v1 schema.
                self.assertEqual(
                    output.as_dict()['alignment_score_kind'],
                    expected_kind,
                )

    def test_legacy_inference_is_exact_not_free_form_guessing(self):
        self.assertEqual(
            alignment_score_kind_from_legacy_semantics(
                'oracle_hard_binary_alignment_label'
            ),
            'hard_oracle_label',
        )
        self.assertIsNone(
            alignment_score_kind_from_legacy_semantics(
                'looks_like_an_iou_proxy_but_is_not_registered'
            )
        )
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            AlignmentVerifierOutput(
                alignment_score=0.5,
                score_semantics='calibrated_alignment_probability',
                score_kind='iou_proxy',
            )

    def test_backend_interface_uses_alignment_output(self):
        output = _Backend().verify_alignment(None)
        self.assertIsInstance(output, AlignmentVerifierOutput)


if __name__ == '__main__':
    unittest.main()
