"""Role-boundary tests preventing candidate/token/GT leakage."""

from dataclasses import fields
import unittest

from grounding_control.contracts import (
    CandidateAlignmentRequest,
    GroundingRequest,
    VerificationRequest,
)


class RoleRequestIsolationTests(unittest.TestCase):
    def setUp(self):
        self.request = VerificationRequest(
            sample_id='sample',
            grounding_step=2,
            object_reference='the cup',
            candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            candidate_coordinate_text='<coor>...</coor>',
            generated_ids=(1, 2, 3),
            candidate_span=(1, 2),
            sample_context={
                'image': object(),
                'image_path': '/tmp/source.png',
                'oracle_targets': [{'object': 'cup'}],
                'question': 'Where is the cup?',
            },
        )

    def test_alignment_request_has_only_candidate_aware_role_fields(self):
        sanitized = self.request.alignment_request()
        self.assertIsInstance(sanitized, CandidateAlignmentRequest)
        self.assertEqual(
            {field.name for field in fields(sanitized)},
            {
                'sample_id',
                'grounding_step',
                'object_reference',
                'candidate_bbox',
                'visual',
                'coordinate_system',
                'image_mode',
            },
        )
        self.assertFalse(hasattr(sanitized, 'generated_ids'))
        self.assertFalse(hasattr(sanitized, 'sample_context'))

    def test_grounder_request_cannot_receive_candidate_tokens_or_gt(self):
        sanitized = self.request.grounding_request()
        self.assertIsInstance(sanitized, GroundingRequest)
        self.assertEqual(
            {field.name for field in fields(sanitized)},
            {
                'sample_id',
                'grounding_step',
                'object_reference',
                'visual',
            },
        )
        for forbidden in (
            'candidate_bbox',
            'candidate_coordinate_text',
            'generated_ids',
            'candidate_span',
            'sample_context',
            'oracle_targets',
        ):
            self.assertFalse(hasattr(sanitized, forbidden))


if __name__ == '__main__':
    unittest.main()
