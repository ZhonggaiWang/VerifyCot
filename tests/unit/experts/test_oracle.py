"""CPU-only tests for GT target resolution and oracle experts."""

import unittest
from types import SimpleNamespace

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import OnlineOracleCoordinateLogitsProcessor
from grounding_control.experts.grounders import OracleGrounderBackend
from grounding_control.oracle_targets import OracleTargetResolver
from grounding_control.four_way.verifiers import OracleIoUVerifierBackend
from grounding_control.four_way.contracts import ActionVerifierOutput
from grounding_control.contracts import (
    VerificationRequest,
)
from grounding_control.core import ExpertUnavailableError


class _Tokenizer:
    boc_id = 10
    eoc_id = 11
    words = {
        1: 'Find',
        2: 'the',
        3: 'cup',
        4: 'it',
        5: 'bottle',
        6: 'tissue',
        7: 'box',
        boc_id: DEFAULT_BOC_TOKEN,
        eoc_id: DEFAULT_EOC_TOKEN,
        99: '0.1,0.2,0.3,0.4',
    }

    def convert_tokens_to_ids(self, token):
        return {
            DEFAULT_BOC_TOKEN: self.boc_id,
            DEFAULT_EOC_TOKEN: self.eoc_id,
        }.get(token)

    def decode(self, token_ids, skip_special_tokens=False):
        return ' '.join(self.words.get(value, str(value)) for value in token_ids)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[99, self.eoc_id])


def _request(reference_token=3):
    return VerificationRequest(
        sample_id='sample',
        grounding_step=1,
        object_reference='Find the cup',
        candidate_bbox=(0.5, 0.5, 0.7, 0.7),
        candidate_coordinate_text='',
        generated_ids=(1, 2, reference_token, 10, 99, 11),
        candidate_span=(3, 5),
        sample_context={
            'oracle_targets': [{
                'object': 'cup',
                'aliases': ['cup'],
                'box': [0.1, 0.2, 0.3, 0.4],
            }],
        },
    )


def _verification(action):
    return ActionVerifierOutput(
        predicted_action=action,
        action_probabilities=None,
        confidence=1.0,
        metadata={'probability_source': 'test'},
    )


def _request_with(generated_ids, candidate_span, oracle_targets):
    return VerificationRequest(
        sample_id='sample',
        grounding_step=1,
        object_reference='local reference',
        candidate_bbox=(0.5, 0.5, 0.7, 0.7),
        candidate_coordinate_text='',
        generated_ids=tuple(generated_ids),
        candidate_span=tuple(candidate_span),
        sample_context={'oracle_targets': oracle_targets},
    )


class OracleExpertTests(unittest.TestCase):
    def setUp(self):
        self.resolver = OracleTargetResolver(_Tokenizer())
        self.grounder_resolver = OracleTargetResolver(
            _Tokenizer(),
            oracle_targets=[{
                'object': 'cup',
                'aliases': ['cup'],
                'box': [0.1, 0.2, 0.3, 0.4],
            }],
        )

    def test_grounder_resolves_target_independently_of_verifier(self):
        result = OracleGrounderBackend(self.grounder_resolver).ground(
            _request().grounding_request(),
        )
        self.assertEqual(result.bbox, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(result.metadata['target_object'], 'cup')

    def test_oracle_verifier_outputs_native_routing_actions(self):
        backend = OracleIoUVerifierBackend(
            resolver=self.resolver,
            iou_threshold=0.5,
        )
        relocate = backend.verify_action(_request())
        self.assertEqual(relocate.predicted_action, 'relocate')
        self.assertFalse(relocate.abstained)
        self.assertIsNone(relocate.action_probabilities)
        self.assertEqual(
            relocate.metadata['match_status'],
            'matched_unique_explicit_target',
        )
        self.assertEqual(relocate.metadata['legacy_reason'], 'wrong_object')

        aligned_request = VerificationRequest(**{
            **_request().__dict__,
            'candidate_bbox': (0.1, 0.2, 0.3, 0.4),
        })
        aligned = backend.verify_action(aligned_request)
        self.assertEqual(aligned.predicted_action, 'no_action')
        self.assertFalse(aligned.abstained)
        self.assertEqual(aligned.metadata['candidate_iou_to_gt'], 1.0)

    def test_oracle_verifier_abstains_when_reference_is_unmatched(self):
        backend = OracleIoUVerifierBackend(
            resolver=self.resolver,
            iou_threshold=0.5,
        )
        request = VerificationRequest(**{
            **_request().__dict__,
            'object_reference': 'it',
            'generated_ids': (1, 4, 10, 99, 11),
            'candidate_span': (2, 4),
        })
        output = backend.verify_action(request)
        self.assertTrue(output.abstained)
        self.assertIsNone(output.predicted_action)
        self.assertEqual(
            output.metadata['match_status'],
            'unverifiable_accept',
        )
        self.assertEqual(
            output.metadata['accept_router_action'],
            'unverifiable_accept',
        )

    def test_grounder_requires_resolver_and_uses_only_request_context(self):
        with self.assertRaises(ValueError):
            OracleGrounderBackend(None)

        corrected = OracleGrounderBackend(self.grounder_resolver).ground(
            _request().grounding_request(),
        )
        self.assertEqual(corrected.bbox, (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(
            corrected.metadata['oracle_target_box'],
            [0.1, 0.2, 0.3, 0.4],
        )
        self.assertNotIn('probability_source', corrected.metadata)

        request = VerificationRequest(**{
            **_request().__dict__,
            'object_reference': 'it',
            'generated_ids': (1, 4, 10, 99, 11),
            'candidate_span': (2, 4),
        })
        with self.assertRaises(ExpertUnavailableError):
            OracleGrounderBackend(self.grounder_resolver).ground(
                request.grounding_request()
            )

    def test_unmatched_reference_is_explicitly_unavailable(self):
        request = _request(reference_token=4)
        request = VerificationRequest(
            **{
                **request.__dict__,
                'object_reference': 'it',
                'generated_ids': (1, 4, 10, 99, 11),
                'candidate_span': (2, 4),
            }
        )
        with self.assertRaises(ExpertUnavailableError):
            OracleGrounderBackend(self.grounder_resolver).ground(
                request.grounding_request(),
            )

    def test_latest_alias_wins(self):
        request = _request_with(
            [1, 2, 3, 5, 10, 99, 11],
            (4, 6),
            [
                {'object': 'cup', 'box': [0.1, 0.1, 0.2, 0.2]},
                {'object': 'bottle', 'box': [0.3, 0.3, 0.4, 0.4]},
            ],
        )
        resolution = self.resolver.resolve(request)
        self.assertTrue(resolution.matched)
        self.assertEqual(resolution.target_object, 'bottle')

    def test_longest_alias_wins_at_same_position(self):
        request = _request_with(
            [1, 2, 6, 7, 10, 99, 11],
            (4, 6),
            [
                {'object': 'box', 'box': [0.1, 0.1, 0.2, 0.2]},
                {
                    'object': 'tissue box',
                    'box': [0.3, 0.3, 0.4, 0.4],
                },
            ],
        )
        resolution = self.resolver.resolve(request)
        self.assertTrue(resolution.matched)
        self.assertEqual(resolution.target_object, 'tissue box')
        self.assertEqual(resolution.matched_alias, 'tissue box')

    def test_reference_does_not_cross_previous_coordinate(self):
        request = _request_with(
            [1, 2, 3, 10, 99, 11, 4, 10, 99, 11],
            (7, 9),
            [{'object': 'cup', 'box': [0.1, 0.1, 0.2, 0.2]}],
        )
        resolution = self.resolver.resolve(request)
        self.assertFalse(resolution.matched)
        self.assertEqual(
            resolution.reason,
            'no_explicit_target_alias',
        )

    def test_shared_alias_across_instances_is_ambiguous_not_exception(self):
        targets = [
            {'object': 'cup', 'box': [0.1, 0.1, 0.2, 0.2]},
            {'object': 'cup', 'box': [0.6, 0.6, 0.8, 0.8]},
        ]
        request = _request_with(
            [1, 2, 3, 10, 99, 11],
            (3, 5),
            targets,
        )
        resolution = self.resolver.resolve(request)
        self.assertFalse(resolution.matched)
        self.assertEqual(
            resolution.reason,
            'ambiguous_explicit_target_alias',
        )

        # Online forcing uses the identical matcher and must fail open too.
        processor = OnlineOracleCoordinateLogitsProcessor(
            _Tokenizer(),
            prompt_length=0,
            oracle_targets=targets,
        )
        decision = processor.decision_for_coordinate(
            request.generated_ids,
            request.candidate_span[0],
            request.grounding_step,
        )
        self.assertEqual(decision['decision'], 'kept_model_box')
        self.assertEqual(
            decision['reason'],
            'ambiguous_explicit_target_alias',
        )

    def test_exact_duplicate_target_records_are_merged(self):
        duplicate = {
            'object': 'cup',
            'aliases': ['the cup'],
            'box': [0.1, 0.1, 0.2, 0.2],
        }
        request = _request_with(
            [1, 2, 3, 10, 99, 11],
            (3, 5),
            [duplicate, dict(duplicate)],
        )
        resolution = self.resolver.resolve(request)
        self.assertTrue(resolution.matched)
        self.assertEqual(resolution.target_object, 'cup')


if __name__ == '__main__':
    unittest.main()
