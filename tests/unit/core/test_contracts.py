"""Tests for the canonical four-action verifier contract."""

import unittest

from grounding_control.four_way.adapters import (
    action_output_to_legacy_lookup,
    legacy_lookup_to_action_output,
)
from grounding_control.four_way.contracts import (
    ACTION_NAMES,
    ActionVerifierOutput,
)
from grounding_control.legacy.verdicts import (
    VerificationLookup,
    VerificationResult,
)


def _probabilities(winner):
    values = {action: 0.1 for action in ACTION_NAMES}
    values[winner] = 0.7
    return values


class ActionVerifierOutputTests(unittest.TestCase):
    def test_complete_four_way_probabilities_are_validated(self):
        output = ActionVerifierOutput(
            predicted_action='relocate',
            action_probabilities=_probabilities('relocate'),
            confidence=0.7,
        )
        self.assertEqual(output.predicted_action, 'relocate')
        self.assertEqual(
            ActionVerifierOutput.from_dict(output.as_dict()),
            output,
        )
        with self.assertRaises(ValueError):
            ActionVerifierOutput(
                predicted_action='relocate',
                action_probabilities={'relocate': 1.0},
                confidence=1.0,
            )
        with self.assertRaises(ValueError):
            ActionVerifierOutput(
                predicted_action='expand',
                action_probabilities=_probabilities('relocate'),
                confidence=0.7,
            )
        with self.assertRaises(ValueError):
            ActionVerifierOutput(
                predicted_action='relocate',
                action_probabilities=_probabilities('relocate'),
                confidence=0.6,
            )

    def test_abstention_is_not_a_fifth_action(self):
        output = ActionVerifierOutput.unknown(
            action_probabilities={
                'no_action': 0.26,
                'relocate': 0.25,
                'expand': 0.25,
                'tighten': 0.24,
            },
            confidence=0.26,
        )
        self.assertTrue(output.abstained)
        self.assertIsNone(output.predicted_action)
        with self.assertRaises(ValueError):
            ActionVerifierOutput(
                predicted_action='no_action',
                action_probabilities=_probabilities('no_action'),
                confidence=0.7,
                abstained=True,
            )

    def test_legacy_four_way_mapping_does_not_invent_probabilities(self):
        output = legacy_lookup_to_action_output(VerificationLookup(
            VerificationResult('misaligned', 'partial_coverage', 0.8)
        ))
        self.assertEqual(output.predicted_action, 'expand')
        self.assertIsNone(output.action_probabilities)
        self.assertEqual(
            output.metadata['probability_source'],
            'unavailable_legacy_hard_label',
        )

    def test_unsupported_and_unknown_abstain_by_default(self):
        unsupported = legacy_lookup_to_action_output(VerificationLookup(
            VerificationResult.unsupported(1.0)
        ))
        unknown = legacy_lookup_to_action_output(VerificationLookup(
            VerificationResult.unknown()
        ))
        self.assertTrue(unsupported.abstained)
        self.assertTrue(unknown.abstained)

    def test_round_trip_preserves_four_action_semantics(self):
        for action in ACTION_NAMES:
            original = ActionVerifierOutput(
                predicted_action=action,
                action_probabilities=_probabilities(action),
                confidence=0.7,
            )
            restored = legacy_lookup_to_action_output(
                action_output_to_legacy_lookup(original)
            )
            self.assertEqual(restored.predicted_action, action)


if __name__ == '__main__':
    unittest.main()
