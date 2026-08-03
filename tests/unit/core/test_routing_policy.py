"""CPU-only tests for semantic routing and expert dispatch."""

import unittest

from grounding_control.contracts import (
    GrounderBackend,
    GroundingResult,
    VerificationRequest,
)
from grounding_control.four_way.contracts import ActionVerifierOutput
from grounding_control.contracts.errors import ExpertNotConfiguredError
from grounding_control.four_way.expert_dispatch import (
    FourWayExpertDispatcher as ExpertRouter,
)
from grounding_control.four_way.routing_policy import RoutingPolicy
from grounding_control.legacy.verdicts import (
    VerificationLookup,
    VerificationResult,
)


def _lookup(verdict, reason, confidence=1.0):
    return VerificationLookup(
        VerificationResult(verdict, reason, confidence)
    )


def _action(action, confidence=1.0):
    return ActionVerifierOutput(
        predicted_action=action,
        action_probabilities=None,
        confidence=confidence,
        metadata={'probability_source': 'test_hard_label'},
    )


def _request():
    return VerificationRequest(
        sample_id='sample',
        grounding_step=1,
        object_reference='the cup',
        candidate_bbox=(0.1, 0.2, 0.3, 0.4),
        candidate_coordinate_text='<coor>0.1,0.2,0.3,0.4</coor>',
        generated_ids=(1, 2),
        candidate_span=(0, 1),
    )


class _Grounder(GrounderBackend):
    def __init__(self):
        self.calls = 0

    def ground(self, request):
        self.calls += 1
        return GroundingResult(
            bbox=(0.4, 0.4, 0.6, 0.6),
            source='fake_grounder',
        )


class RoutingPolicyTests(unittest.TestCase):
    def test_semantic_action_mapping(self):
        policy = RoutingPolicy(confidence_threshold=0.8)
        cases = [
            (_action('no_action'), 'no_action'),
            (_action('relocate'), 'relocate'),
            (_action('expand'), 'expand'),
            (_action('tighten'), 'tighten'),
            (ActionVerifierOutput.unknown(), 'no_action'),
            # Legacy inputs remain accepted at the migration boundary.
            (_lookup('misaligned', 'unsupported'), 'no_action'),
        ]
        for output, expected in cases:
            self.assertEqual(policy.decide(output).action, expected)

    def test_threshold_and_configurable_unsupported(self):
        policy = RoutingPolicy(
            confidence_threshold=0.8,
            unsupported_action='abstain',
        )
        self.assertEqual(
            policy.decide(
                _action('relocate', 0.79)
            ).action,
            'no_action',
        )
        low_confidence = policy.decide(_action('relocate', 0.79))
        self.assertTrue(low_confidence.verifier_abstained)
        self.assertEqual(
            low_confidence.router_action,
            'low_confidence_fail_open',
        )
        self.assertEqual(
            policy.decide(
                _lookup('misaligned', 'unsupported', 0.9)
            ).action,
            'abstain',
        )

    def test_explicit_unsupported_policy_precedes_unknown_policy(self):
        policy = RoutingPolicy(
            unsupported_action='no_action',
            unknown_action='abstain',
        )
        self.assertEqual(
            policy.decide(
                _lookup('misaligned', 'unsupported', 0.9)
            ).action,
            'no_action',
        )
        self.assertEqual(
            policy.decide(ActionVerifierOutput.unknown()).action,
            'abstain',
        )

    def test_expert_router_dispatches_every_rejection_to_grounder(self):
        grounder = _Grounder()
        router = ExpertRouter(grounder=grounder)
        request = _request()

        wrong = _action('relocate')
        relocated = router.route(
            RoutingPolicy().decide(wrong),
            request,
            wrong,
        )
        self.assertEqual(relocated.expert_role, 'grounder')
        self.assertEqual(relocated.source, 'fake_grounder')
        self.assertEqual(grounder.calls, 1)

        direct = router.route_grounder(request, action='call_grounder')
        self.assertEqual(direct.expert_role, 'grounder')
        self.assertEqual(direct.action, 'call_grounder')
        self.assertEqual(grounder.calls, 2)

        partial = _action('expand')
        expanded = router.route(
            RoutingPolicy().decide(partial),
            request,
            partial,
        )
        self.assertEqual(expanded.expert_role, 'grounder')
        self.assertEqual(expanded.action, 'expand')
        self.assertEqual(grounder.calls, 3)

    def test_missing_grounder_is_explicit_for_every_rejection(self):
        output = _action('tighten')
        with self.assertRaises(ExpertNotConfiguredError):
            ExpertRouter().route(
                RoutingPolicy().decide(output),
                _request(),
                output,
            )


if __name__ == '__main__':
    unittest.main()
